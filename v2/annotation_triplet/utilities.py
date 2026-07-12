"""Engine for MoBiKo JOINT triplet extraction.

For each raw sentence, an LLM extracts (subject, relation, object) triplets with
typed entities in one pass; the entity char-spans are resolved deterministically
from the sentence (never trusted from the model). Only what ``run_triplets.py``
uses lives here.
"""
import json
import os
import re
import time
from typing import Optional

import httpx
from openai import APIConnectionError, APIError

# Local helpers: normalize_type (controlled 16-label vocabulary) and resolve_span
# (entity text -> punctuation-trimmed char offsets).
from ontology import normalize_type
from span_utils import resolve_span

# A streamed Qwen call is retried INDEFINITELY — we never skip a sentence. This
# only caps the exponential back-off (in seconds) between successive attempts,
# so a cold/over-loaded gateway is retried patiently rather than abandoned.
MAX_BACKOFF_SECONDS = 60

# Controlled relation vocabulary (documentation/Relation_guideline). The joint
# prompt must emit exactly one of these per triplet; anything else is dropped.
ALLOWED_PREDICATES = {
    "IS_AFFECTING", "RELATED_TO", "LOCATED_IN", "COMPARES_TO", "HAS_PROPERTY",
    "HAS_PROCESS", "CAUSES", "DURING", "IS_PART_OF",
}


# --------------------------------------------------------------------------- #
# JSON parsing (tolerant of reasoning-model preambles)
# --------------------------------------------------------------------------- #
def _last_json_object(text: str) -> Optional[str]:
    """Return the last balanced ``{...}`` block in ``text`` (or None).

    Reasoning models (e.g. Qwen on SwissAI) emit a "Thinking Process:" analysis
    BEFORE the answer, and that prose can itself contain example braces. The real
    answer is the final JSON object, so we scan from the last ``}`` backwards to
    its matching ``{`` and return that span.
    """
    end = text.rfind("}")
    if end == -1:
        return None
    depth = 0
    for i in range(end, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                return text[i:end + 1]
    return None


def _parse_model_json(raw_text: str) -> dict:
    """Parse the model's JSON reply, tolerating fences and reasoning preambles.

    Tries, in order: the raw text, the text with ```json fences stripped, and —
    for reasoning models that prepend a "Thinking Process:" dump — the text with
    any <think>…</think> removed and only the final JSON object kept. Raises
    json.JSONDecodeError if none of these yield valid JSON.
    """
    s = raw_text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    clean = s.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    no_think = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    obj = _last_json_object(no_think)
    if obj is not None:
        return json.loads(obj)  # propagates JSONDecodeError if still invalid
    return json.loads(clean)    # propagates JSONDecodeError


# --------------------------------------------------------------------------- #
# Token file + atomic write
# --------------------------------------------------------------------------- #
def get_file_contents(filename):
    """Extract an API token from a (possibly RTF-wrapped) key file.

    Tries the known key shapes in order: a dotted token (e.g. a JWT
    "aaa.bbb.ccc", as in the SDSC Qwen token), then an OpenAI-style "sk-..." key
    (as in the SwissAI key). Falls back to the stripped file contents.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        raw_rtf = f.read()
    for pattern in (r'[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+',
                    r'sk-[A-Za-z0-9-]+'):
        match = re.search(pattern, raw_rtf)
        if match:
            return match.group(0)
    return raw_rtf.strip()


def _atomic_dump(obj, path):
    """Write ``obj`` to ``path`` as JSON atomically (write-temp-then-rename).

    A crash mid-write therefore never leaves a half-written, unparseable file:
    the previous complete version stays in place until the new one is ready.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)  # atomic on the same filesystem


# --------------------------------------------------------------------------- #
# Model call (streamed, retried, silent)
# --------------------------------------------------------------------------- #
def generate_extraction_prompt_with_context(prompt_template: str, sentence: str,
                                            prior_context: Optional[str] = None) -> str:
    """Insert the target sentence and its preceding-sentence context into the prompt."""
    clean_sentence = sentence.strip()
    clean_context = prior_context.strip() if prior_context else "None. This is the beginning of the text segment."
    p = prompt_template.replace("[INSERT SENTENCE HERE]", clean_sentence)
    p = p.replace("[INSERT CONTEXT HERE]", clean_context)
    return p


def _extract_one_sentence(model, PROMPT, sentences, s, trace=False):
    """Call Qwen for a SINGLE sentence at global index ``s`` and return its reply
    parsed as a dict.

    The preceding sentence is injected as pronoun-resolution context. The reply is
    STREAMED and retried indefinitely with capped back-off through mid-stream
    drops, empty replies, and unparseable JSON — a sentence is never skipped.
    Tokens are accumulated silently (never printed), so a reasoning model's
    "Thinking Process:" dump stays off the console.

    ``model`` is a spec dict ``{'client': <OpenAI-compatible client>, 'name': <model id>}``.
    """
    client = model['client']
    model_name = model['name']
    sentence = sentences[s]['text']
    prior_context = sentences[s - 1]['text'] if s > 0 else None
    prompt = generate_extraction_prompt_with_context(PROMPT, sentence, prior_context)

    # max_retries does NOT cover mid-stream drops / empty / garbled replies; retry
    # the whole streamed call indefinitely until we get a parseable JSON reply.
    attempt = 0
    while True:
        attempt += 1
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,  # first bytes arrive before the activator times out (cold start)
            )
            chunks = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            candidate = "".join(chunks).strip()
            if not candidate:
                raise ValueError("empty response from gateway")
            return _parse_model_json(candidate)  # raises on unparseable JSON -> retry
        except (httpx.HTTPError, APIConnectionError, APIError,
                json.JSONDecodeError, ValueError) as e:
            wait = min(2 ** min(attempt, 6), MAX_BACKOFF_SECONDS)  # cap back-off
            if trace:
                print(f"!! sentence {s} attempt {attempt} failed "
                      f"({type(e).__name__}: {e}). retrying in {wait}s — never skipping")
            time.sleep(wait)


# --------------------------------------------------------------------------- #
# JOINT triplet extraction (entities + relations in one pass)
# --------------------------------------------------------------------------- #
def _nt(x) -> str:
    """Whitespace-collapsed, lower-cased text key for de-duplication / identity."""
    return re.sub(r"\s+", " ", str(x or "")).strip().lower()


def _resolve_entity(text, etype, sentence) -> dict:
    """Turn a model-proposed (text, type) into a located, typed entity.

    Character offsets come from ``resolve_span`` (exact -> case-insensitive ->
    fuzzy token overlap), never from the model — so a hallucinated offset can't
    leak in. ``match_status`` records how the span was found; an unresolved span
    keeps offsets -1 for a human to fix. The type is folded onto the controlled set.
    """
    res = resolve_span(text, sentence)
    return {
        "text": res["text"] or str(text or "").strip(),
        "type": normalize_type(str(etype or "")),
        "start_char": res["start_char"],
        "end_char": res["end_char"],
        "match_status": res["status"],
    }


def extract_triplets_joint(model, PROMPT, sentences, indices, trace=False,
                           save_path=None, seed_by_idx=None):
    """JOINT (entity + relation) triplet extraction over raw sentences.

    No entities are given: for each sentence in ``indices`` the model reads the
    sentence (plus the preceding sentence as context, injected by
    ``_extract_one_sentence``) and returns
        { "text": ..., "triplets": [ {subject, subject_type, relation, predicate,
                                       object, object_type}, ... ] }
    Here we:
      - drop any triplet whose predicate is not in ``ALLOWED_PREDICATES``;
      - locate each subject/object in the sentence with ``resolve_span`` and fill
        the four ``*_char`` fields, tagging the span's ``match_status``;
      - drop self-relations (subject text == object text) and exact duplicates.
    Each sentence is assembled into the per-sentence object
        { "text": <sentence>,
          "spans": [ {text,type,start_char,end_char,match_status}, ... ],
          "triplets": [ {subject, subject_type, subject_start_char, subject_end_char,
                         relation, predicate, object, object_type,
                         object_start_char, object_end_char}, ... ] }.

    Accumulates a dict ``{idx: object}``; with ``save_path`` set it flushes the
    idx-sorted LIST atomically after each sentence (crash-safe). ``seed_by_idx``
    resumes a run without redoing finished sentences. Returns the ``{idx: object}``
    dict.
    """
    results = dict(seed_by_idx) if seed_by_idx else {}
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    def flush():
        if save_path:
            _atomic_dump([results[i] for i in sorted(results)], save_path)

    for s in indices:
        sentence = sentences[s]["text"]
        reply = _extract_one_sentence(model, PROMPT, sentences, s, trace=trace)
        if isinstance(reply, dict):
            raw_triplets = reply.get("triplets", [])
        elif isinstance(reply, list):
            raw_triplets = reply
        else:
            raw_triplets = []

        triplets, spans_by_key, seen = [], {}, set()
        for t in raw_triplets:
            if not isinstance(t, dict):
                continue
            pred = str(t.get("predicate", "")).strip().upper().replace(" ", "_")
            if pred not in ALLOWED_PREDICATES:
                continue
            if not (t.get("subject") and t.get("object")):
                continue
            subj = _resolve_entity(t.get("subject"), t.get("subject_type"), sentence)
            obj = _resolve_entity(t.get("object"), t.get("object_type"), sentence)
            if _nt(subj["text"]) == _nt(obj["text"]):      # subject and object must differ
                continue
            relation = str(t.get("relation", "")).strip()  # free phrase, kept as-is
            key = (subj["start_char"], subj["end_char"], _nt(subj["text"]),
                   relation.lower(), pred,
                   obj["start_char"], obj["end_char"], _nt(obj["text"]))
            if key in seen:                                # drop exact duplicate
                continue
            seen.add(key)
            triplets.append({
                "subject": subj["text"], "subject_type": subj["type"],
                "subject_start_char": subj["start_char"], "subject_end_char": subj["end_char"],
                "relation": relation,
                "predicate": pred,
                "object": obj["text"], "object_type": obj["type"],
                "object_start_char": obj["start_char"], "object_end_char": obj["end_char"],
            })
            for e in (subj, obj):
                sk = (e["start_char"], e["end_char"], _nt(e["text"]), e["type"])
                spans_by_key.setdefault(sk, {
                    "text": e["text"], "type": e["type"],
                    "start_char": e["start_char"], "end_char": e["end_char"],
                    "match_status": e["match_status"],
                })
        # spans in document order; unresolved (-1) sink to the end.
        spans = sorted(spans_by_key.values(),
                       key=lambda e: (e["start_char"] if e["start_char"] >= 0 else 1 << 30,
                                      e["end_char"]))
        results[s] = {"text": sentence, "spans": spans, "triplets": triplets}
        flush()
        if trace:
            print(f" ---- sentence {s}: {len(spans)} entities -> {len(triplets)} triplet(s)")
            for t in triplets:
                print(f"      {t['subject']} [{t['subject_type']}] "
                      f"--{t['relation']} [{t['predicate']}]--> "
                      f"{t['object']} [{t['object_type']}]")
    return results
