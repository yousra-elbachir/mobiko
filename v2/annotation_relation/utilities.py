"""Engine for MoBiKo relation extraction.

Given expert-annotated entities per sentence, an LLM assigns predicates between
them (the 9-label schema in documentation/Relation_guideline). Only what the
relation pipeline uses lives here: robust model calls, JSON parsing, and the
extraction driver.
"""
import json
import os
import re
import time
from typing import Optional

import httpx
from openai import (
    APIError,
    AuthenticationError, NotFoundError, BadRequestError, PermissionDeniedError,
)

# A streamed Qwen call is retried INDEFINITELY — we never skip a sentence. This
# only caps the exponential back-off (in seconds) between successive attempts,
# so a cold/over-loaded gateway is retried patiently rather than abandoned.
MAX_BACKOFF_SECONDS = 60

# The controlled relation vocabulary (documentation/Relation_guideline).
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

    Returns the parsed dict (which may legitimately be ``{}`` for a sentence with
    no relations). Tries, in order: the raw text, the text with ```json fences
    stripped, and — for reasoning models that prepend a "Thinking Process:" dump —
    the text with any <think>…</think> removed and only the final JSON object kept.
    Raises json.JSONDecodeError if none of these yield valid JSON.
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
def _complete_json(model, prompt, label="request", trace=False):
    """Send ``prompt`` to the SwissAI/Qwen (OpenAI-compatible) client and return
    its reply parsed as a JSON dict.

    The reply is STREAMED and retried indefinitely with capped back-off through
    mid-stream drops, empty replies, and unparseable JSON; the parser tolerates a
    reasoning-model "Thinking Process:" preamble (see ``_parse_model_json``).
    Permanent errors (bad key / model id / request) surface immediately instead of
    spinning forever. Tokens are accumulated silently — never printed. ``label``
    only labels trace messages.

    ``model`` is a spec dict ``{'client': <OpenAI SDK client>, 'name': <model id>}``.
    """
    client = model['client']
    model_name = model['name']
    if not hasattr(client, "chat"):
        raise ValueError(
            f"Expected an OpenAI-compatible client with a .chat interface, "
            f"got {type(client).__name__}")

    # max_retries does NOT cover mid-stream drops / empty / garbled replies;
    # retry the whole streamed call indefinitely until we get parseable JSON.
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
            _parse_model_json(candidate)  # validate now; raises if unparseable
            raw_text = candidate
            break
        except (AuthenticationError, NotFoundError, BadRequestError,
                PermissionDeniedError):
            # Permanent (bad key / model id / request) — do NOT retry forever;
            # let it surface so a misconfig fails fast instead of spinning overnight.
            raise
        except (httpx.HTTPError, APIError, json.JSONDecodeError, ValueError) as e:
            # Transient: mid-stream drop, ReadTimeout/ConnectError, rate limit,
            # 5xx, empty/garbled reply. Retry the whole call indefinitely with
            # capped back-off — a disconnection never loses a sentence.
            wait = min(2 ** min(attempt, 6), MAX_BACKOFF_SECONDS)
            if trace:
                print(f"!! {label} attempt {attempt} failed "
                      f"({type(e).__name__}: {e}). retrying in {wait}s — never skipping")
            time.sleep(wait)

    try:
        return _parse_model_json(raw_text)
    except json.JSONDecodeError:
        if trace:
            print(f"!! Failed to parse JSON response for {label}")
        return None


# --------------------------------------------------------------------------- #
# Relation extraction
# --------------------------------------------------------------------------- #
def _format_entities(entities) -> str:
    """One entity per line, exposing the exact fields the model copies verbatim
    into each triplet (text, type, and character offsets)."""
    return "\n".join(
        f'- text="{e["text"]}" | type="{e["type"]}" '
        f'| start_char={e.get("start_char")} | end_char={e.get("end_char")}'
        for e in entities)


def build_relation_prompt(prompt_template, sentence, entities, prior_context=None):
    clean_context = prior_context.strip() if prior_context else "None. This is the beginning of the text segment."
    p = prompt_template.replace("[INSERT SENTENCE HERE]", sentence.strip())
    p = p.replace("[INSERT CONTEXT HERE]", clean_context)
    p = p.replace("[INSERT ENTITIES HERE]", _format_entities(entities))
    return p


def _span_out(e):
    """The per-sentence `spans` entry: the expert entity, verbatim."""
    return {"text": e["text"], "type": e["type"],
            "start_char": e.get("start_char"), "end_char": e.get("end_char"),
            "uid": e.get("uid")}


def extract_relations(model, PROMPT, sentences, entities_by_idx, indices,
                      trace=False, save_path=None, seed_by_idx=None):
    """Relation extraction over already-annotated entities (MoBiKo relation schema).

    For each sentence index in ``indices``, the expert entities in
    ``entities_by_idx[idx]`` (dicts with text/type/start_char/end_char/uid) are
    listed in the prompt with their spans; the model returns triplets in the target
    structure:
        { "text": ..., "triplets": [ {subject, subject_type, subject_start_char,
          subject_end_char, relation, predicate, object, object_type,
          object_start_char, object_end_char}, ... ] }
    where ``relation`` is the named phrase from the sentence and ``predicate`` is its
    type (one of the 9 schema labels).
    Each triplet's subject/object is matched back to a GIVEN entity (by char span,
    else text) and every field is SNAPPED to the expert value; the predicate must be
    in the allowed set; subject and object must differ; exact duplicates are dropped.
    The result is assembled into the target per-sentence object:
        { "text": <sentence>,
          "spans": [ {text,type,start_char,end_char,uid}, ... ],   # the annotator's entities
          "triplets": [ {subject, subject_type, subject_start_char, subject_end_char,
                         predicate, object, object_type, object_start_char, object_end_char}, ... ] }
    subject/object text, types, and char spans are copied VERBATIM from the expert
    entity (the model only supplies the predicate). A sentence with < 2 entities is
    still emitted, with its spans and an empty triplets list. Multiple predicates
    per ordered pair are allowed (one triplet each).

    Accumulates a dict {idx: object}; when ``save_path`` is set it flushes the
    idx-sorted LIST after each sentence (crash-safe). ``seed_by_idx`` resumes a run
    without redoing finished sentences. Returns the {idx: object} dict.
    """
    results = dict(seed_by_idx) if seed_by_idx else {}
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    def flush():
        if save_path:
            _atomic_dump([results[i] for i in sorted(results)], save_path)

    for s in indices:
        ents = entities_by_idx.get(s, [])
        obj = {"text": sentences[s]["text"],
               "spans": [_span_out(e) for e in ents],
               "triplets": []}
        if len(ents) >= 2:
            prompt = build_relation_prompt(PROMPT, sentences[s]["text"], ents,
                                           sentences[s - 1]["text"] if s > 0 else None)
            reply = _complete_json(model, prompt, label=f"sentence {s}", trace=trace)
            # The model returns {"text": ..., "triplets": [ {9 fields} ]} (tolerate a
            # bare list too). Each subject/object is matched back to a GIVEN entity
            # and every field is SNAPPED to the expert value — so a miscopied span
            # or type is corrected, and any entity not in the list is dropped.
            if isinstance(reply, dict):
                raw_triplets = reply.get("triplets", [])
            elif isinstance(reply, list):
                raw_triplets = reply
            else:
                raw_triplets = []

            def _nt(x):
                return re.sub(r"\s+", " ", str(x or "")).strip().lower()
            by_span, by_text = {}, {}
            for e in ents:
                by_span.setdefault((e.get("start_char"), e.get("end_char")), e)
                by_text.setdefault(_nt(e["text"]), e)

            def _match(t, role):
                try:
                    key = (int(t.get(f"{role}_start_char")), int(t.get(f"{role}_end_char")))
                except (TypeError, ValueError):
                    key = None
                return by_span.get(key) or by_text.get(_nt(t.get(role)))

            seen = set()
            for t in raw_triplets:
                if not isinstance(t, dict):
                    continue
                pred = str(t.get("predicate", "")).strip().upper()
                if pred not in ALLOWED_PREDICATES:
                    continue
                a, b = _match(t, "subject"), _match(t, "object")
                if a is None or b is None or a is b:
                    continue
                relation = str(t.get("relation", "")).strip()  # named phrase (free text, not snapped)
                key = (a.get("start_char"), a.get("end_char"),
                       relation.lower(), pred,
                       b.get("start_char"), b.get("end_char"))
                if key in seen:          # drop exact duplicate (same pair + relation + predicate)
                    continue
                seen.add(key)
                obj["triplets"].append({
                    "subject": a["text"], "subject_type": a["type"],
                    "subject_start_char": a.get("start_char"), "subject_end_char": a.get("end_char"),
                    "relation": relation,
                    "predicate": pred,
                    "object": b["text"], "object_type": b["type"],
                    "object_start_char": b.get("start_char"), "object_end_char": b.get("end_char"),
                })
        results[s] = obj
        flush()
        if trace:
            print(f" ---- sentence {s}: {len(ents)} entities -> {len(obj['triplets'])} triplet(s)")
            for t in obj["triplets"]:
                print(f"      {t['subject']} --{t['relation']} [{t['predicate']}]--> {t['object']}")
    return results
