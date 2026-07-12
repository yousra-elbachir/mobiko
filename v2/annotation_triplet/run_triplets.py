"""Headless, resumable runner for MoBiKo v2 JOINT triplet extraction.

For each sentence-per-line ``.txt`` file in ``data/mark_sentences/`` (one file per
scientific paper, named ``<pmcid>.txt``), extract (subject, relation, object)
triplets with typed entities using ``PROMPT_TRIPLET`` on the Qwen model
served by SwissAI (see MODEL_NAME below), and write one result file per paper to
``outputs/triplets/<SOURCE>/{pmcid}.json`` in the target structure (matches
``annotation_relation/prompt_relation.py``, plus a per-sentence ``spans`` union).

``SOURCE`` names where the sentences came from ("mark" here); keeping it as a
folder level lets a later sentence set from someone else live side by side under
``outputs/triplets/`` without collisions.

Crash-safe & resumable: results flush after every sentence (atomic write); a
per-paper ``.{pmcid}.done`` marker lets a restart skip finished papers, and a
partially-written paper resumes at its first unprocessed sentence. Run from this
folder:  python run_triplets.py
"""
import glob
import json
import os
import re
import time
import traceback
import unicodedata

from openai import OpenAI
from prompt_triplet import PROMPT_TRIPLET
from utilities import get_file_contents, extract_triplets_joint

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

SOURCE = "mark"                                   # sentence source -> outputs/triplets/<SOURCE>/
IN_DIR = "data/mark_sentences"                    # dir of <pmcid>.txt, one sentence per line (one file per paper)
OUT_DIR = os.path.join("outputs", "triplets", SOURCE)
TOKEN = "../../tokens/swissAI_key.rtf"
MODEL_NAME = "Qwen/Qwen3-32B-lwvY"
BASE_URL = "https://api.swissai.svc.cscs.ch/v1"
MAX_CRASH_RETRIES = 100


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _norm(t):
    t = unicodedata.normalize("NFKC", t).replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t).strip().lower()


def load_sentences(path):
    """One non-empty line -> one sentence dict (order = file order)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append({"text": s})
    return out


def main():
    os.environ["CSCS_SERVING_API"] = get_file_contents(TOKEN)
    model = {
        "name": MODEL_NAME,
        "client": OpenAI(api_key=os.environ["CSCS_SERVING_API"], base_url=BASE_URL),
    }
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(IN_DIR, "*.txt")))
    if not files:
        log(f"no input files in {IN_DIR} — add <pmcid>.txt files (one sentence per line) there first")
        return 1
    log(f"{len(files)} paper file(s) in {IN_DIR} -> {OUT_DIR}")

    all_done = True
    for path in files:
        pmcid = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(OUT_DIR, f"{pmcid}.json")
        marker = os.path.join(OUT_DIR, f".{pmcid}.done")
        if os.path.exists(marker):
            log(f"{pmcid}: already complete (marker present) — skipping")
            continue

        sentences = load_sentences(path)
        if not sentences:
            log(f"{pmcid}: empty file — skipping")
            open(marker, "w").write("done\n")
            continue

        # Resume: match already-saved sentences back to their index (robust to
        # any whitespace/unicode drift) and only process the rest.
        exact = {s["text"]: i for i, s in enumerate(sentences)}
        normd = {_norm(s["text"]): i for i, s in enumerate(sentences)}

        def text_to_idx(t):
            i = exact.get(t)
            return i if i is not None else normd.get(_norm(t))

        done_paper = False
        for attempt in range(1, MAX_CRASH_RETRIES + 1):
            try:
                existing = json.load(open(out_path)) if os.path.exists(out_path) else []
                seed_by_idx = {}
                for o in existing:
                    i = text_to_idx(o.get("text", ""))
                    if i is not None:
                        seed_by_idx[i] = o
                todo = [i for i in range(len(sentences)) if i not in seed_by_idx]
                log(f"{pmcid}: attempt {attempt} — {len(todo)} of {len(sentences)} sentences to do "
                    f"({len(seed_by_idx)} already saved)")
                res = extract_triplets_joint(
                    model, PROMPT_TRIPLET, sentences,
                    indices=todo, trace=True, save_path=out_path, seed_by_idx=seed_by_idx)
                n_trip = sum(len(o["triplets"]) for o in res.values())
                open(marker, "w").write("done\n")
                log(f"{pmcid}: COMPLETE — {len(res)} sentences, {n_trip} triplets -> {out_path}")
                done_paper = True
                break
            except Exception as e:
                log(f"{pmcid}: attempt {attempt} crashed: {type(e).__name__}: {e}")
                log(traceback.format_exc())
                time.sleep(15)
        if not done_paper:
            all_done = False
            log(f"{pmcid}: GAVE UP after {MAX_CRASH_RETRIES} attempts — needs attention")

    log("ALL PAPERS COMPLETE" if all_done else "FINISHED WITH UNRESOLVED PAPERS")
    return 0 if all_done else 1


if __name__ == "__main__":
    raise SystemExit(main())
