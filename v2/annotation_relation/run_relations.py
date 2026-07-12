"""Headless, self-healing runner for MoBiKo v3 relation extraction.

Standalone (no Jupyter kernel to disconnect). Processes BOTH annotators over the
overlap sentences, resuming after any crash. Per-annotator completion is marked
with a .done file so a full restart skips finished annotators. Run from this
folder:  python run_relations.py
"""
import os, re, json, time, unicodedata, traceback

import utilities
from openai import OpenAI
from prompt_relation import PROMPT_RELATION
from utilities import get_file_contents, extract_relations

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

MAX_CRASH_RETRIES = 100
SOURCE = './data/LASTEST_qwen32_full_C1_abstracts_Yousra.jsonl'
OVERLAP = './data/overlap_sentences.txt'
ANNOTATORS = {
    'davnah': './data/combined_M_D_Davnah_postprocessed.json',
    'mark':   './data/combined_M_D_Mark_postprocessed.json',
}
OUT_DIR = './outputs/relations'


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _norm(t):
    t = unicodedata.normalize('NFKC', t).replace('’', "'").replace('“', '"').replace('”', '"')
    return re.sub(r'\s+', ' ', t).strip().lower()


def load_entities(path):
    sents = json.load(open(path))['sentences']
    by_idx = {}
    for i, s in enumerate(sents):
        ents = [e for e in s.get('spans', [])
                if (e.get('text') or '').strip()
                and e.get('start_char') is not None and e.get('end_char') is not None]
        if ents:
            by_idx[i] = ents
    return by_idx


def main():
    sentences = json.load(open(SOURCE))['sentences']
    exact = {s['text']: i for i, s in enumerate(sentences)}
    normd = {_norm(s['text']): i for i, s in enumerate(sentences)}
    def text_to_idx(t):
        i = exact.get(t)
        return i if i is not None else normd.get(_norm(t))

    overlap_lines = [l.rstrip('\n') for l in open(OVERLAP) if l.strip()]
    overlap_idx, unmatched = [], []
    for t in overlap_lines:
        i = text_to_idx(t)
        (overlap_idx.append(i) if i is not None else unmatched.append(t))
    overlap_idx = sorted(set(overlap_idx))
    assert not unmatched, f"unmatched overlap sentences: {unmatched[:3]}"
    log(f"{len(overlap_lines)} overlap lines -> {len(overlap_idx)} source sentences")

    os.environ['CSCS_SERVING_API'] = get_file_contents('../../tokens/swissAI_key.rtf')
    model = {'name': 'Qwen/Qwen3.5-27B',
             'client': OpenAI(api_key=os.environ['CSCS_SERVING_API'],
                              base_url='https://api.swissai.svc.cscs.ch/v1')}

    os.makedirs(OUT_DIR, exist_ok=True)
    all_done = True
    for who, path in ANNOTATORS.items():
        out_path = os.path.join(OUT_DIR, f'relations_{who}.json')
        marker = os.path.join(OUT_DIR, f'.{who}.done')
        if os.path.exists(marker):
            log(f"{who}: already complete (marker present) — skipping")
            continue

        entities_by_idx = load_entities(path)
        done_ann = False
        for attempt in range(1, MAX_CRASH_RETRIES + 1):
            try:
                existing = json.load(open(out_path)) if os.path.exists(out_path) else []
                seed_by_idx = {}
                for o in existing:
                    i = text_to_idx(o['text'])
                    if i is not None:
                        seed_by_idx[i] = o
                todo = [i for i in overlap_idx if i not in seed_by_idx]
                log(f"{who}: attempt {attempt} — {len(todo)} sentences to process "
                    f"({len(seed_by_idx)} already saved)")
                res = extract_relations(
                    model, PROMPT_RELATION, sentences, entities_by_idx,
                    indices=todo, trace=True, save_path=out_path, seed_by_idx=seed_by_idx)
                n_trip = sum(len(o['triplets']) for o in res.values())
                open(marker, 'w').write('done\n')
                log(f"{who}: COMPLETE — {len(res)} sentences, {n_trip} triplets -> {out_path}")
                done_ann = True
                break
            except Exception as e:
                log(f"{who}: attempt {attempt} crashed: {type(e).__name__}: {e}")
                log(traceback.format_exc())
                time.sleep(15)
        if not done_ann:
            all_done = False
            log(f"{who}: GAVE UP after {MAX_CRASH_RETRIES} attempts — needs attention")

    log("ALL ANNOTATORS COMPLETE" if all_done else "FINISHED WITH UNRESOLVED ANNOTATORS")
    return 0 if all_done else 1


if __name__ == "__main__":
    raise SystemExit(main())
