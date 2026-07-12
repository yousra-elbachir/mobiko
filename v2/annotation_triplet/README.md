# MoBiKo — Joint Triplet Extraction

Extract **(subject, relation, object)** triplets with typed entities from scientific
sentences (mountain biodiversity / ecology / geology), in a single LLM pass. For
each sentence the model proposes triplets; entity **character spans** are then
resolved deterministically from the sentence text (never trusted from the model),
and predicates are validated against a controlled 9-relation schema.

## Layout

```
run_triplets.py     # headless, resumable runner (entry point)
prompt_triplet.py   # the extraction prompt (PROMPT_TRIPLET)
utilities.py        # engine: model call, JSON parsing, extract_triplets_joint
ontology.py         # controlled entity-type vocabulary + normalize_type
span_utils.py       # resolve_span (entity text -> char offsets)
documentation/      # annotation guidelines & schemas (reference)
data/               # input sentences (gitignored — provide your own)
outputs/            # results (gitignored)
```

## Requirements

Python 3.12 and the OpenAI SDK (its dependency `httpx` comes along):

```bash
pip install openai
```

## Setup

1. **API key.** The runner talks to an OpenAI-compatible gateway (SwissAI/CSCS by
   default). Put your key in a file and point `TOKEN` at it in `run_triplets.py`.
   `get_file_contents` extracts the key from a plain-text or RTF file (it matches a
   JWT-style or `sk-...` token). Default: `TOKEN = "../../tokens/swissAI_key.rtf"`.
2. **Model / endpoint.** Set `MODEL_NAME` and `BASE_URL` in `run_triplets.py`
   (defaults target `Qwen/...` on `https://api.swissai.svc.cscs.ch/v1`).
3. **Input sentences.** Create `data/<SOURCE>_sentences/<pmcid>.txt`, **one
   sentence per line**, one file per paper. Set `SOURCE` / `IN_DIR` accordingly.

## Run

```bash
python run_triplets.py
```

- Processes every `<pmcid>.txt` in `IN_DIR`, writing `outputs/triplets/<SOURCE>/<pmcid>.json`.
- **Crash-safe & resumable**: results flush after every sentence (atomic write); a
  `.{pmcid}.done` marker skips finished papers, and a partial paper resumes at its
  first unprocessed sentence. Re-run the same command to continue after any
  interruption. Transient network errors are retried indefinitely; a bad key/model
  fails fast.

## Output format

Each output file is a JSON **list** of per-sentence objects:

```json
{
  "text": "<the sentence>",
  "spans": [
    { "text": "...", "type": "<ENTITY TYPE>", "start_char": 0, "end_char": 9, "match_status": "exact" }
  ],
  "triplets": [
    {
      "subject": "...", "subject_type": "<ENTITY TYPE>",
      "subject_start_char": 0, "subject_end_char": 9,
      "relation": "<phrase from the sentence>",
      "predicate": "<one of the 9 schema labels>",
      "object": "...", "object_type": "<ENTITY TYPE>",
      "object_start_char": 15, "object_end_char": 22
    }
  ]
}
```

- **`relation`** is the phrase naming the link (free text, from the sentence);
  **`predicate`** is its type from the controlled schema:
  `IS_AFFECTING, RELATED_TO, LOCATED_IN, COMPARES_TO, HAS_PROPERTY, HAS_PROCESS, CAUSES, DURING, IS_PART_OF`.
- Entity **types** are folded onto the controlled 16-label vocabulary (see `ontology.py`).
- Char offsets index into the sentence's `text`; `match_status` records how the span
  was located (`exact` / `case_insensitive` / `fuzzy` / `unresolved`, the last with `-1` offsets).
