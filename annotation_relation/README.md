# MoBiKo — Relation Extraction

Assign **relations between already annotated entities** in mountain biodiversity
scientific texts, producing `(subject, relation, object)` triplets for a
knowledge graph of biodiversity states, trends, and drivers of change.

This is the **relation-only** stage: two domain experts have *already* identified and typed the entities in each sentence. An LLM reads each sentence with its given entities and decides **which pairs are related** and **with which predicate**, from a fixed
9 labels schema. It does *not* add, remove, re-span, or re-type entities.

## Relation schema

Each triplet carries a free-text `relation` phrase (grounded in the sentence's
wording, e.g. "drives", "is located in") **and** a `predicate`, exactly one of:

| Predicate | Meaning |
|---|---|
| `IS_AFFECTING` | cause-effect, transformation, indication, enabling, constraint |
| `RELATED_TO` | non-causal association, dependency, interaction |
| `LOCATED_IN` | spatial containment |
| `COMPARES_TO` | comparison (less / more / equal) |
| `HAS_PROPERTY` | subject has an attribute / value / measurement |
| `HAS_PROCESS` | subject undergoes / hosts a process |
| `CAUSES` | subject brings the object into existence (stronger than `IS_AFFECTING`) |
| `DURING` | temporal containment (`LOCATED_IN` in time) |
| `IS_PART_OF` | subject is a component of a larger, same-kind object |

## Files

| File | Purpose |
|---|---|
| `prompt_relation.py` | `PROMPT_RELATION`: the extraction prompt. |
| `utilities.py` | Engine: robust streamed model calls (retried through drops), tolerant JSON parsing, and `extract_relations`. |
| `run_relations.py` | Headless, crash-safe, resumable runner over both annotators. |
| `relation_extraction.ipynb` | Interactive equivalent of the runner. |
| `outputs/relations/` | Example results (`relations_davnah.json`, `relations_mark.json`). |

## Setup

```bash
pip install -r requirements.txt
```

Provide a SwissAI API key in a file at `../tokens/swissAI_key.rtf` (`.rtf` or plain text both work). The path, model id, and gateway
URL are set at the top of `run_relations.py`:

```python
os.environ['CSCS_SERVING_API'] = get_file_contents('../tokens/swissAI_key.rtf')
model = {'name': 'Qwen/Qwen3.5-27B',
         'client': OpenAI(api_key=..., base_url='https://api.swissai.svc.cscs.ch/v1')}
```

> **Note:** the served model id changes over time on SwissAI. If a call returns
> `503 - No provider found`, list the currently-served models
> (`GET /v1/models`) and update `model['name']` accordingly.

## Input data

`run_relations.py` reads (paths configurable at the top of the file):

- **`SOURCE`**: a JSON document `{"sentences": [{"text": ...}, ...]}`.
- **`OVERLAP`**: a text file of sentences to process (one per line).
- **`ANNOTATORS`**: per annotator, a JSON `{"sentences": [{"spans": [{text, type, start_char, end_char, uid}, ...]}, ...]}` of the expert-annotated entities.

> These MoBiKo corpus files are **not** included in this repository. Point the
> path constants at your own data, keeping the shapes above.

## Run

Headless (resumable, safe to stop and restart):

```bash
python run_relations.py
```

or step through `relation_extraction.ipynb`.

Each sentence's entities are listed in the prompt; the model returns triplets,
each subject/object is **snapped back to a given entity** (by char span, else
text) so miscopied spans/types are corrected and any entity not in the list is
dropped. Results flush after every sentence (atomic write); a per-annotator
`.done` marker lets a restart skip finished annotators.

## Output format

One file per annotator, a list of per-sentence objects:

```json
{
  "text": "<sentence>",
  "spans": [ {"text": "...", "type": "...", "start_char": 0, "end_char": 7, "uid": "..."} ],
  "triplets": [
    { "subject": "...", "subject_type": "...", "subject_start_char": 0, "subject_end_char": 7,
      "relation": "<phrase from the sentence>", "predicate": "<one of the 9 labels>",
      "object": "...", "object_type": "...", "object_start_char": 20, "object_end_char": 32 }
  ]
}
```

Subject/object text, types, and char spans are copied verbatim from the expert
entity, the model only supplies the `relation` phrase and the `predicate`.
