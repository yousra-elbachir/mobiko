# Triplet Annotator

A **standalone** Streamlit tool for human validation of
LLM-extracted **subject → relation → object** triplets.

Annotators pick their name and a task, then walk a document sentence-by-sentence
and, for each triplet, **Validate / Ignore / Edit / Delete** it, **add** missing
triplets, fix entity spans (punctuation excluded), adjust the relation label and
its **predicate**, and flag uncertain cases. Work autosaves per (task, annotator)
and **resumes** at the first unfinished sentence.

> **Extraction is fully separate from this tool.** The triplets are extracted
> upstream by an LLM and written to JSON. This folder is the annotation app
> only — it has **no model dependency** and never calls a model; it only
> *loads* the extraction JSON.

Each triplet carries two relation fields:

- **`relation`** — the free-text surface label as it appears in the sentence
  (e.g. `"correlate with"`, `"of"`). Annotators edit it as free text.
- **`predicate`** — the *type* of relation, drawn from a closed 9-item schema.
  Annotators pick it from a dropdown (mirroring how entity `type` is chosen for
  the subject and object).

Predicate schema: `HAS_PROPERTY`, `HAS_PROCESS`, `IS_AFFECTING`, `CAUSES`,
`RELATED_TO`, `COMPARES_TO`, `LOCATED_IN`, `DURING`, `IS_PART_OF`.

---

## Quick start

```bash
cd code/annotation_app
pip install -r requirements.txt
ANNOTATION_PASSWORD="your-shared-password" streamlit run streamlit_app_triplets.py
```

The app is protected by a **single shared password** (see *Access* below). After
entering it, choose your **name** and a **task**
(*Extract relations* or *Extract joint triplets*); the app loads the matching
file and you begin annotating.

### Inputs

The app reads one self-contained extraction file per **(annotator, task)**:

```
extracted_triplets/<name>/<task>_<name>.json      task ∈ { relations, triplets }
```

e.g. `extracted_triplets/davnah/relations_davnah.json`. The start screen only
offers names that have at least one such file. Both task variants share the
**same schema**; only the filename prefix differs.

### Paths (override via env vars)

| Env var               | Default                       | Purpose                                   |
|-----------------------|-------------------------------|-------------------------------------------|
| `ANNOTATION_PASSWORD` | *(required)*                  | Shared password gating the whole app.     |
| `ANNOTATION_DATA_DIR` | `./extracted_triplets`        | Root holding `<name>/<task>_<name>.json`. |
| `ANNOTATION_SAVE_DIR` | `./annotations`               | Per-(task, annotator) saves + snapshots.  |
| `GITHUB_REPO` / `GITHUB_TOKEN` / `GITHUB_BRANCH` | *(unset)* | Optional durable storage (see *Persistence*). |

### Access (shared password)

The whole app sits behind one shared password. Set it via the
`ANNOTATION_PASSWORD` env var (preferred for deployment) **or** `app_password` in
`.streamlit/secrets.toml` (copy `.streamlit/secrets.toml.example`). 

### Persistence (durable storage)

The app writes annotations to `ANNOTATION_SAVE_DIR`. On a VM or Kubernetes PVC
that directory is durable, so nothing else is needed. On **ephemeral hosts whose
disk is wiped on restart** (e.g. **Streamlit Community Cloud**), configure the
optional **GitHub backend** so work isn't lost:

- Set a `[github]` table in secrets (or `GITHUB_REPO`/`GITHUB_TOKEN`/`GITHUB_BRANCH`
  env vars) — see `.streamlit/secrets.toml.example`.
- `token` = a fine-grained PAT with **Contents: read & write** on the repo.
- Each save commits `<task>_<name>_annotated.json` to a **dedicated branch**
  (default `annotations`) that the deployment does **not** track, so saving never
  redeploys the app; on startup the app pulls that file back so work resumes.
- Pushes are throttled for per-click autosaves and forced on milestones (marking
  a sentence Done). Remote failures are non-fatal — local disk stays authoritative
  and the sidebar shows the sync status.
- Git history becomes a free version trail of the annotations.

Deploying free on **Streamlit Community Cloud**: push this folder to a (public)
GitHub repo, create the app from it, and set `app_password` + the `[github]`
secrets in the app's *Settings → Secrets*.

---

## Files

| File | Purpose |
|------|---------|
| `streamlit_app_triplets.py` | The annotation UI (startup gate, cards, navigation). |
| `triplet_store.py` | Load extraction files, build triplet cards, per-(task, annotator) save/resume, atomic writes + snapshots. |
| `span_utils.py` | Resolve entity text → char span, excluding boundary punctuation. |
| `ontology.py` | Controlled entity-type vocabulary + family colors, and the 9-predicate relation schema. |
| `github_store.py` | Optional GitHub-branch storage backend for durable saves on ephemeral hosts. |
| `merge_papers.ipynb` | Append-only merge of per-paper `PMC*.json` into one `<task>_<name>.json` input. |

---

## Input format (extraction file)

A JSON **list** with one item per sentence. Each item is self-contained — the
text, the annotated entity `spans`, and the `triplets` all travel together (no
separate source document is needed):

```jsonc
[
  {
    "text": "We find that centres of species richness correlate with areas of high temperatures.",
    "spans": [
      {"text": "centres of", "type": "SPATIAL ENTITY", "start_char": 13, "end_char": 23, "uid": "e0f4fa6ea4"},
      {"text": "species richness", "type": "BIOTIC PROPERTY", "start_char": 24, "end_char": 40, "uid": "cd43e1707c"}
    ],
    "triplets": [
      {
        "subject": "centres of", "subject_type": "SPATIAL ENTITY",
        "subject_start_char": 13, "subject_end_char": 23,
        "relation": "of", "predicate": "HAS_PROPERTY",
        "object": "species richness", "object_type": "BIOTIC PROPERTY",
        "object_start_char": 24, "object_end_char": 40
      }
    ]
  }
]
```

Triplet character offsets are taken as given. If an offset is missing or
inconsistent with the text, the app falls back to resolving the entity text
against the sentence (punctuation-trimmed) so spans stay usable. `spans` is used
to highlight entities in the sentence.

## Output / save format (per task + annotator)

Saved to `<task>_<name>_annotated.json` (mirrors the input basename; +
timestamped snapshots in `snapshots/`, max 50, atomic writes) under
`ANNOTATION_SAVE_DIR`. Resume opens the first sentence whose `status != "done"`.

```jsonc
{
  "doc_id": "relations_davnah", "annotator": "davnah", "task": "relations",
  "updated": "2026-07-07T..Z",
  "progress": { "last_sentence_idx": 12, "done_count": 12 },
  "sentences": {
    "0": {
      "text": "We find that centres of species richness correlate with ...",
      "status": "done",                       // unstarted | in_progress | done
      "spans": [ /* the input entity spans, kept for reference */ ],
      "triplets": [{
        "uid": "a1b2c3d4e5",
        "source": "llm",                      // llm | human
        "decision": "validated",              // pending|validated|ignored|edited|added
        "subject": {"text":"centres of","type":"SPATIAL ENTITY","start_char":13,"end_char":23,"match_status":"given"},
        "relation": "of",
        "predicate": "HAS_PROPERTY",
        "object":  {"text":"species richness","type":"BIOTIC PROPERTY","start_char":24,"end_char":40,"match_status":"given"},
        "notes": "", "flagged": false,
        "original": { /* raw input triplet, kept for audit when edited */ }
      }]
    }
  }
}
```

Entity `text` is always kept identical to `sentence[start_char:end_char]`, and
edited/added spans are trimmed so they never start/end on punctuation.

---

## Features

- **Startup gate:** pick your name (from detected annotator folders) and the
  task (*Extract relations* / *Extract joint triplets*); switch either at any
  time from the sidebar.
- **Context band:** previous sentence (muted) above the current sentence, which
  is highlighted NER-style — color by entity-type family, **solid** underline for
  subjects, **dotted** for objects.
- **Per-triplet actions:** Validate · Ignore · Edit · Flag · Delete.
- **Edit:** controlled entity-type dropdowns (+ free-text custom type), free-text
  relation label, a **predicate** dropdown (the 9-item schema, with definitions),
  and span adjustment with **Auto-detect span** and a live span preview.
- **Add missing triplets** with automatic span resolution.
- **Resume** at the last unfinished sentence; **autosave** on every action,
  timestamped snapshots on milestones / manual save.
- **Guardrails:** can't mark a sentence done while triplets are pending or have
  unresolved spans; one-click *"no valid triplets here"*.
- **Navigation:** jump to sentence #, *next unreviewed*, prev/skip.
- **Stats:** progress bar, validated / added / ignored / flagged counts.
- **Multi-annotator-ready:** all state is scoped by (task, annotator) on disk,
  so inter-annotator agreement can be computed across the per-annotator outputs.

## Deployment

See `DEPLOY.md` for RunAI / Kubernetes packaging (single replica + PVC for
persistence; extraction files and saves live on the PVC so updating inputs needs
no rebuild).
