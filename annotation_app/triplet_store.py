"""Persistence and resume logic for the triplet annotator.

Responsibilities:
  - Load a self-contained extraction file: a JSON list with one item per text
    unit (sentence), each carrying ``text``, annotated entity ``spans``, and
    ``triplets`` (subject/relation/predicate/object with character offsets).
  - Maintain a per-(task, annotator) annotation store with atomic writes,
    timestamped snapshots (bounded retention) and a `_latest.json` pointer.
  - Compute the resume point (first sentence not marked done).

The tool is standalone: it only *loads* these extraction files (produced
upstream by an LLM) and never calls a model. The on-disk annotation schema is
documented in README.md.
"""

import datetime
import json
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ontology import normalize_type, normalize_predicate
from span_utils import resolve_span, STATUS_GIVEN, STATUS_UNRESOLVED

MAX_SNAPSHOTS = 50
# Minimum seconds between remote (GitHub) pushes for per-click autosaves;
# milestone saves (snapshot=True, e.g. marking a sentence Done) always push.
REMOTE_PUSH_INTERVAL = 15.0

# Decisions a triplet can carry.
PENDING = "pending"
VALIDATED = "validated"
IGNORED = "ignored"
EDITED = "edited"
ADDED = "added"

# Sentence-level review status.
UNSTARTED = "unstarted"
IN_PROGRESS = "in_progress"
DONE = "done"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_compact() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def new_uid() -> str:
    return uuid.uuid4().hex[:10]


# --------------------------------------------------------------------------- #
# Loading the self-contained extraction file
# --------------------------------------------------------------------------- #
def load_extractions(path: str) -> List[dict]:
    """Load an extraction file: a JSON list of per-sentence items.

    Each item is ``{"text": str, "spans": [...], "triplets": [...]}``, where a
    triplet is a flat dict with keys ``subject``, ``subject_type``,
    ``subject_start_char``, ``subject_end_char``, ``relation``, ``predicate``,
    ``object``, ``object_type``, ``object_start_char``, ``object_end_char``.
    Raises ``ValueError`` if the top-level shape is not a list.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            "Expected a JSON list of {text, spans, triplets} items, "
            f"got {type(data).__name__}."
        )
    return data


def extraction_views(items: List[dict]) -> Tuple[List[str], List[list], List[list], List[str]]:
    """Split loaded items into parallel lists: sentences, spans, raw triplets,
    and per-sentence source-document tags (``doc``, empty string if absent)."""
    sentences = [str(it.get("text", "")) for it in items]
    spans = [list(it.get("spans", []) or []) for it in items]
    triplets = [list(it.get("triplets", []) or []) for it in items]
    docs = [str(it.get("doc", "") or "") for it in items]
    return sentences, spans, triplets, docs


# --------------------------------------------------------------------------- #
# Building triplet cards for a sentence (lazy, on first visit)
# --------------------------------------------------------------------------- #
def _entity_from_flat(triplet: dict, role: str, sentence: str) -> dict:
    """Build an entity dict for ``role`` ('subject'|'object') from a flat input
    triplet. Offsets are taken as given; if absent/invalid we fall back to
    resolving the text against the sentence so the span picker still works."""
    text = str(triplet.get(role, "") or "")
    etype = normalize_type(str(triplet.get(f"{role}_type", "") or ""))
    start = triplet.get(f"{role}_start_char", None)
    end = triplet.get(f"{role}_end_char", None)
    try:
        start_i, end_i = int(start), int(end)
    except (TypeError, ValueError):
        start_i, end_i = -1, -1
    if start_i >= 0 and end_i > start_i and end_i <= len(sentence):
        return {"text": text, "type": etype, "start_char": start_i,
                "end_char": end_i, "match_status": STATUS_GIVEN}
    # Offsets missing or inconsistent — try to recover them from the text.
    res = resolve_span(text, sentence)
    return {"text": res["text"] or text, "type": etype,
            "start_char": res["start_char"], "end_char": res["end_char"],
            "match_status": res["status"]}


def build_cards(sentence_text: str, triplet_items: List[dict]) -> List[dict]:
    """Create pending triplet cards from an item's ``triplets`` list."""
    cards = []
    for it in triplet_items:
        subj = _entity_from_flat(it, "subject", sentence_text)
        obj = _entity_from_flat(it, "object", sentence_text)
        cards.append({
            "uid": new_uid(),
            "source": "llm",
            "decision": PENDING,
            "subject": subj,
            "relation": str(it.get("relation", "") or ""),
            "predicate": normalize_predicate(str(it.get("predicate", "") or "")),
            "object": obj,
            "notes": "",
            "flagged": False,
            "original": dict(it),  # raw input triplet, kept verbatim for audit
        })
    return cards


# --------------------------------------------------------------------------- #
# Annotation store
# --------------------------------------------------------------------------- #
class TripletStore:
    """In-memory annotation state with atomic, snapshotted persistence."""

    def __init__(self, doc_id: str, annotator: str, save_dir: str, task: str = "",
                 remote=None):
        self.doc_id = doc_id
        self.annotator = annotator
        self.task = task
        self.save_dir = save_dir
        # Optional remote backend (duck-typed: pull_latest(path)/push_latest(path))
        # for durable storage on ephemeral hosts. Failures are non-fatal — local
        # disk is always the source of truth; the last error is surfaced for the UI.
        self.remote = remote
        self.remote_error: Optional[str] = None
        self._last_push = 0.0
        # Working state stays here; timestamped snapshots go in a subfolder so the
        # outputs dir isn't cluttered with history files.
        self.snapshot_dir = os.path.join(save_dir, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.data: Dict[str, Any] = {
            "doc_id": doc_id,
            "annotator": annotator,
            "task": task,
            "created": _now_iso(),
            "updated": _now_iso(),
            "progress": {"last_sentence_idx": 0, "done_count": 0},
            "sentences": {},
        }

    # ---- file paths (scoped by doc_id = task + annotator) ---- #
    def _safe_id(self) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in self.doc_id)[:60] or "anon"

    def latest_path(self) -> str:
        # Mirrors the input basename: e.g. relations_davnah -> relations_davnah_annotated.json
        return os.path.join(self.save_dir, f"{self._safe_id()}_annotated.json")

    def _snapshot_path(self) -> str:
        return os.path.join(self.snapshot_dir,
                            f"{self._safe_id()}_annotated_{_ts_compact()}.json")

    # ---------------------------- load / save ------------------------------- #
    def load_latest(self) -> bool:
        """Load this annotator's latest saved state if present. Returns True if
        loaded. If a remote backend is configured, its copy is pulled first so
        work resumes even after the local (ephemeral) disk has been wiped."""
        path = self.latest_path()
        if self.remote is not None:
            try:
                self.remote.pull_latest(path)
            except Exception as e:                     # network/API issues are non-fatal
                self.remote_error = str(e)
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.data.setdefault("sentences", {})
            self.data.setdefault("progress", {"last_sentence_idx": 0, "done_count": 0})
            return True
        except (json.JSONDecodeError, OSError):
            return False

    def save(self, snapshot: bool = False) -> str:
        """Atomically write the latest state locally, then (if a remote backend is
        configured) push it. When `snapshot=True` also write a timestamped local
        copy (milestones like marking a sentence done) and prune old snapshots.

        The remote push is throttled for per-click autosaves (see
        REMOTE_PUSH_INTERVAL) and always runs on milestone saves. Remote failures
        never block the local save — local disk stays the source of truth."""
        self.data["updated"] = _now_iso()
        self.data["progress"]["done_count"] = sum(
            1 for s in self.data["sentences"].values() if s.get("status") == DONE
        )
        text = json.dumps(self.data, ensure_ascii=False, indent=2)
        latest = self.latest_path()
        _atomic_write(text, latest)
        if snapshot:
            _atomic_write(text, self._snapshot_path())
            self._prune_snapshots()
        self._maybe_push(latest, force=snapshot)
        return latest

    def _maybe_push(self, latest_path: str, force: bool) -> None:
        if self.remote is None:
            return
        now = time.time()
        if not force and (now - self._last_push) < REMOTE_PUSH_INTERVAL:
            return
        try:
            self.remote.push_latest(
                latest_path,
                message=f"annotations: {self.doc_id} ({self.data['progress']['done_count']} done)",
            )
            self._last_push = now
            self.remote_error = None
        except Exception as e:                         # non-fatal: local save already succeeded
            self.remote_error = str(e)

    def _prune_snapshots(self) -> None:
        prefix = f"{self._safe_id()}_annotated_"
        snaps = [
            os.path.join(self.snapshot_dir, f) for f in os.listdir(self.snapshot_dir)
            if f.startswith(prefix) and f.endswith(".json")
        ]
        snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for stale in snaps[MAX_SNAPSHOTS:]:
            try:
                os.remove(stale)
            except OSError:
                pass

    # ----------------------------- sentence ops ----------------------------- #
    def has_sentence(self, idx: int) -> bool:
        return str(idx) in self.data["sentences"]

    def get_sentence(self, idx: int) -> Optional[dict]:
        return self.data["sentences"].get(str(idx))

    def init_sentence(self, idx: int, sentence_text: str, triplet_items: List[dict],
                      spans: Optional[List[dict]] = None, doc: str = "") -> dict:
        """Create the sentence record from the extraction file on first visit.

        ``spans`` (all annotated entities in the sentence) is kept verbatim for
        reference/highlighting; ``triplet_items`` are the raw input triplets;
        ``doc`` is the source-document tag (provenance), if any.
        """
        rec = {
            "text": sentence_text,
            "doc": doc,
            "status": IN_PROGRESS,
            "spans": list(spans or []),
            "triplets": build_cards(sentence_text, triplet_items),
        }
        self.data["sentences"][str(idx)] = rec
        return rec

    def set_status(self, idx: int, status: str) -> None:
        rec = self.data["sentences"].get(str(idx))
        if rec:
            rec["status"] = status

    def resume_index(self, n_sentences: int) -> int:
        """First sentence not marked done; clamps to last sentence."""
        for i in range(n_sentences):
            rec = self.data["sentences"].get(str(i))
            if rec is None or rec.get("status") != DONE:
                return i
        return max(0, n_sentences - 1)

    def next_flagged(self, from_idx: int, n_sentences: int) -> Optional[int]:
        """Index of the next sentence after `from_idx` (wrapping around) that has
        a flagged triplet. Returns None if nothing is flagged."""
        if n_sentences <= 0:
            return None
        for step in range(1, n_sentences + 1):
            i = (from_idx + step) % n_sentences
            rec = self.data["sentences"].get(str(i))
            if rec and any(t.get("flagged") for t in rec.get("triplets", [])):
                return i
        return None

    def next_unresolved(self, from_idx: int, n_sentences: int) -> Optional[int]:
        """Index of the next sentence after `from_idx` (wrapping around) with a
        kept triplet whose subject/object span is unresolved. None if none."""
        if n_sentences <= 0:
            return None
        for step in range(1, n_sentences + 1):
            i = (from_idx + step) % n_sentences
            rec = self.data["sentences"].get(str(i))
            if rec and any(
                t.get("decision") != IGNORED and any(
                    e.get("start_char", -1) < 0 or e.get("match_status") == STATUS_UNRESOLVED
                    for e in (t.get("subject", {}), t.get("object", {}))
                )
                for t in rec.get("triplets", [])
            ):
                return i
        return None

    # ----------------------------- stats ------------------------------------ #
    def stats(self) -> Dict[str, int]:
        counts = {VALIDATED: 0, IGNORED: 0, ADDED: 0, EDITED: 0, PENDING: 0,
                  "done_sentences": 0, "flagged": 0, "unresolved": 0}
        for rec in self.data["sentences"].values():
            if rec.get("status") == DONE:
                counts["done_sentences"] += 1
            for t in rec.get("triplets", []):
                counts[t.get("decision", PENDING)] = counts.get(t.get("decision", PENDING), 0) + 1
                if t.get("flagged"):
                    counts["flagged"] += 1
                # Kept triplet with an unlocatable subject/object span (data-quality flag).
                if t.get("decision") != IGNORED and any(
                    e.get("start_char", -1) < 0 or e.get("match_status") == STATUS_UNRESOLVED
                    for e in (t.get("subject", {}), t.get("object", {}))
                ):
                    counts["unresolved"] += 1
        return counts


def _atomic_write(text: str, dst_path: str) -> None:
    """Write text to dst_path atomically (temp file + os.replace + fsync)."""
    dirpath = os.path.dirname(os.path.abspath(dst_path)) or "."
    fd, tmppath = tempfile.mkstemp(prefix=".tmp_", dir=dirpath, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmppath, dst_path)
        try:
            dir_fd = os.open(dirpath, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass  # directory fsync unsupported on some platforms
    finally:
        if os.path.exists(tmppath):
            try:
                os.remove(tmppath)
            except OSError:
                pass
