"""Triplet Annotator — review, edit, and curate LLM-extracted
subject→relation→object triplets.

Standalone tool: it loads a self-contained extraction file per (task, annotator)
from ``extracted_triplets/<name>/<task>_<name>.json`` (task ∈ {relations,
triplets}) and never calls a model. Each triplet carries a free-text *relation*
(the surface label) and a canonical *predicate* (the relation type, from a closed
9-item schema). Annotators walk the document sentence-by-sentence and
Validate / Ignore / Edit / Delete triplets, add missing ones, fix entity spans
and predicates, and flag uncertain cases. Work autosaves per (task, annotator)
and resumes at the first unfinished sentence.
"""

import hmac
import html
import os

import streamlit as st

from ontology import (
    ENTITY_TYPES, type_color,
    PREDICATES, normalize_predicate,
)
from span_utils import (
    span_text, locate_exact, span_from_words,
    STATUS_UNRESOLVED, STATUS_NORMALIZED, STATUS_EDITED,
)
from triplet_store import (
    TripletStore, load_extractions, extraction_views,
    PENDING, VALIDATED, IGNORED, EDITED, ADDED, DONE,
)

# --------------------------------------------------------------------------- #
# Paths (relative to this file; override via env for deployment)
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
# Inputs: <DATA_DIR>/<name>/<task>_<name>.json ; saves go to <SAVE_DIR>.
DATA_DIR = os.environ.get("ANNOTATION_DATA_DIR", os.path.join(HERE, "extracted_triplets"))
SAVE_DIR = os.environ.get("ANNOTATION_SAVE_DIR", os.path.join(HERE, "annotations"))

# Annotation task → filename prefix.
TASKS = {
    "Extract relations": "relations",
    "Extract joint triplets": "triplets",
}

DECISION_META = {
    PENDING:   ("Pending",   "#b26a00", "#fff3e0"),
    VALIDATED: ("Validated", "#2e7d32", "#e8f5e9"),
    IGNORED:   ("Ignored",   "#78909c", "#eceff1"),
    EDITED:    ("Edited",    "#1565c0", "#e3f2fd"),
    ADDED:     ("Added",     "#6a1b9a", "#f3e5f5"),
}

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
:root { --ink:#1f2933; --muted:#7b8794; --line:#e4e7eb; --bg-card:#ffffff; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
.ctx-prev {
  color: var(--muted); font-size: 0.95rem; line-height: 1.6;
  border-left: 3px solid var(--line); padding: 2px 0 2px 14px; margin-bottom: 10px;
}
.ctx-cur {
  font-size: 1.22rem; line-height: 1.95; color: var(--ink); font-weight: 450;
  background: #f6f8fb;
  border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px;
  box-shadow: 0 4px 14px rgba(16,24,40,.10);
}
/* Make the fixed Streamlit header opaque so pinned content never shows through
   it, and leave room below it for the sticky sentence. */
header[data-testid="stHeader"] { background: var(--bg-card); }
/* Pin the WHOLE current-sentence band while the triplet cards below it scroll.
   position:sticky must live on the Streamlit element container (whose parent
   is the tall scrolling block), not on the tightly-wrapped inner div. `top`
   clears the fixed header so the first line isn't hidden underneath it. */
div[data-testid="stElementContainer"]:has(.ctx-cur),
.element-container:has(.ctx-cur) {
  position: sticky; top: 3.5rem; z-index: 50;
  background: var(--bg-card); padding: 6px 0 10px; border-radius: 12px;
}
.ent { padding: 0 2px; border-radius: 3px; }
.tcard {
  border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px 6px 16px;
  margin-bottom: 12px; background: var(--bg-card);
  box-shadow: 0 1px 2px rgba(16,24,40,.04);
}
.tcard.ignored { opacity: .62; }
.badge {
  display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.03em;
  padding:2px 9px; border-radius:999px; text-transform:uppercase;
}
.chip {
  display:inline-block; padding:3px 10px; border-radius:8px; font-weight:600;
  font-size:.98rem; border:1px solid; margin:0 2px;
}
.rel { color:#52606d; font-style:italic; padding:0 8px; font-size:.98rem; }
.pred {
  display:inline-block; font-size:.68rem; font-weight:700; letter-spacing:.04em;
  padding:2px 8px; border-radius:6px; text-transform:uppercase;
  color:#3d4852; background:#eef1f4; border:1px solid #dfe3e8; margin:0 2px;
}
.ttype { font-size:.7rem; color:var(--muted); font-weight:700; letter-spacing:.03em;
         text-transform:uppercase; }
.warn { color:#c62828; font-size:.8rem; font-weight:600; }
.legend span { font-size:.74rem; color:var(--muted); margin-right:12px; }
.legend i { display:inline-block; width:12px; height:12px; border-radius:3px;
            vertical-align:middle; margin-right:4px; }
.smallcap { font-size:.72rem; color:var(--muted); letter-spacing:.04em;
            text-transform:uppercase; font-weight:700; }
.flabel { font-size:.78rem; color:var(--muted); letter-spacing:.02em;
          font-weight:700; }
/* User manual */
.man-intro { font-size:.72rem; color:var(--muted); line-height:1.45; margin:0 0 4px; }
.man-sec { font-size:.66rem; text-transform:uppercase; letter-spacing:.06em;
           color:var(--muted); font-weight:700; margin:14px 0 6px; }
.man-row { display:flex; gap:9px; align-items:baseline; margin:6px 0;
           font-size:.8rem; line-height:1.45; color:var(--ink); }
.man-ic { flex:0 0 18px; text-align:center; font-size:.82rem; font-weight:700; }
.man-row b { font-weight:700; }
.man-row .k { color:var(--muted); font-style:italic; }
.man-note { margin-top:14px; padding-top:9px; border-top:1px solid var(--line);
            font-size:.74rem; color:var(--muted); line-height:1.45; }
</style>
"""


# --------------------------------------------------------------------------- #
# Cached loader
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_extractions(path: str, mtime: float):
    """Load and split an extraction file. `mtime` busts the cache on edits."""
    items = load_extractions(path)
    return extraction_views(items)


def _github_backend():
    """Build the optional GitHub storage backend from the `[github]` secrets
    table or GITHUB_* env vars. Returns None (local-disk only) if unconfigured."""
    cfg = None
    try:
        cfg = dict(st.secrets["github"])       # .streamlit/secrets.toml -> [github]
    except Exception:
        cfg = None
    if not cfg:
        repo, token = os.environ.get("GITHUB_REPO"), os.environ.get("GITHUB_TOKEN")
        if repo and token:
            cfg = {"repo": repo, "token": token,
                   "branch": os.environ.get("GITHUB_BRANCH", "annotations"),
                   "subdir": os.environ.get("GITHUB_SUBDIR", "")}
    if not cfg:
        return None
    from github_store import GitHubBackend
    return GitHubBackend.from_config(cfg)


def _norm_name(name: str) -> str:
    """Normalize an annotator name (trim + lowercase) so 'Mark'/'mark'/' Mark '
    all map to the same annotator and one output file."""
    return " ".join((name or "").split()).lower()


def _input_path(data_dir: str, task_key: str, annotator: str) -> str:
    """Input file for a (task, annotator).

    - relations: per-annotator ``relations_<annotator>.json`` (only some
      annotators have one — they annotate their own extracted entities).
    - triplets:  a single shared ``triplets.json`` for everyone.
    """
    if task_key == "relations":
        return os.path.join(data_dir, f"relations_{annotator}.json")
    return os.path.join(data_dir, f"{task_key}.json")


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def cap_first(text: str) -> str:
    """Capitalize the first letter of a sentence, leaving the rest untouched.
    Length is preserved, so character offsets for spans stay valid."""
    return text[:1].upper() + text[1:] if text else text


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render_sentence_html(text: str, spans: list[dict]) -> str:
    """Render `text` with entity spans highlighted. `spans` items:
    {start, end, color, role ('subject'|'object'), type}. Overlaps: later wins."""
    n = len(text)
    owner = [-1] * n
    valid = [s for s in spans if s.get("start", -1) >= 0 and s.get("end", -1) > s.get("start", -1)]
    for i, s in enumerate(valid):
        for c in range(max(0, s["start"]), min(n, s["end"])):
            owner[c] = i
    out, j = [], 0
    while j < n:
        o = owner[j]
        k = j
        while k < n and owner[k] == o:
            k += 1
        frag = html.escape(text[j:k])
        if o == -1:
            out.append(frag)
        else:
            s = valid[o]
            border = "solid" if s["role"] == "subject" else "dotted"
            style = (f"background:{_rgba(s['color'],0.16)};"
                     f"border-bottom:2.5px {border} {s['color']};border-radius:3px;")
            title = f"{s['role'].title()} · {s.get('type','')}"
            out.append(f'<span class="ent" style="{style}" title="{html.escape(title)}">{frag}</span>')
        j = k
    return "".join(out)


def chip_html(text: str, etype: str) -> str:
    color = type_color(etype)
    return (f'<span class="chip" style="color:{color};border-color:{_rgba(color,0.5)};'
            f'background:{_rgba(color,0.08)};">{html.escape(text or "—")}</span>')


def collect_spans(triplets: list[dict]) -> list[dict]:
    """Spans for the sentence highlight, skipping ignored triplets."""
    spans = []
    for t in triplets:
        if t.get("decision") == IGNORED:
            continue
        for role in ("subject", "object"):
            e = t.get(role, {})
            spans.append({"start": e.get("start_char", -1), "end": e.get("end_char", -1),
                          "color": type_color(e.get("type", "")), "role": role,
                          "type": e.get("type", "")})
    return spans


# --------------------------------------------------------------------------- #
# Type / predicate selectors
# --------------------------------------------------------------------------- #
def type_selector(label: str, current: str, keybase: str,
                  placeholder: str = "Select a type") -> str:
    """Controlled dropdown of the canonical entity types. A custom type can be
    typed directly in the adjacent cell (no 'Other' step) — if filled, it takes
    precedence over the dropdown selection.

    State is keyed so the Swap-subject/object button can rewrite it: the default
    (``index``/``value``) is only supplied on first render, before session_state
    owns the key, to avoid the "default value but also set via Session State"
    warning once the value is managed programmatically."""
    is_known = current in ENTITY_TYPES
    skey, ckey = keybase + "_sel", keybase + "_custom"
    sel_kwargs = {}
    if skey not in st.session_state:
        sel_kwargs["index"] = ENTITY_TYPES.index(current) if is_known else None
    sel = st.selectbox(label, ENTITY_TYPES, key=skey, placeholder=placeholder, **sel_kwargs)
    custom_kwargs = {}
    if ckey not in st.session_state:
        custom_kwargs["value"] = "" if is_known else current
    custom = st.text_input("custom type", key=ckey, label_visibility="collapsed",
                           placeholder="or type a custom type", **custom_kwargs)
    return custom.strip() if custom.strip() else (sel or "")


def predicate_selector(current: str, keybase: str) -> str:
    """Controlled dropdown of the 9 canonical predicates (the relation *type*).
    A custom predicate can be typed directly in the adjacent cell — if filled it
    takes precedence over the dropdown selection (mirrors the entity type
    selector). Anything outside the schema lands in the custom cell rather than
    being silently dropped."""
    cur = normalize_predicate(current)
    is_known = cur in PREDICATES
    skey, ckey = keybase + "_pred", keybase + "_custom"
    sel_kwargs = {}
    if skey not in st.session_state:
        sel_kwargs["index"] = PREDICATES.index(cur) if is_known else None
    sel = st.selectbox("Predicate", PREDICATES, key=skey, placeholder="Select predicate",
                       format_func=lambda p: p.replace("_", " "), **sel_kwargs)
    custom_kwargs = {}
    if ckey not in st.session_state:
        custom_kwargs["value"] = "" if is_known else current
    custom = st.text_input("custom predicate", key=ckey, label_visibility="collapsed",
                           placeholder="or type a custom predicate", **custom_kwargs)
    return custom.strip() if custom.strip() else (sel or "")


def relation_input(current: str, keybase: str) -> str:
    """The free-text relation label (the surface form of the relation).

    Uses a visible "Text" label so the Subject/Relation/Object columns line up.
    """
    if keybase not in st.session_state:
        st.session_state[keybase] = current or ""
    return st.text_input("Text", key=keybase)


def entity_editor(role: str, entity: dict, sentence: str, keybase: str) -> dict:
    """Render editor widgets for one entity; return the updated entity dict.

    The **Text is authoritative** — it's whatever the annotator types and is
    never rewritten by the app. Only the span (start/end) is adjusted to point
    at where the text refers, live as you edit:
      - Edit Text → the app locates it (whole word/phrase) and the start/end
        boxes adapt. A refinement of the current mention keeps the span;
        otherwise the span clears and the text is *unresolved*.
      - Edit start/end → moves the pointer only; the Text does not change.
      - When the span can't be resolved, the "first/last word" boxes appear so
        the annotator can name the anchor. Once used, they stay open (with
        start/end adapting) so the annotator can keep adjusting them.
    Whichever control is edited last wins.
    """
    kt, ks, ke, kms = keybase + "_txt", keybase + "_start", keybase + "_end", keybase + "_ms"
    kfw, klw = keybase + "_fw", keybase + "_lw"
    n = len(sentence)

    def _clamp(v: int) -> int:
        try:
            return max(-1, min(int(v), n))
        except (TypeError, ValueError):
            return -1

    st.session_state.setdefault(kt, entity.get("text", ""))
    st.session_state.setdefault(ks, _clamp(entity.get("start_char", -1)))
    st.session_state.setdefault(ke, _clamp(entity.get("end_char", -1)))
    st.session_state.setdefault(kms, entity.get("match_status", ""))
    st.session_state.setdefault(kfw, "")
    st.session_state.setdefault(klw, "")

    def _on_text_change():
        txt = st.session_state.get(kt, "")
        loc = locate_exact(txt, sentence)
        if loc:                                   # verbatim phrase -> move span to it
            st.session_state[ks], st.session_state[ke] = loc
            st.session_state[kms] = STATUS_EDITED
            return
        cs, ce = st.session_state.get(ks, -1), st.session_state.get(ke, -1)
        prior = span_text(sentence, cs, ce) if (cs >= 0 and ce > cs) else ""
        t, p = txt.strip().lower(), prior.lower()
        if p and t and (p in t or t in p):        # a refinement of the same mention:
            st.session_state[kms] = STATUS_NORMALIZED   # keep the anchor silently
        else:                                     # can't tie it to the sentence -> ask
            st.session_state[ks], st.session_state[ke] = -1, -1
            st.session_state[kms] = STATUS_UNRESOLVED

    def _apply_words():
        fw_, lw_ = st.session_state.get(kfw, ""), st.session_state.get(klw, "")
        if not fw_.strip():                       # cleared -> don't clobber the span
            return
        s, e = span_from_words(sentence, fw_, lw_)
        st.session_state[ks], st.session_state[ke] = s, e
        if s >= 0:                                # keep typed text; anchor to the named words
            st.session_state[kms] = (STATUS_EDITED
                                     if st.session_state.get(kt, "") == span_text(sentence, s, e)
                                     else STATUS_NORMALIZED)
        else:
            st.session_state[kms] = STATUS_UNRESOLVED

    def _on_span_change():                        # manual start/end -> move the pointer only
        cs, ce = _clamp(st.session_state.get(ks, -1)), _clamp(st.session_state.get(ke, -1))
        if cs >= 0 and ce > cs:                    # Text is authoritative; never overwrite it
            st.session_state[kms] = (STATUS_EDITED
                                     if st.session_state.get(kt, "") == span_text(sentence, cs, ce)
                                     else STATUS_NORMALIZED)
        else:
            st.session_state[kms] = STATUS_UNRESOLVED

    st.markdown(f'<span class="flabel">{role.capitalize()}</span>', unsafe_allow_html=True)
    text_val = st.text_input("Text", key=kt, on_change=_on_text_change)
    etype = type_selector("Type", entity.get("type", ""), keybase + "_type")
    s1, s2 = st.columns(2)
    s1.number_input("start", min_value=-1, max_value=n, key=ks, on_change=_on_span_change)
    s2.number_input("end", min_value=-1, max_value=n, key=ke, on_change=_on_span_change)

    start = st.session_state.get(ks, -1)
    end = st.session_state.get(ke, -1)
    resolved = start >= 0 and end > start
    # Show the first/last-word helper when the span isn't resolved, and KEEP it
    # open once the annotator has engaged it (so filling it doesn't hide it).
    fw_val, lw_val = st.session_state.get(kfw, ""), st.session_state.get(klw, "")
    if not resolved or fw_val.strip() or lw_val.strip():
        if not resolved:                          # only while the span is unresolved
            st.caption("This text isn’t a phrase in the sentence — type the first and last "
                       "word it refers to (or set start/end above):")
        w1, w2 = st.columns(2)
        w1.text_input("First word", key=kfw, on_change=_apply_words)
        w2.text_input("Last word (optional)", key=klw, on_change=_apply_words)

    start, end = st.session_state.get(ks, -1), st.session_state.get(ke, -1)
    final_text = text_val              # the Text is authoritative; never derived from the span
    if start >= 0 and end > start:
        ms = st.session_state.get(kms) or entity.get("match_status", "")
    else:
        ms = STATUS_UNRESOLVED
    return {"text": final_text, "type": etype, "start_char": int(start), "end_char": int(end),
            "match_status": ms}


# --------------------------------------------------------------------------- #
# Card rendering
# --------------------------------------------------------------------------- #
def render_card(store: TripletStore, idx: int, t: dict, sentence: str):
    uid = t["uid"]
    kb = f"s{idx}_{uid}"
    editing = st.session_state.get(kb + "_editing", False)
    decision = t.get("decision", PENDING)
    label, fg, bg = DECISION_META.get(decision, DECISION_META[PENDING])

    cls = "tcard ignored" if decision == IGNORED else "tcard"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)

    top = st.columns([3, 2])
    flag = "🚩" if t.get("flagged") else ""
    top[0].markdown(
        f'<span class="badge" style="color:{fg};background:{bg};">{label}</span>'
        + (f'&nbsp;<span class="ttype">{flag}</span>' if flag else ''),
        unsafe_allow_html=True)

    if not editing:
        subj, obj = t.get("subject", {}), t.get("object", {})
        rel = html.escape(t.get("relation", "") or "relates to")
        pred = t.get("predicate", "")
        line = (chip_html(subj.get("text", ""), subj.get("type", "")) +
                f'&nbsp;<span class="rel">{rel}</span>&nbsp;' +
                chip_html(obj.get("text", ""), obj.get("type", "")))
        st.markdown(line, unsafe_allow_html=True)
        # Type line: SUBJECT TYPE - PREDICATE - OBJECT TYPE (predicate de-underscored).
        pred_disp = html.escape(pred.replace("_", " ")) if pred else "?"
        st.markdown(f'<span class="ttype">{html.escape(subj.get("type","") or "?")}'
                    f' &nbsp;-&nbsp; {pred_disp}'
                    f' &nbsp;-&nbsp; {html.escape(obj.get("type","") or "?")}</span>',
                    unsafe_allow_html=True)
        # Unresolved-span warning.
        for e in (subj, obj):
            if e.get("start_char", -1) < 0 or e.get("match_status") == STATUS_UNRESOLVED:
                st.markdown('<span class="warn">⚠ a span is unresolved — Edit to fix</span>',
                            unsafe_allow_html=True)
                break
        if t.get("notes"):
            st.caption(f"📝 {t['notes']}")

        b = st.columns([1, 1, 1, 1, 1])
        if b[0].button("✓ Validate", key=kb + "_val", use_container_width=True,
                       type="primary" if decision != VALIDATED else "secondary"):
            t["decision"] = VALIDATED
            store.save(); st.rerun()
        if b[1].button("✕ Ignore", key=kb + "_ign", use_container_width=True):
            t["decision"] = IGNORED
            store.save(); st.rerun()
        if b[2].button("✎ Edit", key=kb + "_edit", use_container_width=True):
            # Clear any stale editor widget state so it opens from the stored triplet.
            for k in [x for x in st.session_state
                      if x.startswith((kb + "_se", kb + "_oe", kb + "_rel", kb + "_pd"))]:
                st.session_state.pop(k, None)
            st.session_state[kb + "_editing"] = True
            st.rerun()
        if b[3].button("🗑 Delete", key=kb + "_del", use_container_width=True):
            rec = store.get_sentence(idx)
            rec["triplets"] = [x for x in rec["triplets"] if x["uid"] != uid]
            store.save(); st.rerun()
        if b[4].button("🚩 Flag" if not t.get("flagged") else "🚩 Unflag",
                       key=kb + "_flag", use_container_width=True):
            t["flagged"] = not t.get("flagged", False)
            store.save(); st.rerun()
    else:
        def _swap_editor():
            # Swap the subject/object editor fields (text + span + type together,
            # so each text stays consistent with its span). Runs as a callback so
            # it can write widget session_state before the widgets re-render.
            for suf in ("_txt", "_start", "_end", "_type_sel", "_type_custom"):
                ks, ko = kb + "_se" + suf, kb + "_oe" + suf
                if ks in st.session_state or ko in st.session_state:
                    st.session_state[ks], st.session_state[ko] = (
                        st.session_state.get(ko), st.session_state.get(ks))

        cse, crel, cso = st.columns([2, 1.4, 2])
        with cse:
            new_subj = entity_editor("subject", t.get("subject", {}), sentence, kb + "_se")
        with crel:
            st.markdown('<span class="flabel">Relation</span>', unsafe_allow_html=True)
            new_rel = relation_input(t.get("relation", ""), kb + "_rel")
            new_pred = predicate_selector(t.get("predicate", ""), kb + "_pd")
            st.button("⇄ Swap subject/object", key=kb + "_swapedit",
                      on_click=_swap_editor, use_container_width=True,
                      help="Swap the subject and object fields")
        with cso:
            new_obj = entity_editor("object", t.get("object", {}), sentence, kb + "_oe")
        new_notes = st.text_input("Notes (optional)", value=t.get("notes", ""), key=kb + "_notes")
        bb = st.columns([1, 1, 4])
        if bb[0].button("💾 Save", key=kb + "_save", type="primary", use_container_width=True):
            t["subject"], t["object"] = new_subj, new_obj
            t["relation"], t["predicate"], t["notes"] = new_rel, new_pred, new_notes
            if t.get("decision") not in (ADDED,):
                t["decision"] = EDITED
            st.session_state[kb + "_editing"] = False
            store.save(); st.rerun()
        if bb[1].button("Cancel", key=kb + "_cancel", use_container_width=True):
            st.session_state[kb + "_editing"] = False
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def add_triplet_panel(store: TripletStore, idx: int, sentence: str):
    with st.expander("➕ Add a missing triplet", expanded=False):
        kb = f"add_s{idx}"
        c1, crel, c2 = st.columns([2, 1.4, 2])
        with c1:
            st.markdown('<span class="flabel">Subject</span>', unsafe_allow_html=True)
            s_txt = st.text_input("Text", key=kb + "_stxt")
            s_typ = type_selector("Type", "", kb + "_stype")
        with crel:
            st.markdown('<span class="flabel">Relation</span>', unsafe_allow_html=True)
            rel = relation_input("", kb + "_rel")
            pred = predicate_selector("", kb + "_pd")
        with c2:
            st.markdown('<span class="flabel">Object</span>', unsafe_allow_html=True)
            o_txt = st.text_input("Text", key=kb + "_otxt")
            o_typ = type_selector("Type", "", kb + "_otype")
        if st.button("Add triplet", key=kb + "_add", type="primary"):
            if not s_txt.strip() or not o_txt.strip():
                st.warning("Subject and object text are required.")
            else:
                # Text is authoritative (same rule as editing): keep the typed
                # text, locate its span as a whole word/phrase. If it isn't found,
                # leave the span unresolved — the annotator anchors it via Edit
                # (and Done stays blocked until then).
                def _entity(txt: str, typ: str) -> dict:
                    loc = locate_exact(txt, sentence)
                    if loc:
                        return {"text": txt, "type": typ, "start_char": loc[0],
                                "end_char": loc[1], "match_status": STATUS_EDITED}
                    return {"text": txt, "type": typ, "start_char": -1, "end_char": -1,
                            "match_status": STATUS_UNRESOLVED}
                from triplet_store import new_uid
                rec = store.get_sentence(idx)
                rec["triplets"].append({
                    "uid": new_uid(), "source": "human", "decision": ADDED,
                    "subject": _entity(s_txt.strip(), s_typ),
                    "relation": rel, "predicate": pred,
                    "object": _entity(o_txt.strip(), o_typ),
                    "notes": "", "flagged": False, "original": None,
                })
                store.save()
                st.success("Triplet added.")
                st.rerun()


# --------------------------------------------------------------------------- #
# Access gate: single shared password
# --------------------------------------------------------------------------- #
def _expected_password() -> str:
    """The shared app password, from the ANNOTATION_PASSWORD env var or, failing
    that, `app_password` in .streamlit/secrets.toml. Empty string if unset."""
    pw = os.environ.get("ANNOTATION_PASSWORD")
    if pw:
        return pw
    try:
        return str(st.secrets.get("app_password", ""))
    except Exception:
        return ""


def _password_gate() -> None:
    """Require the shared password before anything else. Renders a prompt and
    stops the run until the correct password is entered (kept for the session)."""
    expected = _expected_password()
    if not expected:
        st.title("Triplet Annotator")
        st.error("No app password is configured. Set the `ANNOTATION_PASSWORD` "
                 "environment variable (or `app_password` in "
                 "`.streamlit/secrets.toml`) before deploying.")
        st.stop()

    if st.session_state.get("auth_ok"):
        return

    def _submit():
        entered = str(st.session_state.get("pw_input", ""))
        st.session_state.auth_ok = hmac.compare_digest(entered, str(expected))
        st.session_state.pop("pw_input", None)  # never keep the raw password around

    st.title("Triplet Annotator")
    st.text_input("Password", type="password", key="pw_input", on_change=_submit,
                  placeholder="Enter the shared password to continue")
    if st.session_state.get("auth_ok") is False:
        st.error("Incorrect password.")
    st.stop()


# --------------------------------------------------------------------------- #
# Startup gate: annotator name + task choice
# --------------------------------------------------------------------------- #
def _startup_gate() -> bool:
    """Collect the annotator name and task. Returns True once both are set and a
    matching input file exists (stored in session_state); otherwise renders the
    gate and returns False."""
    if st.session_state.get("annotator") and st.session_state.get("task"):
        return True

    st.title("Triplet Annotator")
    st.caption("Review and curate LLM-extracted subject → relation → object triplets.")

    # Any annotator: type your name (returning annotators resume by typing the
    # same name again — their saved work is loaded automatically).
    name = st.text_input("Your name", key="gate_name",
                         placeholder="Type your name to begin")
    task_label = st.radio("What do you want to annotate?", list(TASKS),
                          horizontal=True, key="gate_task")

    if st.button("Start", type="primary"):
        annotator = _norm_name(name)
        if not annotator:
            st.warning("Please enter your name.")
        else:
            task_key = TASKS[task_label]
            path = _input_path(DATA_DIR, task_key, annotator)
            if task_key == "relations" and not os.path.exists(path):
                st.error("You can only annotate the **triplets** — there's no relations "
                         "file for this name. Choose **Extract joint triplets** above.")
            elif not os.path.exists(path):
                st.error(f"No input file for **{task_label}**.\n\nExpected: `{path}`")
            else:
                st.session_state.annotator = annotator
                st.session_state.task = task_key
                st.session_state.task_label = task_label
                st.rerun()
    st.stop()
    return False


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Triplet Annotator", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    _password_gate()
    _startup_gate()

    annotator = st.session_state.annotator
    task = st.session_state.task
    task_label = st.session_state.get("task_label", task)
    doc_id = f"{task}_{annotator}"          # keeps each annotator's OUTPUT separate
    path = _input_path(DATA_DIR, task, annotator)   # relations: per-annotator; triplets: shared

    if not os.path.exists(path):
        st.error(f"Input file no longer found: {path}")
        st.stop()
    try:
        sentences, spans_by_idx, triplets_by_idx, docs = _load_extractions(path, os.path.getmtime(path))
    except (ValueError, KeyError) as e:
        st.error(f"Could not read `{path}`: {e}")
        st.stop()
    n = len(sentences)
    if n == 0:
        st.warning("The extraction file is empty.")
        st.stop()

    # ----- Store init / resume ----- #
    if "store" not in st.session_state:
        store = TripletStore(doc_id, annotator, SAVE_DIR, task=task,
                             remote=_github_backend())
        resumed = store.load_latest()
        st.session_state.store = store
        st.session_state.cur_idx = store.resume_index(n) if resumed else 0
        if resumed:
            st.toast(f"Resumed at sentence {st.session_state.cur_idx + 1}/{n}", icon="↩️")
    store: TripletStore = st.session_state.store
    idx = min(st.session_state.cur_idx, n - 1)

    # Lazily initialize the current sentence record from the extraction file.
    if not store.has_sentence(idx):
        store.init_sentence(idx, sentences[idx], triplets_by_idx[idx], spans_by_idx[idx],
                            doc=docs[idx] if idx < len(docs) else "")
        store.save()
    rec = store.get_sentence(idx)

    # ============================== SIDEBAR ============================== #
    with st.sidebar:
        st.markdown("### Triplet Annotator")
        st.caption(f"**{annotator}** · `{task_label}`")
        if st.button("↩ Switch task / annotator", use_container_width=True):
            for k in ("annotator", "task", "task_label", "store", "cur_idx"):
                st.session_state.pop(k, None)
            st.rerun()

        with st.expander("📖 User manual", expanded=False):
            def _man_row(icon, color, title, desc):
                return (f'<div class="man-row"><span class="man-ic" '
                        f'style="color:{color}">{icon}</span>'
                        f'<span><b>{title}</b> — {desc}</span></div>')
            st.markdown(
                '<div class="man-intro">For each sentence, resolve every detected '
                'triplet, then close it with <b>Done</b> or <b>Skip</b>.</div>'
                '<div class="man-sec">Resolve each triplet</div>'
                + _man_row("✓", "#2e7d32", "Validate", "accept the triplet as correct.")
                + _man_row("✕", "#78909c", "Ignore",
                           "reject a wrong triplet but keep it on record (greyed out, "
                           "reversible).")
                + _man_row("✎", "#1565c0", "Edit", "fix its subject, relation, predicate or object.")
                + _man_row("🗑", "#c62828", "Delete",
                           "erase a triplet completely, leaving no trace. Use for ones you added by mistake.")
                + _man_row("🚩", "#e65100", "Flag",
                           'mark as uncertain — revisit via <span class="k">Next flagged triplet</span>.')
                + _man_row("➕", "#6a1b9a", "Add", "add a missing triplet (auto-marked <b>Added</b>).")
                + '<div class="man-sec">Close the sentence</div>'
                + _man_row("✔", "#2e7d32", "Done",
                           "mark the sentence fully reviewed — blocked while any triplet is pending.")
                + _man_row("⏭", "#52606d", "Skip",
                           'save progress, leave it open — revisit via <span class="k">Next unreviewed sentence</span>.')
                + '<div class="man-note">Any edit, at any time (even after '
                  '<b>Done</b>), overwrites the triplet\'s previous values.</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        stt = store.stats()
        done = stt["done_sentences"]
        st.progress(done / n if n else 0, text=f"{done}/{n} sentences done")
        m = st.columns(2)
        m[0].metric("Validated", stt[VALIDATED]); m[1].metric("Added", stt[ADDED])
        m2 = st.columns(2)
        m2[0].metric("Ignored", stt[IGNORED]); m2[1].metric("Flagged", stt["flagged"])
        m3 = st.columns(2)
        m3[0].metric("Edited", stt[EDITED])

        st.divider()
        st.markdown('<span class="smallcap">Navigate</span>', unsafe_allow_html=True)
        jump = st.number_input("Go to sentence #", min_value=1, max_value=n, value=idx + 1)
        if st.button("Go", use_container_width=True):
            st.session_state.cur_idx = int(jump) - 1
            st.rerun()
        if st.button("⏭ Next unreviewed sentence", use_container_width=True):
            nxt = store.resume_index(n)
            st.session_state.cur_idx = nxt
            st.rerun()
        if st.button("🚩 Next flagged triplet", use_container_width=True):
            nxt = store.next_flagged(idx, n)
            if nxt is None:
                st.toast("No flagged triplets", icon="🚩")
            else:
                st.session_state.cur_idx = nxt
                st.rerun()

        st.divider()
        st.caption(f"Autosaves to `{os.path.basename(store.latest_path())}`")
        if store.remote is not None:
            if store.remote_error:
                st.caption(f"⚠️ GitHub sync error: {store.remote_error}")
            else:
                st.caption("☁️ Synced to GitHub")

        st.divider()
        st.markdown('<span class="smallcap">Entity-type colors</span>', unsafe_allow_html=True)
        fams = [("Biotic", "BIOTIC ENTITY"), ("Abiotic", "ABIOTIC ENTITY"),
                ("Anthropogenic", "ANTHROPOGENIC ENTITY"), ("Spatial", "SPATIAL ENTITY"),
                ("Temporal", "TEMPORAL ENTITY"), ("Quant/Qual", "QUANTITATIVE PROPERTY"),
                ("Concept", "CONCEPT")]
        legend = "".join(f'<span><i style="background:{type_color(t)}"></i>{name}</span>'
                         for name, t in fams)
        st.markdown(f'<div class="legend">{legend}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="legend">'
            '<span style="border-bottom:2.5px solid var(--ink);padding-bottom:1px;">Subject</span>'
            '&nbsp;&nbsp;&nbsp;'
            '<span style="border-bottom:2.5px dotted var(--ink);padding-bottom:1px;">Object</span>'
            '</div>', unsafe_allow_html=True)

    # ============================== MAIN ============================== #
    doc = docs[idx] if idx < len(docs) else ""
    new_paper = bool(doc) and (idx == 0 or docs[idx - 1] != doc)
    head = f"#### Sentence {idx + 1} / {n}"
    if doc:
        head += f"  ·  📄 {doc}"
    st.markdown(head)
    if new_paper:
        st.caption("🆕 Start of a new source paper")

    # Context band: previous sentence (muted) + current (highlighted).
    if idx > 0:
        st.markdown(f'<div class="ctx-prev">{html.escape(cap_first(sentences[idx-1]))}</div>',
                    unsafe_allow_html=True)
    spans = collect_spans(rec["triplets"])
    st.markdown(f'<div class="ctx-cur">{render_sentence_html(cap_first(sentences[idx]), spans)}</div>',
                unsafe_allow_html=True)
    st.write("")

    # Triplet cards.
    triplets = rec["triplets"]
    pending = sum(1 for t in triplets if t.get("decision") == PENDING)
    st.markdown(f'<span class="smallcap">Triplets ({len(triplets)}) · '
                f'{pending} pending review</span>', unsafe_allow_html=True)
    if not triplets:
        st.info("No triplets for this sentence. Add any you find, or mark the sentence done.")
    for t in list(triplets):
        render_card(store, idx, t, sentences[idx])

    add_triplet_panel(store, idx, sentences[idx])

    # ----- Footer: done + navigation ----- #
    st.divider()
    has_triplets = len(triplets) > 0
    prev_c, done_c, skip_c = st.columns([1, 1, 1])

    if prev_c.button("◀ Previous", use_container_width=True, disabled=idx == 0):
        store.save()
        st.session_state.cur_idx = max(0, idx - 1)
        st.rerun()

    unresolved = any(
        (t.get("decision") != IGNORED) and
        (t["subject"].get("start_char", -1) < 0 or t["object"].get("start_char", -1) < 0)
        for t in triplets)

    done_label = "Confirm no triplets ▶" if not has_triplets else "Done ▶"
    if done_c.button(done_label, type="primary", use_container_width=True):
        if pending > 0:                       # every triplet must be reviewed
            st.warning(f"{pending} triplet(s) still pending — validate, ignore, or edit them first.")
        elif unresolved:                      # hard block: the span must be resolved first
            st.warning("Some kept triplets have an unresolved span — fix them "
                       "(edit the text, set start/end, or name the first/last word), "
                       "or Ignore/Delete them, before marking this sentence done.")
        else:
            store.set_status(idx, DONE)
            store.save(snapshot=True)
            st.session_state.cur_idx = min(n - 1, idx + 1)
            st.rerun()

    if skip_c.button("Skip ▶", use_container_width=True, disabled=idx >= n - 1):
        store.save()
        st.session_state.cur_idx = min(n - 1, idx + 1)
        st.rerun()


if __name__ == "__main__":
    main()
