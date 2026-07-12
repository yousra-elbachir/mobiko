"""Controlled vocabularies and visual styling for the triplet annotator.

Two closed vocabularies drive the editor dropdowns:
  - ``ENTITY_TYPES`` — the entity types a subject/object can take, grouped by
    ontological family (each family shares one display color).
  - ``PREDICATES`` — the relation *types* a triplet can take.

Both are "controlled but not closed": the UI offers the canonical value in a
dropdown and a free-text cell for anything outside the set, and the normalizers
below fold common casing/separator/spelling variants onto the canonical value
rather than dropping them.
"""

# Canonical controlled entity vocabulary, grouped by ontological family.
ENTITY_TYPES = [
    "BIOTIC ENTITY",
    "BIOTIC PROPERTY",
    "BIOTIC PROCESS",
    "ABIOTIC ENTITY",
    "ABIOTIC PROPERTY",
    "ABIOTIC PROCESS",
    "ANTHROPOGENIC ENTITY",
    "ANTHROPOGENIC PROPERTY",
    "ANTHROPOGENIC PROCESS",
    "SPATIAL ENTITY",
    "SPATIAL PROPERTY",
    "TEMPORAL ENTITY",
    "TEMPORAL PROPERTY",
    "QUANTITATIVE PROPERTY",
    "QUALITATIVE PROPERTY",
    "CONCEPT",
]

# --------------------------------------------------------------------------- #
# Controlled relation-predicate vocabulary
# --------------------------------------------------------------------------- #
# Each triplet carries a free-text ``relation`` (the surface label, e.g.
# "correlate with") *and* a canonical ``predicate`` drawn from this closed
# schema, which classifies the *type* of relation. Meaning of each:
PREDICATES = [
    "HAS_PROPERTY",   # subject has an attribute/value/measurement (the object)
    "HAS_PROCESS",    # subject undergoes or hosts the object process
    "IS_AFFECTING",   # subject acts on / alters / drives / constrains the object
    "CAUSES",         # subject brings the object into existence (stronger than IS_AFFECTING)
    "RELATED_TO",     # non-causal association, dependency, or interaction
    "COMPARES_TO",    # a comparison (less than, more than, or equal to)
    "LOCATED_IN",     # spatial containment (subject situated within the object place)
    "DURING",         # temporal containment (subject occurs within the object period)
    "IS_PART_OF",     # subject is a component/part of a larger, same-kind object
]


def normalize_predicate(predicate: str) -> str:
    """Fold predicate spelling/casing variants onto the canonical schema.

    Anything still outside the closed set is passed through unchanged so the UI
    can preserve (and surface) it rather than silently dropping it.
    """
    if not predicate:
        return ""
    p = " ".join(predicate.strip().upper().replace("-", "_").replace(" ", "_").split())
    return p if p in PREDICATES else predicate.strip()


# One base color per ontological family; all members of a family share it.
_FAMILY_COLORS = {
    "BIOTIC": "#2e7d32",         # green
    "ABIOTIC": "#1565c0",        # blue
    "ANTHROPOGENIC": "#e65100",  # orange
    "SPATIAL": "#6a1b9a",        # purple
    "TEMPORAL": "#00838f",       # teal
    "QUANTITATIVE": "#ad1457",   # rose
    "QUALITATIVE": "#ad1457",    # rose
    "CONCEPT": "#455a64",        # slate
}

_DEFAULT_COLOR = "#607d8b"


def type_color(entity_type: str) -> str:
    """Return the display color for an entity type (matched by family prefix)."""
    if not entity_type:
        return _DEFAULT_COLOR
    head = entity_type.strip().upper().split()[0]
    return _FAMILY_COLORS.get(head, _DEFAULT_COLOR)


def normalize_type(entity_type: str) -> str:
    """Fold the format variants the extraction model emits onto the canonical
    vocabulary.

    The controlled types are UPPERCASE, space-separated strings. In practice a
    model also emits Title Case ("Biotic Entity"), underscored
    ("BIOTIC_PROPERTY"), and a couple of typos ("ANTROPOGENIC"/"ANROPOGENIC").
    We canonicalize casing, separators, and those typos so they land on the
    controlled set. Anything still outside the set is passed through unchanged.
    """
    if not entity_type:
        return ""
    t = entity_type.strip().upper()
    t = t.replace("_", " ")                 # BIOTIC_PROPERTY -> BIOTIC PROPERTY
    t = " ".join(t.split())                 # collapse repeated whitespace
    # Spelling variants of ANTHROPOGENIC seen in the data.
    t = t.replace("ANTROPOGENIC", "ANTHROPOGENIC").replace("ANROPOGENIC", "ANTHROPOGENIC")
    return t if t in ENTITY_TYPES else entity_type.strip()
