"""Controlled entity-type vocabulary + type normalization for MoBiKo triplets."""

# Canonical controlled vocabulary (16 labels), grouped by ontological family.
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


def normalize_type(entity_type: str) -> str:
    """Fold the format variants the model emits onto the canonical vocabulary.

    The prompt defines types as UPPERCASE, space-separated strings. The model also
    emits Title Case ("Biotic Entity"), underscored ("BIOTIC_PROPERTY"), and a few
    typos ("ANTROPOGENIC"/"ANROPOGENIC"). We canonicalize casing, separators, and
    those typos so they land on the controlled set; anything still outside it is
    passed through unchanged.
    """
    if not entity_type:
        return ""
    t = entity_type.strip().upper()
    t = t.replace("_", " ")                 # BIOTIC_PROPERTY -> BIOTIC PROPERTY
    t = " ".join(t.split())                 # collapse repeated whitespace
    t = t.replace("ANTROPOGENIC", "ANTHROPOGENIC").replace("ANROPOGENIC", "ANTHROPOGENIC")
    return t if t in ENTITY_TYPES else entity_type.strip()
