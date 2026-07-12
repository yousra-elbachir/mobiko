"""Resolve a triplet entity's text to a character span in its sentence.

Strategy (per the agreed design — pure text-matching, punctuation excluded):
  1. Strip surrounding punctuation/quotes/whitespace from the entity text.
  2. Find it in the sentence: exact match first, then case-insensitive.
  3. If still not found, fall back to a longest-contiguous-token-overlap search.
  4. Trim any punctuation that ends up at the boundaries of the matched region.

Every result carries a `status` so the UI can surface unresolved matches for the
annotator to adjust manually.
"""

import re
import string
from typing import Optional, Tuple

# Punctuation to strip from span boundaries: ASCII punctuation + common unicode
# quotes/dashes/ellipsis. We deliberately keep internal punctuation (e.g.
# "Earth's surface") and only trim the outer edges.
_EDGE_CHARS = set(string.punctuation) | set("“”‘’«»—–…·\t\n\r ")

STATUS_EXACT = "exact"
STATUS_CASE_INSENSITIVE = "case_insensitive"
STATUS_FUZZY = "fuzzy"
STATUS_UNRESOLVED = "unresolved"
# Offsets supplied directly by the input file (already resolved upstream).
STATUS_GIVEN = "given"


def trim_edges(text: str, start: int, end: int) -> Tuple[int, int]:
    """Shrink [start, end) inward so the boundaries exclude edge punctuation."""
    while start < end and text[start] in _EDGE_CHARS:
        start += 1
    while end > start and text[end - 1] in _EDGE_CHARS:
        end -= 1
    return start, end


def _clean_query(entity_text: str) -> str:
    return entity_text.strip().strip("".join(_EDGE_CHARS)).strip()


# Stopwords we won't accept as a *sole* fuzzy match (avoids latching onto "the").
_STOPWORDS = {"the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "are",
              "was", "were", "by", "for", "with", "as", "at", "its", "their", "this", "that"}


def _token_overlap(sentence: str, query: str) -> Optional[Tuple[int, int]]:
    """Longest run of consecutive query tokens found verbatim in the sentence.

    Handles cases where the model lightly rephrased the entity (extra article, plural)
    but a contiguous core still appears in the text. Prefers the longest match at
    the widest token window, and never returns a lone stopword.
    """
    tokens = [t for t in re.split(r"\s+", query) if t]
    if not tokens:
        return None
    low_sentence = sentence.lower()
    # Try progressively shorter contiguous token windows; within a window size,
    # keep the longest (by character length) verbatim match found.
    for win in range(len(tokens), 0, -1):
        best = None
        for i in range(0, len(tokens) - win + 1):
            window = tokens[i:i + win]
            phrase = " ".join(window).lower()
            if len(phrase) < 3:
                continue
            if win == 1 and phrase in _STOPWORDS:
                continue
            pos = low_sentence.find(phrase)
            if pos != -1 and (best is None or len(phrase) > best[1] - best[0]):
                best = (pos, pos + len(phrase))
        if best:
            return best
    return None


def resolve_span(entity_text: str, sentence: str) -> dict:
    """Locate `entity_text` in `sentence`, returning a punctuation-trimmed span.

    Returns a dict: {text, start_char, end_char, status}. When unresolved,
    start_char/end_char are -1 and `text` is the cleaned query for display.
    """
    query = _clean_query(entity_text or "")
    if not query or not sentence:
        return {"text": query, "start_char": -1, "end_char": -1,
                "status": STATUS_UNRESOLVED}

    # 1. Exact (case-sensitive) match.
    pos = sentence.find(query)
    status = STATUS_EXACT

    # 2. Case-insensitive match.
    if pos == -1:
        m = re.search(re.escape(query), sentence, flags=re.IGNORECASE)
        if m:
            pos, status = m.start(), STATUS_CASE_INSENSITIVE

    if pos != -1:
        start, end = trim_edges(sentence, pos, pos + len(query))
        return {"text": sentence[start:end], "start_char": start,
                "end_char": end, "status": status}

    # 3. Fuzzy token-overlap fallback.
    overlap = _token_overlap(sentence, query)
    if overlap:
        start, end = trim_edges(sentence, *overlap)
        return {"text": sentence[start:end], "start_char": start,
                "end_char": end, "status": STATUS_FUZZY}

    # 4. Give up; annotator will set the span manually.
    return {"text": query, "start_char": -1, "end_char": -1,
            "status": STATUS_UNRESOLVED}


def span_text(sentence: str, start: int, end: int) -> str:
    """Safe slice for rendering / consistency checks."""
    n = len(sentence)
    a = max(0, min(n, int(start)))
    b = max(0, min(n, int(end)))
    if b < a:
        a, b = b, a
    return sentence[a:b]
