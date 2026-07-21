from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_MIN_SIGNIFICANT_LENGTH = 4


def significant_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.casefold()) if len(word) >= _MIN_SIGNIFICANT_LENGTH}


def is_grounded(candidate: str, source_text: str) -> bool:
    """Cheap defense against extraction/summarization prompt injection: reject
    LLM output that shares no vocabulary with the source it was derived from.
    Short candidates (no significant words, e.g. "3M") fall back to substring
    containment so legitimate short entities aren't rejected."""
    words = significant_words(candidate)
    if words:
        return bool(words & significant_words(source_text))
    return candidate.casefold().strip() in source_text.casefold()
