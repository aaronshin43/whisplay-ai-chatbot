"""
fast_match.py — Tier 0 fast path for oasis-classify.

Two branches based on normalized word count:
  - <= TIER0_MAX_WORDS: exact dict lookup in SHORT_QUERIES + edit distance 1 for very short tokens
  - >  TIER0_MAX_WORDS: exact dict lookup in SENTENCE_MATCHES only

ASR-robust normalization is applied once before any lookup.
"""

from __future__ import annotations

import json
import re

from config import (
    SHORT_QUERIES_PATH,
    SENTENCE_MATCHES_PATH,
    TIER0_MAX_WORDS,
)

# ---------------------------------------------------------------------------
# Pre-baked response text constants
# ---------------------------------------------------------------------------

GENERIC_HELP_RESPONSE = (
    "Please describe the emergency and I will help you. "
    "If this is life-threatening, call 911 immediately."
)

CALL_911_RESPONSE = (
    "Call 911 now. "
    "Stay on the line with the dispatcher — they will guide you."
)

_RESPONSE_KEYS: dict[str, str] = {
    "GENERIC_HELP": GENERIC_HELP_RESPONSE,
    "CALL_911": CALL_911_RESPONSE,
}

# ---------------------------------------------------------------------------
# Load Tier 0 data files at import time
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    # Strip the __comment key if present
    data.pop("__comment", None)
    return data


SHORT_QUERIES: dict[str, str] = _load_json(SHORT_QUERIES_PATH)
SENTENCE_MATCHES: dict[str, str] = _load_json(SENTENCE_MATCHES_PATH)


# ---------------------------------------------------------------------------
# ASR normalization
# ---------------------------------------------------------------------------

_HOMOPHONES: dict[str, str] = {
    "bleed in": "bleeding",
    "bled in": "bleeding",
    "seizing": "seizure",
    "sieze": "seizure",
    "strock": "stroke",
    "stroak": "stroke",
    "heart tack": "heart attack",
}


def normalize(text: str) -> str:
    """Normalize a raw STT/user query for consistent matching and embedding.

    Applied ONCE at pipeline entry. The normalized string is used for both
    Tier 0 lookups and the gte-small embedding call.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)   # strip punctuation
    text = re.sub(r"\s+", " ", text)       # collapse whitespace

    # Number word normalization
    text = text.replace("nine one one", "911")
    text = text.replace("nine eleven", "911")

    # Common ASR homophone fixes
    for wrong, right in _HOMOPHONES.items():
        text = text.replace(wrong, right)

    # Repeated-letter collapse (ASR stress artifacts: "heeelp" -> "help")
    # Collapse runs of 3+ identical characters down to 2
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Edit distance (Levenshtein) — only used for very short tokens
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # Simple O(mn) DP implementation — only called for short strings
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]


# ---------------------------------------------------------------------------
# Tier 0 lookup
# ---------------------------------------------------------------------------

def tier0_lookup(query: str) -> tuple[str | None, str | None]:
    """Attempt a Tier 0 fast-path match.

    Args:
        query: Raw (un-normalized) query string.

    Returns:
        (result, threshold_path) where:
          - result is a category ID, GENERIC_HELP_RESPONSE text, or CALL_911_RESPONSE text.
            None if no match was found.
          - threshold_path is "tier0_short" | "tier0_sentence" | None.
    """
    norm = normalize(query)
    words = norm.split()
    word_count = len(words)

    if word_count <= TIER0_MAX_WORDS:
        # --- Tier 0A: short query exact match ---
        if norm in SHORT_QUERIES:
            value = SHORT_QUERIES[norm]
            return _resolve_value(value), "tier0_short"

        # Edit distance 1 for very short normalized strings (<=6 chars)
        # Only applied for single tokens to avoid false positives
        if len(norm) <= 6:
            for key in SHORT_QUERIES:
                if _edit_distance(norm, key) == 1:
                    value = SHORT_QUERIES[key]
                    return _resolve_value(value), "tier0_short"

    else:
        # --- Tier 0B: sentence exact match only ---
        if norm in SENTENCE_MATCHES:
            value = SENTENCE_MATCHES[norm]
            return _resolve_value(value), "tier0_sentence"

    return None, None


def _resolve_value(value: str) -> str:
    """Convert a special response key to its text, or return category ID as-is."""
    return _RESPONSE_KEYS.get(value, value)


def is_direct_response(value: str) -> bool:
    """True if the Tier 0 result is a pre-baked text response (not a category ID)."""
    return value in (GENERIC_HELP_RESPONSE, CALL_911_RESPONSE)
