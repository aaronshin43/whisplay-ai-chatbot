"""Shared utilities for OASIS RAG validation tests."""
from __future__ import annotations
import os
import sys
import re

# Add oasis-rag to path
_RAG_DIR = os.path.join(os.path.dirname(__file__), "..")
if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

# Suppress noisy logs during tests
import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from dataclasses import dataclass, field


@dataclass
class TestResult:
    id: str
    passed: bool
    note: str = ""
    details: dict = field(default_factory=dict)


def get_context_text(result) -> str:
    """Return full searchable text from a RetrievalResult (all top-k chunks)."""
    parts = [result.context or ""]
    for c in result.chunks:
        parts.append(c.text or "")
        parts.append(c.compressed_text or "")
    return " ".join(parts)


def context_contains(text: str, keywords: list[str]) -> list[str]:
    """Return list of keywords NOT found in text (case-insensitive)."""
    lower = text.lower()
    return [kw for kw in keywords if kw.lower() not in lower]


def context_has_forbidden(text: str, forbidden: list[str]) -> list[str]:
    """
    Return forbidden phrases found in context WITHOUT a preceding negation.
    Prevents false-positives from 'do not suck/rub/remove/release/move...' contexts.
    """
    lower = text.lower()
    found = []
    for phrase in forbidden:
        ph = phrase.lower()
        idx = lower.find(ph)
        while idx >= 0:
            prefix = lower[max(0, idx - 60):idx]   # 60-char window catches "do not pack ... ice"
            negated = any(neg in prefix for neg in
                          ["do not ", "don't ", "not ", "never ", "avoid ", "no "])
            if not negated:
                found.append(phrase)
                break
            idx = lower.find(ph, idx + 1)
    return found


def top_source(result) -> str:
    if result.chunks:
        return result.chunks[0].source
    return ""


def top_score(result) -> float:
    """Hybrid score of the top chunk."""
    if result.chunks:
        return result.chunks[0].hybrid_score
    return 0.0


def top_cosine(result) -> float:
    """Cosine score of the top chunk (more stable scale than hybrid)."""
    if result.chunks:
        return result.chunks[0].cosine_score
    return 0.0


def source_matches(source: str, pattern: str) -> bool:
    if not pattern:
        return True
    return bool(re.search(pattern, source))


def any_source_matches(result, pattern: str) -> bool:
    """True if ANY of the returned chunks matches the source pattern.
    More robust than checking only rank-1, since relevant info may be rank 2-3.
    """
    if not pattern:
        return True
    return any(source_matches(c.source, pattern) for c in result.chunks)
