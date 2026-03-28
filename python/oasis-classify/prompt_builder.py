"""
prompt_builder.py — Assembles compact LLM prompts from pre-written manuals.

Handles multi-label: primary manual + short "also check" block pulled from
data/also_check_summaries.json. Enforces MAX_PROMPT_TOKENS hard ceiling with
tiktoken (cl100k_base) — never generates medical text at runtime.
"""

from __future__ import annotations

import json
import os

import tiktoken

from config import (
    ALSO_CHECK_PATH,
    MAX_PROMPT_TOKENS,
    MULTI_LABEL_RATIO,
    MAX_CATEGORIES,
    PRIORITY_CRITICAL,
    PRIORITY_URGENT,
)
from manual_store import get_manual

# ---------------------------------------------------------------------------
# Tokenizer (cl100k_base — lightweight, no transformers required)
# ---------------------------------------------------------------------------

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base."""
    return len(_ENCODING.encode(text))


# ---------------------------------------------------------------------------
# Load also_check_summaries at import time
# ---------------------------------------------------------------------------

def _load_also_check() -> dict[str, str]:
    if not os.path.isfile(ALSO_CHECK_PATH):
        return {}
    with open(ALSO_CHECK_PATH, encoding="utf-8") as fh:
        return json.load(fh)


_ALSO_CHECK: dict[str, str] = _load_also_check()


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are OASIS, a first-aid assistant.
Rules: Only use information on MANUAL. Numbered list only. One sentence per step. No extra text.

MANUAL:
{manual}

QUESTION: {query}
RESPONSE:"""

_ALSO_CHECK_TEMPLATE = "ALSO CHECK: {category} — {summary}"


# ---------------------------------------------------------------------------
# Priority ordering for multi-label conflict resolution
# ---------------------------------------------------------------------------

def _priority_rank(category_id: str) -> int:
    """Lower rank = higher priority."""
    if category_id in PRIORITY_CRITICAL:
        return 0
    if category_id in PRIORITY_URGENT:
        return 1
    return 2


def sort_categories_by_priority(categories: list[str], scores: dict[str, float]) -> list[str]:
    """Sort categories with PRIORITY_CRITICAL first, then PRIORITY_URGENT, then by score."""
    return sorted(
        categories,
        key=lambda c: (_priority_rank(c), -scores.get(c, 0.0)),
    )


# ---------------------------------------------------------------------------
# Multi-label resolution
# ---------------------------------------------------------------------------

def resolve_categories(
    top3: list[dict],
) -> list[str]:
    """Return up to MAX_CATEGORIES category IDs, filtered by MULTI_LABEL_RATIO.

    The primary category (highest score) is always included.
    Secondary is included only if score >= primary_score * MULTI_LABEL_RATIO.
    """
    if not top3:
        return []

    primary_score = top3[0]["score"]
    selected = [top3[0]["category"]]

    for entry in top3[1:MAX_CATEGORIES]:
        if entry["score"] >= primary_score * MULTI_LABEL_RATIO:
            selected.append(entry["category"])

    scores_map = {e["category"]: e["score"] for e in top3}
    return sort_categories_by_priority(selected, scores_map)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def build_prompt(
    query: str,
    primary_category: str,
    secondary_category: str | None = None,
) -> str:
    """Build the compact LLM prompt.

    Assembles primary manual + optional also-check block.
    Drops the secondary block if the combined prompt exceeds MAX_PROMPT_TOKENS.

    Args:
        query: Normalized user query.
        primary_category: Primary category ID.
        secondary_category: Optional secondary category ID.

    Returns:
        Assembled prompt string within MAX_PROMPT_TOKENS.
    """
    manual = get_manual(primary_category) or ""

    # Build primary-only prompt first
    primary_prompt = _PROMPT_TEMPLATE.format(manual=manual, query=query)

    if secondary_category and secondary_category != primary_category:
        also_check_summary = _ALSO_CHECK.get(secondary_category, "")
        if also_check_summary:
            also_check_block = _ALSO_CHECK_TEMPLATE.format(
                category=secondary_category.replace("_", " ").title(),
                summary=also_check_summary,
            )
            # Insert also-check block after the manual, before QUESTION
            manual_with_also = manual + "\n\n" + also_check_block
            combined_prompt = _PROMPT_TEMPLATE.format(
                manual=manual_with_also, query=query
            )
            if count_tokens(combined_prompt) <= MAX_PROMPT_TOKENS:
                return combined_prompt
            # Over token ceiling — drop secondary block entirely
    return primary_prompt
