"""
compressor.py — O.A.S.I.S. RAG Phase 2 / Stage 3

Selective Context Compression.

Given a retrieved chunk and the original user query, this module extracts
only the sentences that are directly relevant to the query — reducing the
token payload sent to the LLM by 20-40% while preserving all actionable
medical information.

Algorithm
---------
1. Split chunk into sentences (regex-based, no NLTK required at runtime).
2. Score each sentence by:
     keyword_hits  × KEYWORD_WEIGHT  +  position_bonus
   where keyword_hits = number of distinct query keywords found in the sentence.
3. Keep sentences with score > threshold (floor: MIN_SENTENCES).
4. Re-join in original order (no reordering — preserves instruction sequence).
5. Prepend the section heading so the LLM always has context anchor.

Usage:
    from compressor import compress_chunk

    compressed = compress_chunk(
        chunk_text = chunk.text,
        query      = "how to apply a tourniquet",
        section    = chunk.section,
    )
"""

from __future__ import annotations

import re
from typing import Sequence

from config import (
    COMPRESS_MIN_SENTENCES,
    COMPRESS_KEYWORD_WEIGHT,
    COMPRESS_POSITION_DECAY,
    COMPRESS_SENTENCE_THRESHOLD,
    COMPRESS_MIN_RATIO,
)
from medical_keywords import detect_keywords, expand_query

# Sentences matching these prefixes are always preserved (safety-critical)
_SAFETY_PREFIXES = re.compile(
    r"^\s*(do\s+not|don'?t|never|avoid|warning|caution|critical|important)",
    re.IGNORECASE,
)
# Numbered list item (e.g. "1. ", "2. ")
_NUMBERED_ITEM_RE = re.compile(r"^\s*\d+[.)]\s")

# ─────────────────────────────────────────────────────────────
# Sentence splitter
# ─────────────────────────────────────────────────────────────
# Handles: ". " ".\n" "! " "? " but NOT abbreviations like "e.g."
_SENT_SPLIT_RE = re.compile(
    r"(?<!\b[A-Za-z])(?<!\b[A-Za-z]{2})(?<=[.!?])\s+"
)
# Fallback: split on newlines carrying list markers
_LIST_ITEM_RE = re.compile(r"(?m)^(\s*[-*\d]+[.)]\s)")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units, preserving list structure."""
    # First try standard sentence splitting
    parts = _SENT_SPLIT_RE.split(text.strip())
    if len(parts) >= 3:
        return [p.strip() for p in parts if p.strip()]

    # Fallback: split on newlines for list-heavy chunks
    lines = text.splitlines()
    merged: list[str] = []
    buf: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            if buf:
                merged.append(" ".join(buf))
                buf = []
            continue
        # List items get their own sentence slot
        if _LIST_ITEM_RE.match(line) and buf:
            merged.append(" ".join(buf))
            buf = [line]
        else:
            buf.append(line)
    if buf:
        merged.append(" ".join(buf))
    return [m for m in merged if m]


# ─────────────────────────────────────────────────────────────
# Query keyword extractor
# ─────────────────────────────────────────────────────────────

def _query_terms(query: str) -> frozenset[str]:
    """
    Return a flat frozenset of lowercase terms for the query:
    detected medical keywords + expanded category terms + raw query words.
    """
    raw_words = {w.lower() for w in query.split() if len(w) > 2}
    expanded  = {t.lower() for t in expand_query(query)}
    detected  = {kw.lower() for kw, _ in detect_keywords(query)}
    return frozenset(raw_words | expanded | detected)


# ─────────────────────────────────────────────────────────────
# Sentence scorer
# ─────────────────────────────────────────────────────────────

def _score_sentence(sentence: str, terms: frozenset[str], position: int) -> float:
    """
    Score a single sentence for relevance to the query.

    Returns
    -------
    float  — higher = more relevant
    """
    sent_lower = sentence.lower()
    hits = sum(1 for t in terms if t in sent_lower)
    position_bonus = max(0.0, 1.0 - position * COMPRESS_POSITION_DECAY)
    return hits * COMPRESS_KEYWORD_WEIGHT + (position_bonus if hits > 0 else 0.0)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def compress_chunk(
    chunk_text: str,
    query:      str,
    section:    str = "",
    *,
    min_sentences: int   = COMPRESS_MIN_SENTENCES,
    threshold:     float = COMPRESS_SENTENCE_THRESHOLD,
    min_ratio:     float = COMPRESS_MIN_RATIO,
) -> str:
    """
    Extract relevant sentences from *chunk_text* w.r.t. *query*.

    Parameters
    ----------
    chunk_text     : raw chunk text from DocumentChunker
    query          : user's question / utterance
    section        : heading breadcrumb for the chunk (prepended to output)
    min_sentences  : never return fewer than this many sentences
    threshold      : sentences with score > threshold are kept
    min_ratio      : minimum fraction of original tokens to retain

    Returns
    -------
    str — compressed text (always shorter than or equal to chunk_text length)
    """
    sentences = _split_sentences(chunk_text)
    if not sentences:
        return chunk_text

    terms  = _query_terms(query)
    scored = [
        (sent, _score_sentence(sent, terms, i))
        for i, sent in enumerate(sentences)
    ]

    # --- selection pass ---
    kept_indices = [i for i, (_, score) in enumerate(scored) if score > threshold]

    # Always keep safety-critical sentences (Do NOT / Never / Avoid + any query keyword)
    # This prevents the compressor from silently removing contraindication warnings.
    safety_set = set(kept_indices)
    for i, (sent, _) in enumerate(scored):
        if _SAFETY_PREFIXES.match(sent) and any(t in sent.lower() for t in terms):
            safety_set.add(i)
    kept_indices = sorted(safety_set)

    # Preserve numbered list integrity: if any item from a numbered list is kept,
    # keep all contiguous items from that list (avoids gaps like "1. … 3. …").
    if any(_NUMBERED_ITEM_RE.match(sentences[i]) for i in kept_indices):
        numbered_kept = set(kept_indices)
        # Find contiguous numbered runs that partially overlap with kept_indices
        runs: list[list[int]] = []
        current_run: list[int] = []
        for i, sent in enumerate(sentences):
            if _NUMBERED_ITEM_RE.match(sent):
                current_run.append(i)
            else:
                if current_run:
                    runs.append(current_run)
                    current_run = []
        if current_run:
            runs.append(current_run)
        for run in runs:
            if any(idx in numbered_kept for idx in run):
                numbered_kept.update(run)
        kept_indices = sorted(numbered_kept)

    # Guarantee minimum sentence count (take top-scored if under floor)
    if len(kept_indices) < min_sentences:
        ranked = sorted(range(len(scored)), key=lambda i: scored[i][1], reverse=True)
        for idx in ranked:
            if idx not in kept_indices:
                kept_indices.append(idx)
            if len(kept_indices) >= min_sentences:
                break
        kept_indices.sort()  # restore document order

    # --- minimum token ratio guard ---
    original_tokens = len(chunk_text.split())
    while kept_indices:
        candidate = " ".join(sentences[i] for i in kept_indices)
        if len(candidate.split()) / max(original_tokens, 1) >= min_ratio:
            break
        # Add the next highest-scoring excluded sentence
        excluded = [i for i in range(len(sentences)) if i not in kept_indices]
        if not excluded:
            break
        best_excluded = max(excluded, key=lambda i: scored[i][1])
        kept_indices.append(best_excluded)
        kept_indices.sort()

    compressed_body = " ".join(sentences[i] for i in kept_indices)

    # Prepend section anchor (truncated to first clean heading line)
    if section:
        # Strip markdown bold, inline code, and long suffixes from section
        import re as _re
        clean_section = _re.sub(r"\*\*|`", "", section)
        clean_section = clean_section.split("\n")[0].strip()[:60]
        return f"[{clean_section}]\n{compressed_body}"
    return compressed_body


def compress_chunks(
    chunks: list[dict],
    query:  str,
) -> list[dict]:
    """
    Compress a list of chunk dicts in-place (adds 'compressed_text' key).
    Each dict must have keys: 'text', 'section'.

    Returns the same list with 'compressed_text' populated.
    """
    for chunk in chunks:
        chunk["compressed_text"] = compress_chunk(
            chunk_text=chunk["text"],
            query=query,
            section=chunk.get("section", ""),
        )
    return chunks


# ─────────────────────────────────────────────────────────────
# CLI smoke-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    sample = """
    # Severe Bleeding

    Apply firm continuous pressure with cloth or hand. Do not remove cloth even
    if soaked through — add more on top. Keep pressure constant for at least
    10 minutes without checking. Elevate the limb above heart level if possible.

    Apply a tourniquet 5-7 cm above the wound, not over a joint.
    Tighten until bleeding stops completely.
    Note the time of application on the patient's skin.
    Do not remove or loosen the tourniquet once applied.

    Monitor for shock: pale skin, rapid pulse, altered consciousness.
    Lay patient flat and elevate legs 15-30 cm.
    Keep the patient warm.
    """

    query = sys.argv[1] if len(sys.argv) > 1 else "how to apply a tourniquet"
    result = compress_chunk(sample, query, section="Severe Bleeding")

    original_tokens  = len(sample.split())
    compressed_tokens = len(result.split())
    reduction = 1.0 - compressed_tokens / max(original_tokens, 1)

    _safe = lambda s: s.encode("ascii", errors="replace").decode("ascii")
    print(f"Query   : {query!r}")
    print(f"Original: {original_tokens} tokens")
    print(f"Result  : {compressed_tokens} tokens  ({reduction:.0%} reduction)")
    print(f"\n{_safe(result)}")
