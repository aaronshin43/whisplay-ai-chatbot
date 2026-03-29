"""
classifier.py — Tier 1 centroid-based medical intent classifier.

Loads centroids.npy at import time and embeds queries with gte-small.
Returns a DispatchResult dataclass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from config import (
    CENTROIDS_PATH,
    CLASSIFY_THRESHOLD,
    CATEGORY_THRESHOLDS,
    OOD_FLOOR,
    TRIAGE_HINT_BOOST,
    TRIAGE_HINT_MIN_RELEVANCE,
    EMBEDDING_MODEL,
)
from categories import CATEGORY_IDS, CATEGORY_INDEX

# ---------------------------------------------------------------------------
# DispatchResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    mode: Literal["direct_response", "llm_prompt", "triage_prompt", "ood_response"]
    response_text: str | None
    system_prompt: str | None
    category: str | None       # ALWAYS set for triage_prompt; None for ood/direct
    top3: list[dict]           # [{"category": str, "score": float}, ...]
    score: float | None        # None for Tier 0 hits
    threshold_path: str
    latency_ms: float
    hint_changed_result: bool


# ---------------------------------------------------------------------------
# OOD pre-baked response
# ---------------------------------------------------------------------------

OOD_RESPONSE_TEXT = (
    "I am a first-aid assistant and can only help with medical emergencies. "
    "Please describe a medical situation and I will guide you. "
    "If this is an emergency, call 911."
)


# ---------------------------------------------------------------------------
# Model loading (lazy — only when first classify() call is made)
# ---------------------------------------------------------------------------

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# ---------------------------------------------------------------------------
# Centroid loading
# ---------------------------------------------------------------------------

_centroids: np.ndarray | None = None


def _get_centroids() -> np.ndarray:
    global _centroids
    if _centroids is None:
        _centroids = np.load(CENTROIDS_PATH)
    return _centroids


# ---------------------------------------------------------------------------
# Cosine similarity (batch query vector vs centroid matrix)
# ---------------------------------------------------------------------------

def _cosine_scores(query_vec: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return cosine similarity scores between query_vec and each centroid row."""
    # Normalize query vector
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    # Normalize centroid rows
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10
    c_norm = centroids / norms
    return c_norm @ q_norm  # (N,)


# ---------------------------------------------------------------------------
# classify() — main entry point
# ---------------------------------------------------------------------------

def classify(
    normalized_query: str,
    prev_triage_hint: str | None = None,
) -> DispatchResult:
    """Classify a normalized query using centroid cosine similarity.

    Args:
        normalized_query: Pre-normalized query string (run normalize() once before calling).
        prev_triage_hint: Category ID from a previous triage turn, or None.

    Returns:
        DispatchResult with mode, category, scores, and telemetry.
    """
    t_start = time.perf_counter()

    model = _get_model()
    centroids = _get_centroids()

    # Embed the normalized query
    query_vec = model.encode(normalized_query, normalize_embeddings=False)
    scores = _cosine_scores(query_vec, centroids)  # (N,)

    # ---------------------------------------------------------------------------
    # Triage hint boost
    # ---------------------------------------------------------------------------
    hint_changed_result = False
    if prev_triage_hint and prev_triage_hint in CATEGORY_INDEX:
        hint_idx = CATEGORY_INDEX[prev_triage_hint]
        hint_relevance = float(scores[hint_idx])
        if hint_relevance >= TRIAGE_HINT_MIN_RELEVANCE:
            original_top1 = int(np.argmax(scores))
            scores[hint_idx] += TRIAGE_HINT_BOOST
            hint_changed_result = (int(np.argmax(scores)) != original_top1)

    # ---------------------------------------------------------------------------
    # Route by top score
    # ---------------------------------------------------------------------------
    top3 = _build_top3(scores)
    best_idx = int(np.argmax(scores))
    best_category = CATEGORY_IDS[best_idx]
    best_score = float(scores[best_idx])

    latency_ms = (time.perf_counter() - t_start) * 1000

    # Two-layer OOD defense
    if best_score < OOD_FLOOR:
        return DispatchResult(
            mode="ood_response",
            response_text=OOD_RESPONSE_TEXT,
            system_prompt=None,
            category=None,
            top3=top3,
            score=best_score,
            threshold_path="ood_floor",
            latency_ms=latency_ms,
            hint_changed_result=hint_changed_result,
        )

    if best_category == "out_of_domain":
        return DispatchResult(
            mode="ood_response",
            response_text=OOD_RESPONSE_TEXT,
            system_prompt=None,
            category=None,
            top3=top3,
            score=best_score,
            threshold_path="ood_cluster",
            latency_ms=latency_ms,
            hint_changed_result=hint_changed_result,
        )

    # Per-category threshold override
    threshold = CATEGORY_THRESHOLDS.get(best_category, CLASSIFY_THRESHOLD)

    if best_score >= threshold:
        return DispatchResult(
            mode="llm_prompt",
            response_text=None,
            system_prompt=None,   # filled in by prompt_builder in service.py
            category=best_category,
            top3=top3,
            score=best_score,
            threshold_path="classifier_hit",
            latency_ms=latency_ms,
            hint_changed_result=hint_changed_result,
        )

    # Triage band [OOD_FLOOR, threshold)
    return DispatchResult(
        mode="triage_prompt",
        response_text=None,
        system_prompt=None,   # filled in by triage.py in service.py
        category=best_category,  # ALWAYS set for triage_prompt
        top3=top3,
        score=best_score,
        threshold_path="triage",
        latency_ms=latency_ms,
        hint_changed_result=hint_changed_result,
    )


# ---------------------------------------------------------------------------
# Top3 builder
# ---------------------------------------------------------------------------

def _build_top3(scores: np.ndarray) -> list[dict]:
    """Return top-3 categories as a list of dicts with 'category' and 'score'."""
    top_indices = np.argsort(scores)[::-1][:3]
    return [
        {"category": CATEGORY_IDS[int(i)], "score": round(float(scores[i]), 6)}
        for i in top_indices
    ]
