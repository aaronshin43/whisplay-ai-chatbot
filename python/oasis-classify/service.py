"""
service.py — Flask API for oasis-classify (:5002).

POST /dispatch  — main dispatch endpoint
GET  /health    — health check
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from flask import Flask, jsonify, request

from config import (
    THRESHOLD_PATH_NETWORK_ERROR,   # noqa: F401 — exported for TypeScript contract
    THRESHOLD_PATH_SERVICE_ERROR,   # noqa: F401
    THRESHOLD_PATH_INVALID_SCHEMA,  # noqa: F401
)
from fast_match import normalize, tier0_lookup, is_direct_response, GENERIC_HELP_RESPONSE
from classifier import classify, DispatchResult, OOD_RESPONSE_TEXT
from prompt_builder import build_prompt, resolve_categories
from triage import build_triage_prompt

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Core dispatch function (used by both HTTP handler and tests)
# ---------------------------------------------------------------------------

def dispatch(query: str, prev_triage_hint: str | None = None) -> DispatchResult:
    """Run the full dispatch pipeline for a query.

    normalize() is called ONCE here. The normalized string is passed to both
    Tier 0 lookups and the classifier (which passes it to gte-small).

    Args:
        query: Raw user query (may contain ASR noise).
        prev_triage_hint: Category ID from a previous triage turn, or None.

    Returns:
        DispatchResult with all fields populated.
    """
    t_start = time.perf_counter()

    # --- Normalize once at pipeline entry ---
    norm = normalize(query)

    # --- Tier 0 fast path ---
    tier0_result, tier0_path = tier0_lookup(norm)   # pass already-normalized string

    if tier0_result is not None:
        latency_ms = (time.perf_counter() - t_start) * 1000
        if is_direct_response(tier0_result):
            return DispatchResult(
                mode="direct_response",
                response_text=tier0_result,
                system_prompt=None,
                category=None,
                top3=[],
                score=None,
                threshold_path=tier0_path,
                latency_ms=latency_ms,
                hint_changed_result=False,
            )
        else:
            # Category ID returned by Tier 0 — build the LLM prompt directly
            category = tier0_result
            prompt = build_prompt(query=norm, primary_category=category)
            return DispatchResult(
                mode="llm_prompt",
                response_text=None,
                system_prompt=prompt,
                category=category,
                top3=[],
                score=None,
                threshold_path=tier0_path,
                latency_ms=latency_ms,
                hint_changed_result=False,
            )

    # --- Tier 1 classifier ---
    result = classify(normalized_query=norm, prev_triage_hint=prev_triage_hint)
    # Patch latency to include Tier 0 overhead (classifier resets its own timer)
    result.latency_ms = (time.perf_counter() - t_start) * 1000

    if result.mode == "llm_prompt":
        # Build multi-label-aware prompt
        categories = resolve_categories(result.top3)
        primary = categories[0] if categories else result.category
        secondary = categories[1] if len(categories) > 1 else None
        result.system_prompt = build_prompt(
            query=norm,
            primary_category=primary,
            secondary_category=secondary,
        )

    elif result.mode == "triage_prompt":
        result.system_prompt = build_triage_prompt(query=norm)

    return result


# ---------------------------------------------------------------------------
# Tier 0 normalization note:
# tier0_lookup() already accepts a normalized string here because we pass
# norm instead of query. The internal normalize() call inside tier0_lookup()
# is a no-op for an already-normalized string (idempotent).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "oasis-classify", "port": 5002})


@app.route("/dispatch", methods=["POST"])
def dispatch_endpoint():
    payload: dict[str, Any] = request.get_json(force=True, silent=True) or {}

    query: str = payload.get("query", "")
    prev_triage_hint: str | None = payload.get("prev_triage_hint", None)

    if not isinstance(query, str) or not query.strip():
        return jsonify({
            "mode": "ood_response",
            "response_text": GENERIC_HELP_RESPONSE,
            "system_prompt": None,
            "category": None,
            "top3": [],
            "score": None,
            "threshold_path": "ood_floor",
            "latency_ms": 0.0,
            "hint_changed_result": False,
        }), 200

    result = dispatch(query=query, prev_triage_hint=prev_triage_hint)
    return jsonify(dataclasses.asdict(result)), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)
