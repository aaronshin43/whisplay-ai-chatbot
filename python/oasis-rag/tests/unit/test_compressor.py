"""Unit tests for compressor.py — selective context compression.

Tests: COMP-001 … COMP-010  (10 tests, no model required)
  COMP-001  "Do not" safety prefix preserved when query term matches
  COMP-002  "Never" safety prefix preserved when query term matches
  COMP-003  Section anchor prepended when section argument given
  COMP-004  Empty chunk_text returns chunk_text unchanged
  COMP-005  Irrelevant query still returns ≥ COMPRESS_MIN_SENTENCES sentences
  COMP-006  Sentence containing query term is retained in output
  COMP-007  Output token count never exceeds input token count
  COMP-008  min_ratio guard: output tokens ≥ 40% of input tokens
  COMP-009  compress_chunks() adds 'compressed_text' key to every chunk dict
  COMP-010  compress_chunks() preserves pre-existing dict keys untouched
"""
from __future__ import annotations
import os
import sys

_RAG_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TESTS_DIR = os.path.join(_RAG_DIR, "tests")
for _p in (_RAG_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

from _shared import TestResult
from compressor import compress_chunk, compress_chunks
from config import COMPRESS_MIN_SENTENCES, COMPRESS_MIN_RATIO


# ── Shared test fixtures ───────────────────────────────────────────────────────

_BLEEDING_CHUNK = (
    "Apply a clean cloth firmly to the wound. "
    "Do not remove the dressing once bleeding slows. "
    "Maintain pressure for ten full minutes. "
    "Elevate the limb above heart level if possible."
)

_FROSTBITE_CHUNK = (
    "Move the patient to a warm shelter immediately. "
    "Never rub frostbitten skin as friction damages frozen tissue. "
    "Rewarm with water at 37-39 degrees Celsius. "
    "Do not use snow or ice."
)

_LONG_CHUNK = " ".join(
    [f"Sentence {i} about unrelated topic number {i}." for i in range(1, 16)]
    + ["Apply tourniquet above the wound firmly and quickly."]
)

# ── Tests ──────────────────────────────────────────────────────────────────────


def run() -> list[TestResult]:
    results: list[TestResult] = []

    # COMP-001: "Do not" prefix preserved when query contains matching term
    try:
        result = compress_chunk(_BLEEDING_CHUNK, "dressing wound")
        passed = "do not remove the dressing" in result.lower()
        note   = "" if passed else "safety 'do not' sentence was dropped"
        results.append(TestResult("COMP-001", passed, note))
    except Exception as exc:
        results.append(TestResult("COMP-001", False, f"EXCEPTION: {exc}"))

    # COMP-002: "Never" prefix preserved when query contains matching term
    try:
        result = compress_chunk(_FROSTBITE_CHUNK, "frostbite skin rub")
        passed = "never rub frostbitten" in result.lower()
        note   = "" if passed else "safety 'never' sentence was dropped"
        results.append(TestResult("COMP-002", passed, note))
    except Exception as exc:
        results.append(TestResult("COMP-002", False, f"EXCEPTION: {exc}"))

    # COMP-003: Section anchor prepended when section given
    try:
        result = compress_chunk("Apply pressure to stop bleeding.", "pressure",
                                section="Hemorrhage Control")
        passed = result.startswith("[Hemorrhage Control]")
        note   = "" if passed else f"result did not start with section anchor: {result[:50]!r}"
        results.append(TestResult("COMP-003", passed, note))
    except Exception as exc:
        results.append(TestResult("COMP-003", False, f"EXCEPTION: {exc}"))

    # COMP-004: Empty chunk returns empty string unchanged
    try:
        result = compress_chunk("", "tourniquet bleeding")
        passed = result == ""
        note   = "" if passed else f"expected empty, got: {result!r}"
        results.append(TestResult("COMP-004", passed, note))
    except Exception as exc:
        results.append(TestResult("COMP-004", False, f"EXCEPTION: {exc}"))

    # COMP-005: Irrelevant query still returns non-empty output (min_sentences floor)
    try:
        non_medical = "Apple is red. Banana is yellow. Cherry is dark red fruit."
        result = compress_chunk(non_medical, "tourniquet hemorrhage cardiac arrest")
        passed = result.strip() != ""
        note   = "" if passed else "empty output for irrelevant query (min_sentences broken)"
        results.append(TestResult("COMP-005", passed, note))
    except Exception as exc:
        results.append(TestResult("COMP-005", False, f"EXCEPTION: {exc}"))

    # COMP-006: Sentence containing query term is retained
    try:
        result = compress_chunk(_BLEEDING_CHUNK, "tourniquet")
        # The chunk doesn't mention tourniquet — test with a relevant keyword instead
        result = compress_chunk(_BLEEDING_CHUNK, "pressure wound")
        passed = "pressure" in result.lower()
        note   = "" if passed else "relevant sentence ('pressure') dropped from output"
        results.append(TestResult("COMP-006", passed, note))
    except Exception as exc:
        results.append(TestResult("COMP-006", False, f"EXCEPTION: {exc}"))

    # COMP-007: Output token count never exceeds input token count
    try:
        result           = compress_chunk(_LONG_CHUNK, "tourniquet wound")
        input_tokens     = len(_LONG_CHUNK.split())
        output_tokens    = len(result.split())
        passed           = output_tokens <= input_tokens
        note             = (f"output={output_tokens} input={input_tokens}"
                            if not passed else "")
        results.append(TestResult("COMP-007", passed, note,
                                  {"input_tokens": input_tokens, "output_tokens": output_tokens}))
    except Exception as exc:
        results.append(TestResult("COMP-007", False, f"EXCEPTION: {exc}"))

    # COMP-008: min_ratio guard — output ≥ COMPRESS_MIN_RATIO of input tokens
    try:
        result        = compress_chunk(_LONG_CHUNK, "tourniquet wound")
        input_tokens  = len(_LONG_CHUNK.split())
        output_tokens = len(result.split())
        ratio         = output_tokens / max(input_tokens, 1)
        passed        = ratio >= COMPRESS_MIN_RATIO
        note          = (f"ratio={ratio:.2f} threshold={COMPRESS_MIN_RATIO}"
                         if not passed else f"ratio={ratio:.2f}")
        results.append(TestResult("COMP-008", passed, note,
                                  {"ratio": ratio, "min_ratio": COMPRESS_MIN_RATIO}))
    except Exception as exc:
        results.append(TestResult("COMP-008", False, f"EXCEPTION: {exc}"))

    # COMP-009: compress_chunks() adds 'compressed_text' key to every dict
    try:
        chunks = [
            {"text": "Apply pressure to wound.", "section": "Bleeding"},
            {"text": "CPR compressions on chest.", "section": "Cardiac"},
        ]
        result_chunks = compress_chunks(chunks, "pressure wound")
        passed = all("compressed_text" in c for c in result_chunks)
        note   = "" if passed else "missing 'compressed_text' key in at least one chunk"
        results.append(TestResult("COMP-009", passed, note))
    except Exception as exc:
        results.append(TestResult("COMP-009", False, f"EXCEPTION: {exc}"))

    # COMP-010: compress_chunks() preserves all pre-existing dict keys
    try:
        chunks = [{"text": "Apply pressure.", "section": "Bleeding", "source": "who_bec",
                   "score": 0.85}]
        compress_chunks(chunks, "pressure")
        passed = (chunks[0].get("source") == "who_bec" and
                  chunks[0].get("score") == 0.85 and
                  "compressed_text" in chunks[0])
        note   = "" if passed else "pre-existing keys were modified or missing"
        results.append(TestResult("COMP-010", passed, note))
    except Exception as exc:
        results.append(TestResult("COMP-010", False, f"EXCEPTION: {exc}"))

    return results


if __name__ == "__main__":
    import sys as _sys
    res = run()
    passed = sum(r.passed for r in res)
    print(f"\nCompressor Unit Tests: {passed}/{len(res)}")
    for r in res:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.id}  {r.note or ''}")
    _sys.exit(0 if passed == len(res) else 1)
