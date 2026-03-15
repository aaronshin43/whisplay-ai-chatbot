"""
test_retriever.py — O.A.S.I.S. RAG Phase 3

Integration test for the 3-Stage Hybrid Retriever.

Runs 5 realistic emergency queries and prints a structured report for each:
  - Stage 1 candidate count
  - Stage 2 top-k results (source, section, scores)
  - Stage 3 compression ratio (original vs compressed tokens)
  - Final context preview (first 300 chars)

Usage:
    python python/oasis-rag/test_retriever.py
    python python/oasis-rag/test_retriever.py --verbose   (full context)
"""

from __future__ import annotations

import sys
import time
import textwrap
from pathlib import Path

# ── path setup so we can run from project root ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from indexer   import load_index
from retriever import Retriever, RetrievalResult

# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# Format: (label, query, expected_top_doc)
# ─────────────────────────────────────────────────────────────────────────────
TEST_QUERIES: list[tuple[str, str, str]] = [
    (
        "Severe Arm Bleeding",
        "there is blood everywhere from his arm",
        "severe_bleeding.md",
    ),
    (
        "Cardiac Arrest",
        "she collapsed and is not breathing",
        "cpr_adult.md",
    ),
    (
        "Anaphylaxis - Bee Sting",
        "throat is swelling after bee sting",
        "anaphylaxis.md",
    ),
    (
        "Choking",
        "something stuck in his throat cant breathe",
        "choking_adult.md",
    ),
    (
        "Panic Bleeding (All Caps)",
        "HELP THERE IS SO MUCH BLOOD",
        "severe_bleeding.md",
    ),
]

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
SEPARATOR = "=" * 70


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _token_count(text: str) -> int:
    return len(text.split())


def _compression_ratio(original: str, compressed: str) -> float:
    orig_tok = _token_count(original)
    comp_tok = _token_count(compressed)
    if orig_tok == 0:
        return 1.0
    return comp_tok / orig_tok


def _safe(s: str, width: int = 0) -> str:
    """ASCII-safe string for Windows cp949 console."""
    out = s.encode("ascii", errors="replace").decode("ascii")
    return textwrap.shorten(out, width=width, placeholder="...") if width else out


def _check_expected(result: RetrievalResult, expected_doc: str) -> str:
    """Return PASS/FAIL based on whether expected_doc appears in top chunks."""
    sources = [c.source for c in result.chunks]
    return "PASS" if any(expected_doc in s for s in sources) else "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(retriever: Retriever) -> list[dict]:
    results: list[dict] = []

    for label, query, expected_doc in TEST_QUERIES:
        print(f"\n{SEPARATOR}")
        print(f"  TEST: {label}")
        print(f"  Query: {_safe(query)!r}")
        print(SEPARATOR)

        t0 = time.perf_counter()
        result = retriever.retrieve(query)
        elapsed = (time.perf_counter() - t0) * 1000

        status = _check_expected(result, expected_doc)
        print(f"  Expected doc   : {expected_doc}  [{status}]")
        print(f"  Stage 1 cands  : {result.stage1_count}")
        print(f"  Stage 2 passing: {result.stage2_count}")
        print(f"  Top-k returned : {len(result.chunks)}")
        print(f"  Total latency  : {elapsed:.1f} ms")
        print()

        # Per-chunk detail
        for i, chunk in enumerate(result.chunks, start=1):
            orig_tok = _token_count(chunk.text)
            comp_tok = _token_count(chunk.compressed_text)
            ratio    = comp_tok / max(orig_tok, 1)
            reduction = (1.0 - ratio) * 100

            print(f"  Chunk {i}:")
            print(f"    Source       : {chunk.source}")
            print(f"    Section      : {_safe(chunk.section, width=60)}")
            print(f"    cosine={chunk.cosine_score:.3f}  "
                  f"lexical={chunk.lexical_score:.3f}  "
                  f"hybrid={chunk.hybrid_score:.3f}")
            print(f"    Tokens       : {orig_tok} → {comp_tok}  "
                  f"({reduction:.0f}% reduction)")

            if VERBOSE:
                print(f"\n    [Compressed text]")
                wrapped = textwrap.fill(
                    _safe(chunk.compressed_text),
                    width=66,
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
                print(wrapped)
            else:
                preview = _safe(chunk.compressed_text[:200])
                print(f"    Preview      : {preview!r}")
            print()

        # Final context summary
        context_tok = _token_count(result.context)
        print(f"  Context tokens : {context_tok}")
        if not VERBOSE:
            preview = _safe(result.context[:300])
            print(f"  Context preview:\n    {preview!r}")

        record = {
            "label":         label,
            "query":         query,
            "expected_doc":  expected_doc,
            "status":        status,
            "stage1_count":  result.stage1_count,
            "stage2_count":  result.stage2_count,
            "top_k_returned": len(result.chunks),
            "latency_ms":    elapsed,
            "context_tokens": context_tok,
        }
        results.append(record)

    return results


def print_summary(records: list[dict]) -> None:
    print(f"\n{SEPARATOR}")
    print("  SUMMARY")
    print(SEPARATOR)

    passed = sum(1 for r in records if r["status"] == "PASS")
    total  = len(records)
    avg_latency = sum(r["latency_ms"] for r in records) / max(total, 1)
    avg_ctx_tok = sum(r["context_tokens"] for r in records) / max(total, 1)

    print(f"  Passed         : {passed}/{total}")
    print(f"  Avg latency    : {avg_latency:.1f} ms")
    print(f"  Avg ctx tokens : {avg_ctx_tok:.0f}")
    print()

    header = f"  {'Test':<28} {'Status':<6} {'S1':>4} {'S2':>4} {'TopK':>5} {'ms':>7} {'CTok':>6}"
    print(header)
    print("  " + "-" * 62)
    for r in records:
        print(
            f"  {r['label']:<28} {r['status']:<6} "
            f"{r['stage1_count']:>4} {r['stage2_count']:>4} "
            f"{r['top_k_returned']:>5} {r['latency_ms']:>7.1f} "
            f"{r['context_tokens']:>6}"
        )

    print(SEPARATOR)
    if passed == total:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {total - passed} TEST(S) FAILED — check expected_doc mapping")
    print(SEPARATOR)


if __name__ == "__main__":
    print("\nO.A.S.I.S. RAG -- Retriever Integration Test")
    print(f"Mode: {'verbose' if VERBOSE else 'compact'}  (add --verbose for full context)")
    print()

    # Load index
    print("Loading index...")
    t_load = time.perf_counter()
    try:
        store = load_index()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Run `python python/oasis-rag/indexer.py` first.")
        sys.exit(1)

    retriever = Retriever(store)
    print(f"Index loaded: {store.chunk_count} chunks  ({(time.perf_counter()-t_load)*1000:.0f} ms)")

    # Run tests
    records = run_tests(retriever)

    # Summary
    print_summary(records)
