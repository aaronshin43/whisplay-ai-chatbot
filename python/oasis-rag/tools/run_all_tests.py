"""
run_all_tests.py -- O.A.S.I.S. RAG Phase 6

Full integration test runner. Executes all test stages in order:

  Stage 0 -- Indexer     : rebuild index from data/knowledge/
  Stage 1 -- Retriever   : 5 query functional tests  (test_retriever.py)
  Stage 2 -- Accuracy    : 30 content-accuracy tests (test_accuracy.py)
  Stage 3 -- Benchmark   : latency measurement       (benchmark.py)

Each stage runs in-process (no subprocess), sharing the loaded index
so the model is loaded only once.

Usage:
    python tools/run_all_tests.py
    python tools/run_all_tests.py --skip-index   (use existing index)
    python tools/run_all_tests.py --skip-bench   (skip latency benchmark)
    python tools/run_all_tests.py --iterations 5 (benchmark iterations)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_RAG_DIR   = os.path.dirname(_TOOLS_DIR)
if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)
# Also ensure tools/ is on path so sibling imports work
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

SKIP_INDEX = "--skip-index" in sys.argv
SKIP_BENCH = "--skip-bench" in sys.argv
N_ITER     = 5   # default benchmark iterations when run via run_all_tests

if "--iterations" in sys.argv:
    idx = sys.argv.index("--iterations")
    if idx + 1 < len(sys.argv):
        N_ITER = int(sys.argv[idx + 1])

SEP  = "=" * 68
SEP2 = "-" * 68


# ─────────────────────────────────────────────────────────────────────────────
# Stage report helpers
# ─────────────────────────────────────────────────────────────────────────────

def _header(title: str, stage_num: int) -> None:
    print(f"\n{SEP}")
    print(f"  STAGE {stage_num}: {title}")
    print(SEP)


def _result(passed: bool, detail: str = "") -> None:
    mark = "PASS" if passed else "FAIL"
    msg  = f"  [{mark}]"
    if detail:
        msg += f"  {detail}"
    print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 -- Indexer
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_index() -> bool:
    _header("Indexer -- Rebuild Knowledge Index", 0)

    if SKIP_INDEX:
        print("  Skipped (--skip-index). Using existing index.")
        return True

    import config
    from indexer import Indexer

    knowledge_dir = config.KNOWLEDGE_DIR
    print(f"  Knowledge dir : {knowledge_dir}")

    try:
        t0      = time.perf_counter()
        indexer = Indexer(knowledge_dir=knowledge_dir)
        result  = indexer.build()
        elapsed = time.perf_counter() - t0

        chunk_count = result["chunk_count"]
        ok = chunk_count > 0
        _result(ok, f"{chunk_count} chunks indexed in {elapsed:.2f}s")

        # Verify artifacts exist
        index_dir = Path(result["index_dir"])
        for fname in [config.FAISS_INDEX_FILE, config.METADATA_FILE, config.KEYWORD_MAP_FILE]:
            fpath = index_dir / fname
            if fpath.exists():
                size_kb = fpath.stat().st_size // 1024
                print(f"  Artifact: {fname:<22} {size_kb:>5} KB")
            else:
                print(f"  MISSING artifact: {fname}")
                ok = False

        return ok

    except Exception as e:
        _result(False, str(e))
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 -- Retriever functional tests
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_retriever(store, retriever) -> bool:
    _header("Retriever -- Functional Tests (5 queries)", 1)

    from test_retriever import TEST_QUERIES, _check_expected, _safe, SEPARATOR

    passed_count = 0
    for label, query, expected_doc in TEST_QUERIES:
        t0     = time.perf_counter()
        result = retriever.retrieve(query)
        ms     = (time.perf_counter() - t0) * 1000
        status = _check_expected(result, expected_doc)

        ok = status == "PASS"
        if ok:
            passed_count += 1

        ctx_tok = len(result.context.split())
        print(
            f"  [{status}] {label:<28} "
            f"top={len(result.chunks)}  ctx={ctx_tok}tok  {ms:.0f}ms"
        )
        if not ok:
            sources = [c.source for c in result.chunks]
            print(f"         Expected: {expected_doc}  Got: {sources}")

    ok = passed_count == len(TEST_QUERIES)
    _result(ok, f"{passed_count}/{len(TEST_QUERIES)} queries matched expected document")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 -- Accuracy tests
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_accuracy(retriever) -> tuple[bool, float]:
    _header("Accuracy -- Content Correctness (30 test cases)", 2)

    from test_accuracy import TESTS, run_test, print_summary

    results   = [run_test(tc, retriever) for tc in TESTS]
    passed    = sum(1 for r in results if r.passed)
    total     = len(results)
    accuracy  = passed / total * 100

    # Compact per-category output
    categories = sorted({r.case.category for r in results})
    for cat in categories:
        cat_res  = [r for r in results if r.case.category == cat]
        cat_pass = sum(1 for r in cat_res if r.passed)
        mark     = "PASS" if cat_pass == len(cat_res) else "FAIL"
        print(f"  [{mark}] {cat:<22} {cat_pass}/{len(cat_res)}")

        # Show individual failures
        for r in cat_res:
            if not r.passed:
                q_safe = r.case.query.encode("ascii", errors="replace").decode("ascii")[:50]
                print(f"       !! {r.case.id}: {q_safe!r}")
                for f in r.failures:
                    print(f"           {f.encode('ascii', errors='replace').decode('ascii')}")

    ok = accuracy >= 90.0
    _result(ok, f"{passed}/{total} passed  ({accuracy:.1f}%  target >= 90%)")
    return ok, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 -- Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_benchmark(store, retriever) -> bool:
    _header(f"Benchmark -- Latency ({N_ITER} iterations, quiet mode)", 3)

    if SKIP_BENCH:
        print("  Skipped (--skip-bench).")
        return True

    from benchmark import (
        BENCHMARK_QUERIES, TARGETS, WARMUP_ITERATIONS,
        run_benchmark, _stats,
    )

    # Warmup
    print(f"  Warmup ({WARMUP_ITERATIONS} iterations)...")
    for _ in range(WARMUP_ITERATIONS):
        for q in BENCHMARK_QUERIES:
            retriever.retrieve(q)

    import benchmark as _bm
    _bm.QUIET = True   # suppress per-run lines inside this runner

    stats = run_benchmark(store, retriever, n_iter=N_ITER)

    all_ok = True
    for key, label in [
        ("stage1_lexical",   "Stage 1  Lexical"),
        ("stage2_semantic",  "Stage 2  Semantic"),
        ("stage3_compress",  "Stage 3  Compress"),
        ("total",            "Total Pipeline"),
    ]:
        s      = stats[key]
        target = TARGETS[key]
        ok     = s["mean"] <= target
        if not ok:
            all_ok = False
        mark = "OK" if ok else "!!"
        print(
            f"  [{mark}] {label:<20} "
            f"mean={s['mean']:>7.1f}ms  "
            f"std={s['std']:>6.1f}ms  "
            f"p95={s['p95']:>7.1f}ms  "
            f"target<{target:.0f}ms"
        )

    if not all_ok:
        print("  Note: targets are PC/GPU baselines. Pi5 CPU is 3-8x slower.")

    _result(all_ok, "All latency targets met" if all_ok else "Some targets exceeded (see above)")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary(
    stage_results: list[tuple[str, bool]],
    accuracy_pct:  float,
    total_elapsed: float,
) -> int:
    print(f"\n{SEP}")
    print("  FINAL INTEGRATION TEST SUMMARY")
    print(SEP)

    all_ok = True
    for stage_name, ok in stage_results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{mark}] {stage_name}")

    print(f"\n  Accuracy       : {accuracy_pct:.1f}%")
    print(f"  Total wall time: {total_elapsed:.1f}s")
    print(SEP)

    if all_ok:
        print("  ALL STAGES PASSED")
    else:
        failed = [n for n, ok in stage_results if not ok]
        print(f"  FAILED STAGES: {', '.join(failed)}")

    print(SEP)
    return 0 if all_ok else 1


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'#' * 68}")
    print("  O.A.S.I.S. RAG -- Full Integration Test Suite")
    print(f"{'#' * 68}")
    flags = []
    if SKIP_INDEX: flags.append("--skip-index")
    if SKIP_BENCH: flags.append("--skip-bench")
    flags.append(f"--iterations {N_ITER}")
    print(f"  Flags: {' '.join(flags)}")

    wall_start = time.perf_counter()
    stage_results: list[tuple[str, bool]] = []
    accuracy_pct = 0.0

    # ── Stage 0: Indexer ─────────────────────────────────────────────────────
    idx_ok = run_stage_index()
    stage_results.append(("Stage 0: Indexer", idx_ok))

    if not idx_ok and not SKIP_INDEX:
        print("\n[ABORT] Indexer failed. Cannot continue without a valid index.")
        sys.exit(1)

    # ── Load shared index (once) ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Loading shared index for remaining stages...")
    from indexer   import load_index
    from retriever import Retriever

    try:
        store     = load_index()
        retriever = Retriever(store)
        print(f"  Index loaded: {store.chunk_count} chunks")
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)

    # ── Stage 1: Retriever ────────────────────────────────────────────────────
    ret_ok = run_stage_retriever(store, retriever)
    stage_results.append(("Stage 1: Retriever", ret_ok))

    # ── Stage 2: Accuracy ─────────────────────────────────────────────────────
    acc_ok, accuracy_pct = run_stage_accuracy(retriever)
    stage_results.append(("Stage 2: Accuracy", acc_ok))

    # ── Stage 3: Benchmark ────────────────────────────────────────────────────
    bench_ok = run_stage_benchmark(store, retriever)
    stage_results.append(("Stage 3: Benchmark", bench_ok))

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - wall_start
    exit_code = print_final_summary(stage_results, accuracy_pct, total_elapsed)
    sys.exit(exit_code)
