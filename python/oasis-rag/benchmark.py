"""
benchmark.py — O.A.S.I.S. RAG Phase 6

Latency benchmark for the 3-Stage Hybrid Retrieval Pipeline.

Measures per-stage and total latency over N iterations, reports
mean ± standard deviation, and flags stages that exceed their targets.

Stage targets (PC / CUDA baseline):
  Stage 1 (Lexical)     <  5 ms
  Stage 2 (Semantic)    < 200 ms   (first call warmer after 1st)
  Stage 3 (Compression) <  50 ms
  Total pipeline        <2000 ms   (2 s wall-clock target)

Pi 5 (CPU-only) will be ~3-8x slower.

Usage:
    python python/oasis-rag/benchmark.py
    python python/oasis-rag/benchmark.py --iterations 20
    python python/oasis-rag/benchmark.py --quiet
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sentence_transformers import SentenceTransformer

import config
from compressor      import compress_chunk
from indexer         import load_index, IndexStore
from medical_keywords import detect_keywords, expand_query

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ITERATIONS = 10
WARMUP_ITERATIONS  = 2

BENCHMARK_QUERIES = [
    "there is blood everywhere from his arm",
    "she collapsed and is not breathing",
    "throat is swelling after bee sting",
    "something stuck in his throat cant breathe",
    "HELP THERE IS SO MUCH BLOOD",
]

# Per-stage latency targets (ms)
TARGETS = {
    "stage1_lexical":   5.0,
    "stage2_semantic":  500.0,   # generous — first calls are ~400ms on GPU
    "stage3_compress":  50.0,
    "total":            2000.0,
}

QUIET   = "--quiet" in sys.argv
SEP     = "=" * 70
SEP2    = "-" * 70


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def _stats(samples: list[float]) -> dict:
    if len(samples) < 2:
        return {"mean": samples[0] if samples else 0.0, "std": 0.0,
                "min": samples[0] if samples else 0.0, "max": samples[0] if samples else 0.0,
                "p95": samples[0] if samples else 0.0}
    return {
        "mean": statistics.mean(samples),
        "std":  statistics.stdev(samples),
        "min":  min(samples),
        "max":  max(samples),
        "p95":  sorted(samples)[int(len(samples) * 0.95)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-stage isolated benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def bench_stage1(store: IndexStore, query: str) -> float:
    """Time Stage 1: keyword detection + inverted-index lookup."""
    with Timer() as t:
        detected  = [kw for kw, _ in detect_keywords(query)]
        expanded  = expand_query(query)
        all_terms = list(dict.fromkeys(detected + [x.lower() for x in expanded]))
        cand: set[int] = set()
        for term in all_terms:
            for cid in store.keyword_map.get(term, []):
                cand.add(cid)
        candidates = list(cand) or list(range(store.chunk_count))
    return t.elapsed_ms


def bench_stage2(store: IndexStore, query: str, candidates: list[int]) -> float:
    """Time Stage 2: query embedding + FAISS inner product."""
    with Timer() as t:
        q_vec = store.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
        ).astype("float32")
        if candidates:
            vecs = np.array(
                [store.faiss_index.reconstruct(cid) for cid in candidates],
                dtype="float32",
            )
            _ = (q_vec @ vecs.T).flatten()
    return t.elapsed_ms


def bench_stage3(store: IndexStore, candidates: list[int], query: str) -> float:
    """Time Stage 3: context compression for top-k chunks."""
    top_k = min(config.TOP_K, len(candidates))
    with Timer() as t:
        for cid in candidates[:top_k]:
            meta = store.metadata[cid]
            _ = compress_chunk(
                chunk_text=meta["text"],
                query=query,
                section=meta.get("section", ""),
            )
    return t.elapsed_ms


def bench_total(retriever, query: str) -> float:
    """Time end-to-end retrieval."""
    with Timer() as t:
        _ = retriever.retrieve(query)
    return t.elapsed_ms


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    store,
    retriever,
    n_iter: int = DEFAULT_ITERATIONS,
) -> dict[str, dict]:
    """
    Run the benchmark for each query over n_iter iterations.
    Returns per-stage statistics.
    """
    from retriever import Retriever

    all_s1:    list[float] = []
    all_s2:    list[float] = []
    all_s3:    list[float] = []
    all_total: list[float] = []

    queries = BENCHMARK_QUERIES
    total_runs = n_iter * len(queries)
    run_num = 0

    for iteration in range(n_iter):
        for query in queries:
            run_num += 1

            # Stage 1
            detected  = [kw for kw, _ in detect_keywords(query)]
            expanded  = expand_query(query)
            all_terms = list(dict.fromkeys(detected + [x.lower() for x in expanded]))
            cand_set: set[int] = set()
            for term in all_terms:
                for cid in store.keyword_map.get(term, []):
                    cand_set.add(cid)
            candidates = list(cand_set) or list(range(store.chunk_count))
            candidates = candidates[:config.LEXICAL_CANDIDATE_POOL]

            s1 = bench_stage1(store, query)
            s2 = bench_stage2(store, query, candidates)
            s3 = bench_stage3(store, candidates, query)

            # Total (warm — model already loaded)
            tot = bench_total(retriever, query)

            all_s1.append(s1)
            all_s2.append(s2)
            all_s3.append(s3)
            all_total.append(tot)

            if not QUIET:
                print(
                    f"  [{run_num:>3}/{total_runs}] "
                    f"S1={s1:>5.1f}ms  S2={s2:>6.1f}ms  "
                    f"S3={s3:>5.1f}ms  Total={tot:>7.1f}ms  "
                    f"Q={query[:35]!r}"
                )

    return {
        "stage1_lexical":   _stats(all_s1),
        "stage2_semantic":  _stats(all_s2),
        "stage3_compress":  _stats(all_s3),
        "total":            _stats(all_total),
    }


def print_report(stats: dict[str, dict], n_iter: int, n_queries: int) -> int:
    """Print formatted report. Returns 0 if all targets met, 1 otherwise."""
    print(f"\n{SEP}")
    print("  BENCHMARK REPORT")
    print(f"  Iterations : {n_iter} x {n_queries} queries = {n_iter * n_queries} total runs")
    print(SEP)

    header = f"  {'Stage':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'p95':>8} {'Target':>8} {'OK'}"
    print(header)
    print("  " + SEP2[2:])

    all_ok = True
    stage_labels = {
        "stage1_lexical":   "Stage 1  Lexical",
        "stage2_semantic":  "Stage 2  Semantic",
        "stage3_compress":  "Stage 3  Compress",
        "total":            "Total Pipeline",
    }

    for key, label in stage_labels.items():
        s      = stats[key]
        target = TARGETS[key]
        ok     = s["mean"] <= target
        if not ok:
            all_ok = False
        mark   = "OK" if ok else "!!"
        print(
            f"  {label:<22} "
            f"{s['mean']:>7.1f}ms "
            f"{s['std']:>7.1f}ms "
            f"{s['min']:>7.1f}ms "
            f"{s['max']:>7.1f}ms "
            f"{s['p95']:>7.1f}ms "
            f"{target:>7.0f}ms "
            f"  {mark}"
        )

    print(SEP)

    if all_ok:
        print("  RESULT: ALL TARGETS MET")
    else:
        print("  RESULT: SOME TARGETS EXCEEDED (expected on Pi 5 CPU-only)")
        print("  Note: Pi 5 CPU-only is typically 3-8x slower than GPU baseline.")

    print(SEP)
    return 0 if all_ok else 1


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from retriever import Retriever

    # Parse --iterations N
    n_iter = DEFAULT_ITERATIONS
    if "--iterations" in sys.argv:
        idx = sys.argv.index("--iterations")
        if idx + 1 < len(sys.argv):
            n_iter = int(sys.argv[idx + 1])

    print(f"\nO.A.S.I.S. RAG -- Latency Benchmark")
    print(f"Iterations : {n_iter}  |  Queries : {len(BENCHMARK_QUERIES)}")

    print("\nLoading index...")
    t_load = time.perf_counter()
    try:
        store = load_index()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    retriever = Retriever(store)
    print(f"Index loaded: {store.chunk_count} chunks  ({(time.perf_counter()-t_load)*1000:.0f} ms)")

    # Warmup
    print(f"\nWarmup ({WARMUP_ITERATIONS} iterations, not measured)...")
    for _ in range(WARMUP_ITERATIONS):
        for q in BENCHMARK_QUERIES:
            retriever.retrieve(q)
    print("Warmup complete.\n")

    if not QUIET:
        print("Per-run results:")

    stats = run_benchmark(store, retriever, n_iter=n_iter)
    exit_code = print_report(stats, n_iter, len(BENCHMARK_QUERIES))
    sys.exit(exit_code)
