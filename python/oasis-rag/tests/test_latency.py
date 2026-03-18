"""Part 4: Latency Benchmark — end-to-end + per-stage.

End-to-end tests (LAT-001..004): 20 repeats per query, target < 200ms (PC/CUDA).
Stage tests (LAT-S1..S3): 5 queries × 5 repeats, one test per pipeline stage.
  LAT-S1  Stage 1 Lexical     avg < 5 ms
  LAT-S2  Stage 2 Semantic    avg < 500 ms
  LAT-S3  Stage 3 Compression avg < 50 ms

Stage tests require the IndexStore object; pass store=None to skip them.
"""
from __future__ import annotations
import time
import statistics
from _shared import TestResult

LATENCY_TESTS = [
    {"query": "bleeding",                                                                "repeats": 20},
    {"query": "she collapsed and is not breathing",                                      "repeats": 20},
    {"query": "HELP THERE IS SO MUCH BLOOD OH GOD PLEASE HELP ME",                      "repeats": 20},
    {"query": "so we were hiking and my friend fell and now his leg is broken and bleeding",
                                                                                         "repeats": 20},
]

# Queries used for per-stage bench (same as former benchmark.py)
_BENCH_QUERIES = [
    "there is blood everywhere from his arm",
    "she collapsed and is not breathing",
    "throat is swelling after bee sting",
    "something stuck in his throat cant breathe",
    "HELP THERE IS SO MUCH BLOOD",
]
_BENCH_REPEATS = 5  # fewer than standalone benchmark since this runs as part of suite

TARGET_PC_MS   = 200.0
TARGET_PI5_MS  = 2000.0
TARGET_S1_MS   = 5.0
TARGET_S2_MS   = 500.0
TARGET_S3_MS   = 50.0


def run(retriever, store=None) -> tuple[list[TestResult], dict]:
    """Returns (test_results, summary_stats).

    Parameters
    ----------
    retriever : Retriever
        Loaded retriever instance (required).
    store : IndexStore | None
        If provided, per-stage latency tests (LAT-S1..S3) are also run.
    """
    results: list[TestResult] = []
    all_latencies: list[float] = []

    # ── End-to-end tests (LAT-001..004) ──────────────────────────────────────
    for lt in LATENCY_TESTS:
        query   = lt["query"]
        repeats = lt["repeats"]
        latencies: list[float] = []

        for _ in range(repeats):
            ret = retriever.retrieve(query)
            latencies.append(ret.latency_ms)

        avg = statistics.mean(latencies)
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        mx  = max(latencies)

        all_latencies.extend(latencies)
        passed = avg < TARGET_PC_MS
        note   = (f"avg={avg:.1f}ms p50={p50:.1f}ms p95={p95:.1f}ms max={mx:.1f}ms "
                  f"[target<{TARGET_PC_MS}ms]")
        tid = f"LAT-{LATENCY_TESTS.index(lt)+1:03d}"
        results.append(TestResult(tid, passed, note,
                                  {"query": query[:40], "avg_ms": avg,
                                   "p50_ms": p50, "p95_ms": p95, "max_ms": mx}))

    # ── Per-stage tests (LAT-S1..S3) — only when IndexStore is available ─────
    stage_summary: dict = {}
    if store is not None:
        try:
            s1_times, s2_times, s3_times = _bench_stages(store)

            s1_avg = statistics.mean(s1_times)
            s2_avg = statistics.mean(s2_times)
            s3_avg = statistics.mean(s3_times)
            stage_summary = {
                "stage1_avg_ms": s1_avg,
                "stage2_avg_ms": s2_avg,
                "stage3_avg_ms": s3_avg,
            }

            results.append(TestResult(
                "LAT-S1", s1_avg < TARGET_S1_MS,
                f"Stage1 Lexical avg={s1_avg:.2f}ms [target<{TARGET_S1_MS}ms]",
                {"avg_ms": s1_avg, "target_ms": TARGET_S1_MS},
            ))
            results.append(TestResult(
                "LAT-S2", s2_avg < TARGET_S2_MS,
                f"Stage2 Semantic avg={s2_avg:.1f}ms [target<{TARGET_S2_MS}ms]",
                {"avg_ms": s2_avg, "target_ms": TARGET_S2_MS},
            ))
            results.append(TestResult(
                "LAT-S3", s3_avg < TARGET_S3_MS,
                f"Stage3 Compress avg={s3_avg:.2f}ms [target<{TARGET_S3_MS}ms]",
                {"avg_ms": s3_avg, "target_ms": TARGET_S3_MS},
            ))
        except Exception as exc:
            for tid in ("LAT-S1", "LAT-S2", "LAT-S3"):
                results.append(TestResult(tid, False, f"EXCEPTION: {exc}"))

    overall_avg = statistics.mean(all_latencies)
    overall_p95 = sorted(all_latencies)[int(len(all_latencies) * 0.95)]
    overall_max = max(all_latencies)

    summary = {
        "overall_avg_ms": overall_avg,
        "overall_p95_ms": overall_p95,
        "overall_max_ms": overall_max,
        "pc_target_ms":   TARGET_PC_MS,
        "pi5_target_ms":  TARGET_PI5_MS,
        "pc_pass":        overall_avg < TARGET_PC_MS,
        "pi5_pass":       overall_avg < TARGET_PI5_MS,
        **stage_summary,
    }
    return results, summary


def _bench_stages(store) -> tuple[list[float], list[float], list[float]]:
    """Isolated per-stage timing over _BENCH_QUERIES × _BENCH_REPEATS."""
    import numpy as np
    import config
    from medical_keywords import detect_keywords, expand_query
    from compressor import compress_chunk

    s1_times: list[float] = []
    s2_times: list[float] = []
    s3_times: list[float] = []

    for _ in range(_BENCH_REPEATS):
        for query in _BENCH_QUERIES:
            # Stage 1 — keyword detection + inverted-index lookup
            t0 = time.perf_counter()
            detected  = [kw for kw, _ in detect_keywords(query)]
            expanded  = expand_query(query)
            all_terms = list(dict.fromkeys(detected + [x.lower() for x in expanded]))
            cand_set: set[int] = set()
            for term in all_terms:
                for cid in store.keyword_map.get(term, []):
                    cand_set.add(cid)
            candidates = list(cand_set) or list(range(store.chunk_count))
            candidates = candidates[:config.LEXICAL_CANDIDATE_POOL]
            s1_times.append((time.perf_counter() - t0) * 1000)

            # Stage 2 — query embedding + FAISS inner product
            t0    = time.perf_counter()
            q_vec = store.model.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True,
            ).astype("float32")
            if candidates:
                vecs = np.array(
                    [store.faiss_index.reconstruct(cid) for cid in candidates],
                    dtype="float32",
                )
                _ = (q_vec @ vecs.T).flatten()
            s2_times.append((time.perf_counter() - t0) * 1000)

            # Stage 3 — context compression for top-k chunks
            t0    = time.perf_counter()
            top_k = min(config.TOP_K, len(candidates))
            for cid in candidates[:top_k]:
                meta = store.metadata[cid]
                compress_chunk(
                    chunk_text=meta["text"],
                    query=query,
                    section=meta.get("section", ""),
                )
            s3_times.append((time.perf_counter() - t0) * 1000)

    return s1_times, s2_times, s3_times
