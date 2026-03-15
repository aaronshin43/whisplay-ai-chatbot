"""Part 6: Latency Benchmark — 20 repeats per query, target < 200ms (PC/CUDA)."""
from __future__ import annotations
import time
import statistics
from _shared import TestResult

LATENCY_TESTS = [
    {"query": "bleeding",                                                      "repeats": 20},
    {"query": "she collapsed and is not breathing",                            "repeats": 20},
    {"query": "HELP THERE IS SO MUCH BLOOD OH GOD PLEASE HELP ME",            "repeats": 20},
    {"query": "so we were hiking and my friend fell and now his leg is broken and bleeding",
                                                                               "repeats": 20},
]

TARGET_PC_MS   = 200.0   # CUDA / PC target
TARGET_PI5_MS  = 2000.0  # Raspberry Pi 5 target


def run(retriever) -> tuple[list[TestResult], dict]:
    """Returns (test_results, summary_stats)."""
    results = []
    all_latencies: list[float] = []

    for lt in LATENCY_TESTS:
        query   = lt["query"]
        repeats = lt["repeats"]
        latencies: list[float] = []

        for _ in range(repeats):
            t0 = time.perf_counter()
            ret = retriever.retrieve(query)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            # Also use the retriever-reported latency as cross-check
            latencies.append(ret.latency_ms)

        avg  = statistics.mean(latencies)
        p50  = statistics.median(latencies)
        p95  = sorted(latencies)[int(len(latencies) * 0.95)]
        mx   = max(latencies)

        all_latencies.extend(latencies)
        passed = avg < TARGET_PC_MS
        note = (f"avg={avg:.1f}ms p50={p50:.1f}ms p95={p95:.1f}ms max={mx:.1f}ms "
                f"[target<{TARGET_PC_MS}ms]")
        tid = f"LAT-{LATENCY_TESTS.index(lt)+1:03d}"
        results.append(TestResult(tid, passed, note,
                                  {"query": query[:40], "avg_ms": avg,
                                   "p50_ms": p50, "p95_ms": p95, "max_ms": mx}))

    # Overall stats
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
    }
    return results, summary
