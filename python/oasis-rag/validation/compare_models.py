"""
compare_models.py — O.A.S.I.S. LLM Model Comparison

Runs each model N times against the 20 LLM test cases and produces
a side-by-side comparison table with averaged metrics.

Usage:
    python compare_models.py                # 3 models × 3 runs (default)
    python compare_models.py --runs 2
    python compare_models.py --models gemma3:1b qwen3:0.6b

Pi5 latency estimate: multiply PC avg by PI5_FACTOR (default 12×).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Import test module first — it wraps sys.stdout for UTF-8 on Windows
_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from test_llm_response import make_tests, run_test   # noqa: E402 (side-effect: UTF-8 stdout)

import requests

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODELS = ["gemma3:1b", "qwen3:0.6b", "qwen3.5:0.8b"]
DEFAULT_RUNS   = 3
PI5_FACTOR     = 12   # empirical: Pi5 ≈ 12× slower than modern PC

CRITERIA = ["CONTENT_CORRECT", "FORMAT_CORRECT", "SAFE", "NO_HALLUCINATION"]
CRIT_SHORT = {"CONTENT_CORRECT": "CONTENT", "FORMAT_CORRECT": "FORMAT",
              "SAFE": "SAFE", "NO_HALLUCINATION": "NO_HALL"}

# Life-threatening test IDs — failures here are weighted more heavily
CRITICAL_IDS = {"LLM-001", "LLM-002", "LLM-003", "LLM-004",
                "LLM-005", "LLM-011", "LLM-012"}

RAG_URL    = "http://localhost:5001/health"
OLLAMA_URL = "http://localhost:11434/api/tags"


# ── Preflight ─────────────────────────────────────────────────────────────────
def check_services():
    for url, name in ((RAG_URL, "RAG :5001"), (OLLAMA_URL, "Ollama :11434")):
        try:
            r = requests.get(url, timeout=5)
            assert r.status_code == 200
            print(f"  [OK] {name}")
        except Exception:
            print(f"  [ERROR] {name} not reachable — start it first")
            sys.exit(1)


# ── Single model run ──────────────────────────────────────────────────────────
def run_one(model: str, tests: list[dict], run_idx: int, total_runs: int) -> list[dict]:
    """Run all tests for one model. Returns list of result dicts."""
    print(f"\n  [{model}] run {run_idx}/{total_runs}")
    results = []
    for i, tc in enumerate(tests, 1):
        sys.stdout.write(
            f"    [{i:02d}/{len(tests)}] {tc['id']} {tc['query'][:45]:<45} "
        )
        sys.stdout.flush()
        r = run_test(tc, model=model)
        status = "PASS" if r.passed else f"FAIL[{','.join(k for k,v in r.scores.items() if not v)}]"
        print(f"{status}  ({r.latency_ms:.0f}ms)")
        results.append({
            "id":        r.id,
            "passed":    r.passed,
            "scores":    r.scores,
            "latency_ms": r.latency_ms,
            "error":     r.error,
            "response":  r.response,
        })
    return results


# ── Aggregate N runs for one model ────────────────────────────────────────────
def aggregate(all_runs: list[list[dict]]) -> dict:
    """
    all_runs: list of N run-result-lists.
    Returns aggregated stats per test ID and overall.
    """
    n = len(all_runs)
    tests = all_runs[0]
    ids   = [r["id"] for r in tests]

    # Per-test: pass count across runs, per-criterion pass count
    per_test: dict[str, dict] = {}
    for tid in ids:
        pass_count = sum(
            1 for run in all_runs
            for r in run if r["id"] == tid and r["passed"]
        )
        crit_counts = {c: 0 for c in CRITERIA}
        lat_vals    = []
        for run in all_runs:
            for r in run:
                if r["id"] == tid:
                    for c in CRITERIA:
                        if r["scores"].get(c, False):
                            crit_counts[c] += 1
                    lat_vals.append(r["latency_ms"])
        per_test[tid] = {
            "pass_rate":   pass_count / n,
            "crit_rates":  {c: crit_counts[c] / n for c in CRITERIA},
            "avg_lat":     sum(lat_vals) / len(lat_vals) if lat_vals else 0,
            "min_pass":    pass_count == 0,   # never passed
        }

    # Overall
    all_results = [r for run in all_runs for r in run]
    total_pass  = sum(r["passed"] for r in all_results)
    total_tests = len(all_results)

    crit_overall = {
        c: sum(r["scores"].get(c, False) for r in all_results) / total_tests
        for c in CRITERIA
    }
    avg_lat = sum(r["latency_ms"] for r in all_results) / total_tests

    # Critical scenarios pass rate
    crit_results = [r for r in all_results if r["id"] in CRITICAL_IDS]
    crit_pass = sum(r["passed"] for r in crit_results) / len(crit_results) if crit_results else 0

    # Worst-run pass rate (reliability metric)
    run_totals = [sum(r["passed"] for r in run) / len(run) for run in all_runs]
    worst_run  = min(run_totals)

    return {
        "per_test":      per_test,
        "overall_rate":  total_pass / total_tests,
        "worst_run":     worst_run,
        "crit_scenario": crit_pass,
        "crit_overall":  crit_overall,
        "avg_lat_ms":    avg_lat,
        "pi5_est_ms":    avg_lat * PI5_FACTOR,
    }


# ── Print comparison table ────────────────────────────────────────────────────
def print_comparison(models: list[str], agg: dict[str, dict]):
    SEP  = "=" * 78
    SEP2 = "-" * 78

    print(f"\n{SEP}")
    print("  O.A.S.I.S.  Model Comparison Report")
    print(SEP)

    # ── Overall summary ───────────────────────────────────────────────────────
    col = 22
    header = f"{'Metric':<30}" + "".join(f"{m[:col]:>{col}}" for m in models)
    print(f"\n{header}")
    print(SEP2)

    def row(label, fn):
        print(f"  {label:<28}" + "".join(f"{fn(agg[m]):>{col}}" for m in models))

    row("Avg pass rate (all runs)",   lambda a: f"{a['overall_rate']*100:.1f}%")
    row("Worst-run pass rate",        lambda a: f"{a['worst_run']*100:.1f}%")
    row("Critical scenarios rate",    lambda a: f"{a['crit_scenario']*100:.1f}%")
    print(SEP2)
    for c in CRITERIA:
        row(f"  {CRIT_SHORT[c]}",     lambda a, c=c: f"{a['crit_overall'][c]*100:.1f}%")
    print(SEP2)
    row("PC avg latency",             lambda a: f"{a['avg_lat_ms']:.0f}ms")
    row(f"Pi5 est. ({PI5_FACTOR}×)",  lambda a: f"{a['pi5_est_ms']/1000:.0f}s")

    # ── Per-test detail ───────────────────────────────────────────────────────
    print(f"\n{'Test':<12}" + "".join(f"{m[:col]:>{col}}" for m in models))
    print(SEP2)
    all_ids = list(agg[models[0]]["per_test"].keys())
    for tid in all_ids:
        marker = " ★" if tid in CRITICAL_IDS else "  "
        vals = []
        for m in models:
            pt = agg[m]["per_test"][tid]
            pct = pt["pass_rate"] * 100
            lat = pt["avg_lat"]
            vals.append(f"{pct:4.0f}%  {lat:5.0f}ms")
        print(f"{marker}{tid:<10}" + "".join(f"{v:>{col}}" for v in vals))

    # ── Recommendation ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RECOMMENDATION")
    print(SEP2)

    def score(m):
        a = agg[m]
        # Disqualify if SAFE < 100% average
        if a["crit_overall"]["SAFE"] < 1.0:
            return -1
        # Composite: content 50% + worst_run 30% + speed bonus 20%
        speed_norm = 1.0 - min(a["pi5_est_ms"] / 120_000, 1.0)   # 120s = worst
        return (a["crit_overall"]["CONTENT_CORRECT"] * 0.50
                + a["worst_run"]                      * 0.30
                + speed_norm                          * 0.20)

    ranked = sorted(models, key=score, reverse=True)
    for i, m in enumerate(ranked, 1):
        a    = agg[m]
        sc   = score(m)
        safe = a["crit_overall"]["SAFE"] * 100
        note = ""
        if safe < 100:
            note = "  ⚠ SAFE < 100% — DISQUALIFIED"
        elif i == 1:
            note = "  ← RECOMMENDED"
        print(
            f"  #{i} {m:<22}  score={sc:.3f}"
            f"  pass={a['overall_rate']*100:.1f}%"
            f"  safe={safe:.0f}%"
            f"  Pi5≈{a['pi5_est_ms']/1000:.0f}s"
            f"{note}"
        )
    print(SEP)


# ── Save JSON ─────────────────────────────────────────────────────────────────
def save_results(models: list[str], all_data: dict[str, list[list[dict]]], agg: dict):
    out_dir  = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"models": models, "runs": all_data, "aggregated": agg},
                  f, indent=2, ensure_ascii=False)
    print(f"\n  Full results: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare LLM models for OASIS")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--runs",   type=int,  default=DEFAULT_RUNS,
                        help="Number of runs per model (default: 3)")
    args = parser.parse_args()

    print("\n  O.A.S.I.S. Model Comparison")
    print(f"  Models : {args.models}")
    print(f"  Runs   : {args.runs} per model")
    print(f"  Pi5 est: ×{PI5_FACTOR}\n")

    check_services()

    tests    = make_tests()
    all_data = {}
    agg      = {}

    for model in args.models:
        runs = []
        for r in range(1, args.runs + 1):
            runs.append(run_one(model, tests, r, args.runs))
        all_data[model] = runs
        agg[model]      = aggregate(runs)

    print_comparison(args.models, agg)
    save_results(args.models, all_data, agg)


if __name__ == "__main__":
    main()
