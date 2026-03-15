"""
O.A.S.I.S. RAG Validation Suite
Run: python python/oasis-rag/validation/run_all.py
"""
from __future__ import annotations
import os
import sys
import json
import datetime
import logging

# ── Path setup ────────────────────────────────────────────────────────────────
_VAL_DIR = os.path.dirname(os.path.abspath(__file__))
_RAG_DIR = os.path.join(_VAL_DIR, "..")
for p in (_VAL_DIR, _RAG_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Suppress noisy logs
for noisy in ("sentence_transformers", "faiss", "httpx",
              "transformers", "huggingface_hub", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

# ── Import RAG components ─────────────────────────────────────────────────────
from indexer  import load_index
from retriever import Retriever
import config

# ── Import test modules ───────────────────────────────────────────────────────
import test_retrieval_accuracy
import test_safety
import test_coverage
import test_edge_cases
import test_source_quality
import test_latency

RESULTS_DIR = os.path.join(_VAL_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

W = 60  # line width


def bar(char="─", w=W): return char * w
def header(title):       print(f"\n{bar('═')}\n  {title}\n{bar('═')}")
def section(title):      print(f"\n{bar('─')}\n  {title}\n{bar('─')}")


def summarize(results: list, label: str) -> tuple[int, int]:
    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    icon   = "✓" if passed == total else "✗"
    print(f"  {icon} {label}: {passed}/{total}")
    for r in results:
        if not r.passed:
            print(f"    FAIL [{r.id}] {r.note}")
    return passed, total


def save_json(name: str, data: dict):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def to_serializable(results):
    return [
        {"id": r.id, "passed": r.passed, "note": r.note, "details": r.details}
        for r in results
    ]


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Load index ────────────────────────────────────────────────────────────
    # Resolve index_dir relative to this file (avoids cwd-dependent config paths)
    _project_root = os.path.abspath(os.path.join(_VAL_DIR, "..", "..", ".."))
    _index_dir    = os.path.join(_project_root, "data", "rag_index")
    print("Loading RAG index …", end=" ", flush=True)
    store     = load_index(index_dir=_index_dir)
    retriever = Retriever(store)
    print(f"OK ({store.chunk_count} chunks)")

    header(f"O.A.S.I.S. RAG Validation Report\n  Date : {now}\n  Index: {store.chunk_count} chunks | Model: {config.EMBEDDING_MODEL}")

    grand_pass = grand_total = 0

    # ── Part 1: Retrieval Accuracy ────────────────────────────────────────────
    section("Part 1: Retrieval Accuracy")
    p1_results = test_retrieval_accuracy.run(retriever)

    groups = {
        "BLD": "Bleeding",
        "CPR": "CPR / Cardiac",
        "CHK": "Choking",
        "ANA": "Anaphylaxis",
        "SHK": "Shock",
        "TRM": "Trauma",
        "BRN": "Burns",
        "BRT": "Breathing",
        "AMS": "Altered Mental Status",
        "WLD": "Wilderness",
    }
    p1_pass = p1_total = 0
    for prefix, label in groups.items():
        subset = [r for r in p1_results if r.id.startswith(prefix)]
        p, t = summarize(subset, f"{prefix}: {label}")
        p1_pass += p; p1_total += t
    print(f"  {'─'*40}")
    print(f"  Total Part 1: {p1_pass}/{p1_total} ({100*p1_pass/p1_total:.1f}%)")
    grand_pass += p1_pass; grand_total += p1_total
    save_json("part1_retrieval_accuracy", to_serializable(p1_results))

    # ── Part 2: Safety ────────────────────────────────────────────────────────
    section("Part 2: Safety")
    p2_results = test_safety.run(retriever)
    p2_pass, p2_total = summarize(p2_results, "Safety checks")
    grand_pass += p2_pass; grand_total += p2_total
    save_json("part2_safety", to_serializable(p2_results))

    # ── Part 3: Coverage ──────────────────────────────────────────────────────
    section("Part 3: Scenario Coverage  (cosine ≥ 0.75)")
    p3_results = test_coverage.run(retriever)
    p3_pass, p3_total = summarize(p3_results, "Coverage scenarios")
    grand_pass += p3_pass; grand_total += p3_total
    save_json("part3_coverage", to_serializable(p3_results))

    # ── Part 4: Edge Cases ────────────────────────────────────────────────────
    section("Part 4: Edge Cases")
    p4_results = test_edge_cases.run(retriever)
    p4_pass, p4_total = summarize(p4_results, "Edge cases")
    grand_pass += p4_pass; grand_total += p4_total
    save_json("part4_edge_cases", to_serializable(p4_results))

    # ── Part 5: Medical Facts ─────────────────────────────────────────────────
    section("Part 5: Medical Source Quality")
    p5_results = test_source_quality.run(retriever)
    p5_pass, p5_total = summarize(p5_results, "Medical fact checks")
    grand_pass += p5_pass; grand_total += p5_total
    save_json("part5_source_quality", to_serializable(p5_results))

    # ── Part 6: Latency ───────────────────────────────────────────────────────
    section("Part 6: Latency Benchmark  (target < 200ms PC / < 2000ms Pi5)")
    print("  Running 4 × 20 queries …", end=" ", flush=True)
    p6_results, lat_summary = test_latency.run(retriever)
    print("done")
    for r in p6_results:
        icon = "✓" if r.passed else "✗"
        q = r.details.get("query", "")
        print(f"  {icon} [{r.id}] \"{q}…\" | {r.note}")
    print(f"\n  Overall latency avg={lat_summary['overall_avg_ms']:.1f}ms "
          f"p95={lat_summary['overall_p95_ms']:.1f}ms "
          f"max={lat_summary['overall_max_ms']:.1f}ms")
    pc_icon  = "PASS" if lat_summary["pc_pass"]  else "FAIL"
    pi5_icon = "PASS" if lat_summary["pi5_pass"] else "FAIL"
    print(f"  PC  target (<{lat_summary['pc_target_ms']:.0f}ms):  {pc_icon}")
    print(f"  Pi5 target (<{lat_summary['pi5_target_ms']:.0f}ms): {pi5_icon}")
    p6_pass  = sum(1 for r in p6_results if r.passed)
    p6_total = len(p6_results)
    grand_pass += p6_pass; grand_total += p6_total
    save_json("part6_latency", {"results": to_serializable(p6_results), "summary": lat_summary})

    # ── Grand Summary ─────────────────────────────────────────────────────────
    all_pass = grand_pass == grand_total
    verdict  = "ALL TESTS PASSED ✓  Ready for Pi5 deployment" if all_pass \
               else f"SOME TESTS FAILED — {grand_total - grand_pass} failure(s)"

    print(f"\n{bar('═')}")
    print(f"  OVERALL: {grand_pass}/{grand_total}  ({100*grand_pass/grand_total:.1f}%)")
    print(f"  {verdict}")
    print(bar("═"))
    print(f"  Results saved to: {RESULTS_DIR}/\n")

    # Save grand summary JSON
    save_json("summary", {
        "date": now,
        "index_chunks": store.chunk_count,
        "model": config.EMBEDDING_MODEL,
        "grand_pass": grand_pass,
        "grand_total": grand_total,
        "all_pass": all_pass,
        "latency": lat_summary,
    })

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
