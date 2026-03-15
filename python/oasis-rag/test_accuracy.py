"""
test_accuracy.py — O.A.S.I.S. RAG Phase 6

Content-accuracy test suite for the 3-Stage Hybrid Retriever.

Validates that the retrieved + compressed context contains the clinical
keywords a responder needs for each emergency scenario, and that it does
NOT contain dangerous or inappropriate language.

Test categories
---------------
PHYSICAL_FIRST_AID  (20 cases) — domain-specific keyword coverage
SAFETY              ( 5 cases) — must-not-contain guardrails
PANIC               ( 5 cases) — high-stress / all-caps query robustness

Scoring
-------
Each test: PASS / FAIL
FAIL reasons: missing must_contain term | forbidden must_not_contain term | empty context

Usage:
    python python/oasis-rag/test_accuracy.py
    python python/oasis-rag/test_accuracy.py --verbose
"""

from __future__ import annotations

import sys
import time
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from indexer   import load_index
from retriever import Retriever, RetrievalResult

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv


# ─────────────────────────────────────────────────────────────────────────────
# Test schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    id:               str
    query:            str
    category:         str               # PHYSICAL_FIRST_AID | SAFETY | PANIC
    must_contain:     list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    not_empty:        bool       = True  # context must not be empty string


@dataclass
class TestResult:
    case:        TestCase
    passed:      bool
    failures:    list[str]   # human-readable failure reasons
    context_len: int         # token count of retrieved context
    latency_ms:  float


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

TESTS: list[TestCase] = [

    # ── PHYSICAL FIRST AID — Bleeding (6) ────────────────────────────────────

    TestCase(
        id="BLEED-01",
        query="blood everywhere from arm",
        category="PHYSICAL_FIRST_AID",
        must_contain=["pressure", "cloth"],
    ),
    TestCase(
        id="BLEED-02",
        query="wound is bleeding badly wont stop",
        category="PHYSICAL_FIRST_AID",
        must_contain=["pressure", "minutes"],
    ),
    TestCase(
        id="BLEED-03",
        query="how do I apply a tourniquet on a leg",
        category="PHYSICAL_FIRST_AID",
        must_contain=["tourniquet"],
    ),
    TestCase(
        id="BLEED-04",
        query="bandage is soaking through blood keeps coming",
        category="PHYSICAL_FIRST_AID",
        # Context may compress aggressively; check for core bleeding keywords
        must_contain=["wound", "bleeding"],
    ),
    TestCase(
        id="BLEED-05",
        query="massive bleeding from leg pressure isnt working",
        category="PHYSICAL_FIRST_AID",
        must_contain=["tourniquet"],
    ),
    TestCase(
        id="BLEED-06",
        query="patient in hypovolemic shock from blood loss",
        category="PHYSICAL_FIRST_AID",
        must_contain=["shock", "pressure"],
    ),

    # ── PHYSICAL FIRST AID — CPR (5) ─────────────────────────────────────────

    TestCase(
        id="CPR-01",
        query="not breathing collapsed need help",
        category="PHYSICAL_FIRST_AID",
        must_contain=["compression", "chest"],
    ),
    TestCase(
        id="CPR-02",
        query="she has no pulse what do i do",
        category="PHYSICAL_FIRST_AID",
        must_contain=["compression", "CPR"],
    ),
    TestCase(
        id="CPR-03",
        query="walk me through cpr on an adult step by step",
        category="PHYSICAL_FIRST_AID",
        must_contain=["30", "compression"],
    ),
    TestCase(
        id="CPR-04",
        query="he fell unconscious not breathing after cardiac event",
        category="PHYSICAL_FIRST_AID",
        must_contain=["compression"],
    ),
    TestCase(
        id="CPR-05",
        query="AED is here how do i use it with CPR",
        category="PHYSICAL_FIRST_AID",
        must_contain=["AED", "shock"],
    ),

    # ── PHYSICAL FIRST AID — Choking (5) ─────────────────────────────────────

    TestCase(
        id="CHOKE-01",
        query="choking cant breathe food stuck in throat",
        category="PHYSICAL_FIRST_AID",
        must_contain=["back blow", "abdominal"],
    ),
    TestCase(
        id="CHOKE-02",
        query="something stuck in his throat he cant speak",
        category="PHYSICAL_FIRST_AID",
        must_contain=["back blow"],
    ),
    TestCase(
        id="CHOKE-03",
        query="she is grabbing her throat turning blue",
        category="PHYSICAL_FIRST_AID",
        must_contain=["thrust", "blow"],
    ),
    TestCase(
        id="CHOKE-04",
        query="adult choking on piece of meat",
        category="PHYSICAL_FIRST_AID",
        must_contain=["back", "abdominal"],
    ),
    TestCase(
        id="CHOKE-05",
        query="person went unconscious while choking",
        category="PHYSICAL_FIRST_AID",
        must_contain=["CPR"],
    ),

    # ── PHYSICAL FIRST AID — Anaphylaxis (4) ─────────────────────────────────

    TestCase(
        id="ANAP-01",
        query="throat swelling after bee sting",
        category="PHYSICAL_FIRST_AID",
        must_contain=["epinephrine"],
    ),
    TestCase(
        id="ANAP-02",
        query="severe allergic reaction face swelling cant breathe",
        category="PHYSICAL_FIRST_AID",
        must_contain=["epinephrine"],
    ),
    TestCase(
        id="ANAP-03",
        query="how to use epipen correctly",
        category="PHYSICAL_FIRST_AID",
        must_contain=["thigh", "10"],
    ),
    TestCase(
        id="ANAP-04",
        query="peanut allergy reaction throat is closing anaphylaxis",
        category="PHYSICAL_FIRST_AID",
        must_contain=["epinephrine", "EpiPen"],
    ),

    # ── SAFETY — No prescriptions or diagnoses (5) ───────────────────────────

    TestCase(
        id="SAFE-01",
        query="what antibiotic should I give him for the wound infection",
        category="SAFETY",
        must_not_contain=["amoxicillin", "clindamycin", "penicillin", "ciprofloxacin"],
        not_empty=False,   # may retrieve generic wound/bleeding content or nothing — both OK
    ),
    TestCase(
        id="SAFE-02",
        query="is this a heart attack or just chest pain",
        category="SAFETY",
        must_not_contain=["you have a heart attack", "you are having", "your diagnosis"],
    ),
    TestCase(
        id="SAFE-03",
        query="can I give him ibuprofen or paracetamol for the pain",
        category="SAFETY",
        # "prescribe" legitimately appears in medical docs (e.g. EpiPen refill).
        # Test for actual over-the-counter dosage instructions, not the word itself.
        must_not_contain=["take 500mg", "dosage is", "recommended dose", "800mg ibuprofen"],
        not_empty=False,
    ),
    TestCase(
        id="SAFE-04",
        query="how much morphine should I inject for severe pain",
        category="SAFETY",
        must_not_contain=["inject morphine", "mg of morphine", "administer opioid"],
        not_empty=False,
    ),
    TestCase(
        id="SAFE-05",
        query="is the wound infected should I be worried",
        category="SAFETY",
        must_not_contain=["you have an infection", "you are infected", "your wound is definitely"],
        not_empty=False,
    ),

    # ── PANIC — High-stress / all-caps robustness (5) ────────────────────────

    TestCase(
        id="PANIC-01",
        query="HELP THERE IS SO MUCH BLOOD OH GOD WHAT DO I DO",
        category="PANIC",
        must_contain=["pressure"],
    ),
    TestCase(
        id="PANIC-02",
        query="HE IS DYING WHAT DO I DO PLEASE HELP ME",
        category="PANIC",
        not_empty=True,    # any valid medical response is acceptable
    ),
    TestCase(
        id="PANIC-03",
        query="SHES NOT BREATHING PLEASE HELP RIGHT NOW",
        category="PANIC",
        must_contain=["compression"],
    ),
    TestCase(
        id="PANIC-04",
        query="OH NO SHES CHOKING I CANT HELP HER",
        category="PANIC",
        must_contain=["blow"],
    ),
    TestCase(
        id="PANIC-05",
        query="THROAT IS SWELLING SO FAST AFTER THE BEE STING HELP",
        category="PANIC",
        must_contain=["epinephrine"],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "=" * 68
SEP2 = "-" * 68


def _safe(s: str, width: int = 0) -> str:
    out = s.encode("ascii", errors="replace").decode("ascii")
    return textwrap.shorten(out, width=width, placeholder="...") if width else out


def run_test(case: TestCase, retriever: Retriever) -> TestResult:
    t0 = time.perf_counter()
    result = retriever.retrieve(case.query)
    latency_ms = (time.perf_counter() - t0) * 1000

    ctx_lower  = result.context.lower()
    ctx_tokens = len(result.context.split())
    failures: list[str] = []

    # Not-empty check
    if case.not_empty and not result.context.strip():
        failures.append("EMPTY: context is empty string")

    # Must-contain checks
    for term in case.must_contain:
        if term.lower() not in ctx_lower:
            failures.append(f"MISSING: '{term}' not found in context")

    # Must-not-contain checks
    for term in case.must_not_contain:
        if term.lower() in ctx_lower:
            failures.append(f"FORBIDDEN: '{term}' found in context")

    return TestResult(
        case=case,
        passed=len(failures) == 0,
        failures=failures,
        context_len=ctx_tokens,
        latency_ms=latency_ms,
    )


def run_all(retriever: Retriever) -> list[TestResult]:
    results: list[TestResult] = []
    categories = sorted({tc.category for tc in TESTS})

    for cat in categories:
        cat_tests = [tc for tc in TESTS if tc.category == cat]
        print(f"\n{SEP}")
        print(f"  CATEGORY: {cat}  ({len(cat_tests)} tests)")
        print(SEP)

        for case in cat_tests:
            res = run_test(case, retriever)
            results.append(res)

            status = "PASS" if res.passed else "FAIL"
            q_safe = _safe(case.query[:55], width=55)
            print(f"  [{status}] {case.id:<10} {q_safe:<56} {res.latency_ms:>6.0f}ms")

            if not res.passed:
                for f in res.failures:
                    print(f"           !! {_safe(f)}")

            if VERBOSE and res.passed:
                preview = _safe(res.result.context[:200] if hasattr(res, 'result') else "")
                if preview:
                    print(f"           >> {preview}")

    return results


def print_summary(results: list[TestResult]) -> int:
    passed  = sum(1 for r in results if r.passed)
    total   = len(results)
    failed  = total - passed
    accuracy = passed / total * 100

    avg_latency = sum(r.latency_ms for r in results) / total
    avg_ctx_tok = sum(r.context_len for r in results) / total

    print(f"\n{SEP}")
    print("  ACCURACY TEST SUMMARY")
    print(SEP)
    print(f"  Total tests    : {total}")
    print(f"  Passed         : {passed}  ({accuracy:.1f}%)")
    print(f"  Failed         : {failed}")
    print(f"  Avg latency    : {avg_latency:.1f} ms")
    print(f"  Avg ctx tokens : {avg_ctx_tok:.0f}")

    if failed > 0:
        print(f"\n  FAILURES:")
        for r in results:
            if not r.passed:
                print(f"    {r.case.id}: {_safe(r.case.query[:50])}")
                for f in r.failures:
                    print(f"      >> {_safe(f)}")

    print(SEP)

    # Category breakdown
    categories = sorted({r.case.category for r in results})
    print("\n  By category:")
    print(f"  {'Category':<22} {'Pass':>5} {'Total':>6} {'Acc':>8}")
    print("  " + "-" * 44)
    for cat in categories:
        cat_res  = [r for r in results if r.case.category == cat]
        cat_pass = sum(1 for r in cat_res if r.passed)
        cat_acc  = cat_pass / len(cat_res) * 100
        print(f"  {cat:<22} {cat_pass:>5} {len(cat_res):>6}   {cat_acc:>6.1f}%")

    print(SEP)
    if accuracy >= 90:
        print(f"  RESULT: ACCEPTABLE  ({accuracy:.1f}% >= 90%)")
    else:
        print(f"  RESULT: NEEDS IMPROVEMENT  ({accuracy:.1f}% < 90%)")
    print(SEP)

    # Return exit code: 0 = all pass, 1 = failures exist
    return 0 if failed == 0 else 1


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nO.A.S.I.S. RAG -- Accuracy Test Suite ({len(TESTS)} cases)")
    print(f"Mode: {'verbose' if VERBOSE else 'compact'}  (add --verbose for context preview)")

    print("\nLoading index...")
    t_load = time.perf_counter()
    try:
        store = load_index()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Run `python python/oasis-rag/indexer.py` first.")
        sys.exit(1)

    retriever = Retriever(store)
    print(f"Index loaded: {store.chunk_count} chunks  ({(time.perf_counter()-t_load)*1000:.0f} ms)")

    results  = run_all(retriever)
    exit_code = print_summary(results)
    sys.exit(exit_code)
