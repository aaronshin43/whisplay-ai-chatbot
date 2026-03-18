"""Unit tests for medical_keywords.py — keyword taxonomy, detection, and expansion.

Tests: MKW-001 … MKW-008  (8 tests, no model required)
  MKW-001  detect_keywords finds "cardiac arrest" → circulation_cardiac
  MKW-002  detect_keywords finds "bleeding" → hemorrhage_control
  MKW-003  detect_keywords finds "choking" → airway_management
  MKW-004  get_category("tourniquet") → "hemorrhage_control"
  MKW-005  expand_query("bleeding heavily") includes tourniquet (same category)
  MKW-006  expand_query("cardiac arrest") includes "CPR" (same category)
  MKW-007  detect_keywords on non-medical text → empty list
  MKW-008  MEDICAL_KEYWORDS frozenset contains known critical terms
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

from _shared import TestResult
from medical_keywords import MEDICAL_KEYWORDS, detect_keywords, expand_query, get_category


def run() -> list[TestResult]:
    results: list[TestResult] = []

    # MKW-001: detect_keywords("cardiac arrest") → at least one hit in circulation_cardiac
    try:
        hits     = detect_keywords("cardiac arrest has occurred")
        cats     = [cat for _, cat in hits]
        passed   = "circulation_cardiac" in cats
        note     = f"categories found: {cats}" if not passed else ""
        results.append(TestResult("MKW-001", passed, note, {"hits": hits}))
    except Exception as exc:
        results.append(TestResult("MKW-001", False, f"EXCEPTION: {exc}"))

    # MKW-002: detect_keywords("severe bleeding") → hit in hemorrhage_control
    try:
        hits   = detect_keywords("severe bleeding from the arm")
        cats   = [cat for _, cat in hits]
        passed = "hemorrhage_control" in cats
        note   = f"categories found: {cats}" if not passed else ""
        results.append(TestResult("MKW-002", passed, note, {"hits": hits}))
    except Exception as exc:
        results.append(TestResult("MKW-002", False, f"EXCEPTION: {exc}"))

    # MKW-003: detect_keywords("choking") → hit in airway_management
    try:
        hits   = detect_keywords("patient is choking on food")
        cats   = [cat for _, cat in hits]
        passed = "airway_management" in cats
        note   = f"categories found: {cats}" if not passed else ""
        results.append(TestResult("MKW-003", passed, note, {"hits": hits}))
    except Exception as exc:
        results.append(TestResult("MKW-003", False, f"EXCEPTION: {exc}"))

    # MKW-004: get_category("tourniquet") → "hemorrhage_control"
    try:
        cat    = get_category("tourniquet")
        passed = cat == "hemorrhage_control"
        note   = f"got: {cat!r}" if not passed else ""
        results.append(TestResult("MKW-004", passed, note, {"category": cat}))
    except Exception as exc:
        results.append(TestResult("MKW-004", False, f"EXCEPTION: {exc}"))

    # MKW-005: expand_query("bleeding heavily") → includes tourniquet (same category)
    try:
        expanded = [t.lower() for t in expand_query("bleeding heavily")]
        passed   = "tourniquet" in expanded
        note     = f"'tourniquet' not in expanded ({len(expanded)} terms)" if not passed else ""
        results.append(TestResult("MKW-005", passed, note,
                                  {"expanded_count": len(expanded), "has_tourniquet": "tourniquet" in expanded}))
    except Exception as exc:
        results.append(TestResult("MKW-005", False, f"EXCEPTION: {exc}"))

    # MKW-006: expand_query("cardiac arrest") → includes "CPR" (same category)
    try:
        expanded = [t.lower() for t in expand_query("cardiac arrest")]
        passed   = "cpr" in expanded
        note     = f"'CPR' not in expanded ({len(expanded)} terms)" if not passed else ""
        results.append(TestResult("MKW-006", passed, note,
                                  {"expanded_count": len(expanded), "has_cpr": "cpr" in expanded}))
    except Exception as exc:
        results.append(TestResult("MKW-006", False, f"EXCEPTION: {exc}"))

    # MKW-007: non-medical text → detect_keywords returns empty list
    try:
        hits   = detect_keywords("programming syntax errors in javascript code")
        passed = len(hits) == 0
        note   = f"unexpected hits: {hits}" if not passed else ""
        results.append(TestResult("MKW-007", passed, note, {"hits": hits}))
    except Exception as exc:
        results.append(TestResult("MKW-007", False, f"EXCEPTION: {exc}"))

    # MKW-008: MEDICAL_KEYWORDS frozenset contains known critical terms
    try:
        required = {"tourniquet", "cpr", "fracture", "airway", "bleeding", "seizure"}
        missing  = required - MEDICAL_KEYWORDS
        passed   = len(missing) == 0
        note     = f"missing from MEDICAL_KEYWORDS: {missing}" if not passed else ""
        results.append(TestResult("MKW-008", passed, note,
                                  {"required": list(required), "missing": list(missing)}))
    except Exception as exc:
        results.append(TestResult("MKW-008", False, f"EXCEPTION: {exc}"))

    return results


if __name__ == "__main__":
    import sys as _sys
    res = run()
    passed = sum(r.passed for r in res)
    print(f"\nMedical Keywords Unit Tests: {passed}/{len(res)}")
    for r in res:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.id}  {r.note or ''}")
    _sys.exit(0 if passed == len(res) else 1)
