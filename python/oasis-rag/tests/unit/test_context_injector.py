"""Unit tests for context_injector.py — 22-signal injection engine.

Tests: CI-001 … CI-025  (25 tests, no model required)
  CI-001..022  one test per signal
  CI-023       burn blocked by eye-chemical (special mutual-exclusion rule)
  CI-024       lightning appends reminder at END of context
  CI-025       empty query → context unchanged (no injection)
"""
from __future__ import annotations
import os
import sys

# Path setup — works both when imported by run_all.py and run standalone
_RAG_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TESTS_DIR = os.path.join(_RAG_DIR, "tests")
for _p in (_RAG_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

from _shared import TestResult
from context_injector import inject_context

_BASE_CTX = "baseline context"

# (test_id, query, expected_phrase_in_result, signal_description)
_SIGNAL_TESTS = [
    ("CI-001", "she collapsed not breathing",           "100-120 compressions per minute",  "cardiac_arrest"),
    ("CI-002", "patient has neck injury",               "Do NOT move this person",           "spinal"),
    ("CI-003", "he has frostbite on fingers",           "Do NOT rub the frostbitten",        "frostbite"),
    ("CI-004", "theres so much blood everywhere",       "Apply direct PRESSURE",             "panic_blood"),
    ("CI-005", "we have no epipen available",           "NO EPINEPHRINE available",          "no_epipen"),
    ("CI-006", "lightning is coming what do we do",     "Trees are LETHAL in lightning",     "lightning"),
    ("CI-007", "burn on my arm",                        "cool running water for 20 minutes", "burn"),
    ("CI-008", "snake bit him on the ankle",            "Do NOT cut the wound",              "snakebite"),
    ("CI-009", "she stopped shivering hypothermia",     "Move the person to WARM shelter",   "hypothermia"),
    ("CI-010", "heat stroke hot skin not sweating",     "MOVE the person to shade",          "heat_stroke"),
    ("CI-011", "going into shock after accident",       "Lay the person flat",               "shock"),
    ("CI-012", "asthma attack wheezing",                "Sit the person UPRIGHT",            "asthma"),
    ("CI-013", "broken arm bone visible",               "do NOT try to straighten",          "fracture"),
    ("CI-014", "person is choking on food",             "5 firm BACK BLOWS",                 "choking"),
    ("CI-015", "think i'm having a heart attack",       "chew one adult aspirin",            "heart_attack"),
    ("CI-016", "she is having a seizure convulsing",    "DO NOT restrain",                   "seizure"),
    ("CI-017", "stroke face drooping arm weakness",     "FAST check",                        "stroke"),
    ("CI-018", "drowning pulled from water",            "5 RESCUE BREATHS first",            "drowning"),
    ("CI-019", "he swallowed bleach help",              "Do NOT induce vomiting",            "poisoning"),
    ("CI-020", "electric shock touched live wire",      "DO NOT TOUCH the person",           "electric_shock"),
    ("CI-021", "baby not breathing infant cpr needed",  "INFANT CPR",                        "infant_cpr"),
    ("CI-022", "chemical in eye pain",                  "FLUSH the eye",                     "eye_chemical"),
]


def run() -> list[TestResult]:
    results: list[TestResult] = []

    # CI-001..022: one test per signal — verify protocol phrase is injected
    for tid, query, phrase, desc in _SIGNAL_TESTS:
        try:
            result = inject_context(_BASE_CTX, query)
            passed = phrase.lower() in result.lower()
            note   = "" if passed else f"phrase not found: {phrase!r}"
            results.append(TestResult(tid, passed, note, {"signal": desc, "phrase": phrase}))
        except Exception as exc:
            results.append(TestResult(tid, False, f"EXCEPTION: {exc}"))

    # CI-023: eye-chemical signal blocks burn protocol (mutual exclusion)
    try:
        result       = inject_context(_BASE_CTX, "chemical in eye with burn")
        burn_present = "cool running water for 20 minutes" in result.lower()
        eye_present  = "flush the eye" in result.lower()
        passed       = eye_present and not burn_present
        note         = f"eye_injected={eye_present}  burn_injected={burn_present}"
        results.append(TestResult("CI-023", passed, note,
                                  {"eye_injected": eye_present, "burn_injected": burn_present}))
    except Exception as exc:
        results.append(TestResult("CI-023", False, f"EXCEPTION: {exc}"))

    # CI-024: lightning prepends protocol AND appends reminder after original context
    try:
        result          = inject_context(_BASE_CTX, "lightning storm coming")
        has_protocol    = "LIGHTNING SAFETY" in result
        has_reminder    = "FINAL REMINDER: TREES ARE LETHAL IN LIGHTNING" in result
        base_pos        = result.find(_BASE_CTX)
        reminder_pos    = result.find("FINAL REMINDER")
        reminder_after  = base_pos >= 0 and reminder_pos > base_pos
        passed          = has_protocol and has_reminder and reminder_after
        note            = (f"protocol={has_protocol}  reminder={has_reminder}  "
                           f"reminder_after_base={reminder_after}")
        results.append(TestResult("CI-024", passed, note))
    except Exception as exc:
        results.append(TestResult("CI-024", False, f"EXCEPTION: {exc}"))

    # CI-025: empty query → context returned unchanged (no signals detected)
    try:
        result = inject_context(_BASE_CTX, "")
        passed = result == _BASE_CTX
        note   = "" if passed else f"context was modified for empty query: {result!r}"
        results.append(TestResult("CI-025", passed, note))
    except Exception as exc:
        results.append(TestResult("CI-025", False, f"EXCEPTION: {exc}"))

    return results


if __name__ == "__main__":
    import sys as _sys
    res = run()
    passed = sum(r.passed for r in res)
    print(f"\nContext Injector Unit Tests: {passed}/{len(res)}")
    for r in res:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.id}  {r.note or ''}")
    _sys.exit(0 if passed == len(res) else 1)
