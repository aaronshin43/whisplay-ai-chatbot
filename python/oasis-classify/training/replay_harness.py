"""
training/replay_harness.py — Regression check against prior session logs.

Runs classify() on each logged query and compares the result to the expected
category stored in the log. Flags queries that regressed (previously correct,
now incorrect).

Exit codes:
  0 — no regressions (or no log file found)
  1 — regressions detected (or regression rate exceeds --strict threshold)

Log format (one JSON object per line):
  {"query": str, "expected_category": str, "result": {DispatchResult as dict}}

Usage:
    python training/replay_harness.py --log path/to/session.jsonl
    python training/replay_harness.py --log path/to/session.jsonl --strict
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CLASSIFY_DIR = os.path.dirname(_SCRIPT_DIR)

# Ensure classify modules are importable
sys.path.insert(0, _CLASSIFY_DIR)

STRICT_REGRESSION_RATE = 0.02  # 2% — exit 1 if --strict and rate exceeds this


# ---------------------------------------------------------------------------
# Replay a single query
# ---------------------------------------------------------------------------

def _replay_query(query: str) -> str | None:
    """Run the current classify() pipeline on a query.

    Returns the predicted category, or None if OOD/direct_response.
    """
    from fast_match import normalize, tier0_lookup, is_direct_response
    from service import dispatch

    result = dispatch(query)
    return result.category


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def run_harness(log_path: str, strict: bool = False) -> int:
    """Run replay harness. Returns exit code (0 or 1)."""

    if not os.path.exists(log_path):
        print(f"No log file found at {log_path}. No history - no regressions.")
        return 0

    print(f"Loading session log: {log_path}")

    entries: list[dict] = []
    with open(log_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "query" not in entry or "expected_category" not in entry:
                    print(
                        f"  WARNING: line {lineno} missing 'query' or 'expected_category', skipping.",
                        file=sys.stderr,
                    )
                    continue
                entries.append(entry)
            except json.JSONDecodeError as exc:
                print(f"  WARNING: line {lineno} is not valid JSON: {exc}", file=sys.stderr)
                continue

    if not entries:
        print("Log file is empty or has no valid entries. No regressions.")
        return 0

    print(f"Loaded {len(entries)} entries. Replaying...")

    regressions: list[dict] = []
    errors: list[dict] = []

    for i, entry in enumerate(entries, 1):
        query = entry["query"]
        expected = entry["expected_category"]

        try:
            predicted = _replay_query(query)
        except Exception as exc:
            errors.append({"query": query, "expected": expected, "error": str(exc)})
            print(f"  [{i}/{len(entries)}] ERROR on '{query[:50]}': {exc}", file=sys.stderr)
            continue

        is_correct = (predicted == expected)

        # Check if previous result was also correct
        prior_result = entry.get("result", {})
        prior_category = prior_result.get("category")
        was_correct = (prior_category == expected)

        if was_correct and not is_correct:
            regressions.append({
                "query": query,
                "expected": expected,
                "previous": prior_category,
                "current": predicted,
            })

        if i % 50 == 0:
            print(f"  Progress: {i}/{len(entries)}, regressions so far: {len(regressions)}")

    total = len(entries)
    regression_count = len(regressions)
    regression_rate = regression_count / total if total > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"Replay complete: {total} entries")
    print(f"  Regressions: {regression_count} ({regression_rate:.2%})")
    print(f"  Errors: {len(errors)}")
    print(f"{'=' * 60}")

    if regressions:
        print("\nRegressed queries:")
        for r in regressions:
            print(
                f"  Query: {r['query'][:60]!r}\n"
                f"    Expected:  {r['expected']}\n"
                f"    Previous:  {r['previous']}\n"
                f"    Current:   {r['current']}\n"
            )

    if errors:
        print(f"\nQueries that caused errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e['query'][:60]!r}: {e['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    if regression_count > 0:
        print(f"\nFAIL: {regression_count} regression(s) detected.")
        return 1

    if strict and regression_rate > STRICT_REGRESSION_RATE:
        print(
            f"\nFAIL (--strict): regression rate {regression_rate:.2%} exceeds "
            f"{STRICT_REGRESSION_RATE:.0%} threshold."
        )
        return 1

    print("\nPASS: No regressions detected.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regression check against prior session dispatch logs."
    )
    parser.add_argument(
        "--log",
        required=True,
        help="Path to session JSONL log file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help=f"Exit 1 if regression rate exceeds {STRICT_REGRESSION_RATE:.0%}.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exit_code = run_harness(log_path=args.log, strict=args.strict)
    sys.exit(exit_code)
