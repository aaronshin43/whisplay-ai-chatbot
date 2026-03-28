"""
training/generate_data.py — Generate synthetic training data for oasis-classify.

Default mode: offline text transformations only (no LLM required).
Optional --llm flag: augments with Ollama paraphrases if available.

Output: training/data/synthetic_queries.csv with columns: query,category

Usage:
    python training/generate_data.py
    python training/generate_data.py --output training/data/synthetic_queries.csv
    python training/generate_data.py --llm --llm-model gemma3:1b
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — allow running as standalone script from any cwd
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CLASSIFY_DIR = os.path.dirname(_SCRIPT_DIR)

PROTOTYPES_PATH = os.path.join(_CLASSIFY_DIR, "data", "prototypes.json")
DEFAULT_OUTPUT = os.path.join(_SCRIPT_DIR, "data", "synthetic_queries.csv")


# ---------------------------------------------------------------------------
# Offline text transformation helpers
# ---------------------------------------------------------------------------

# Subject pronoun swap variants
_SUBJECT_SWAPS = [
    ("he ", "she "),
    ("he ", "they "),
    ("he ", "my friend "),
    ("he ", "the person "),
    ("he ", "someone "),
    ("she ", "he "),
    ("she ", "they "),
    ("she ", "my friend "),
    ("my friend ", "he "),
    ("the person ", "someone "),
    ("someone ", "my friend "),
    ("i ", "we "),
]

# Urgency prefixes
_URGENCY_PREFIXES = [
    "quickly ",
    "please ",
    "help ",
    "urgent ",
    "emergency ",
    "please help ",
]

# Filler suffixes
_FILLER_SUFFIXES = [
    " what do i do",
    " please help",
    " help me",
    " what should i do",
    " i need help",
]

# Question vs statement variants
_QUESTION_STARTERS = [
    "what do i do if ",
    "how do i help someone who is ",
    "what should i do when ",
    "help someone is ",
    "how to help with ",
]

# Tense variants (simple past-tense suffixes that apply to some prototypes)
_PAST_TENSE_SUBSTITUTIONS = [
    (" is ", " was "),
    (" are ", " were "),
    (" has ", " had "),
    (" can't ", " couldn't "),
    (" cannot ", " could not "),
]


def _apply_subject_swap(text: str) -> list[str]:
    """Generate variants by swapping subject pronouns."""
    variants = []
    for wrong, right in _SUBJECT_SWAPS:
        if wrong in text:
            variants.append(text.replace(wrong, right, 1))
    return variants


def _apply_urgency_prefix(text: str) -> list[str]:
    """Add urgency prefixes to the query."""
    return [prefix + text for prefix in _URGENCY_PREFIXES]


def _apply_filler_suffix(text: str) -> list[str]:
    """Add filler suffixes to the query."""
    return [text + suffix for suffix in _FILLER_SUFFIXES]


def _apply_question_variant(text: str) -> list[str]:
    """Convert statement to question form."""
    variants = []
    for starter in _QUESTION_STARTERS:
        variants.append(starter + text)
    return variants


def _apply_past_tense(text: str) -> list[str]:
    """Generate past tense variants."""
    variants = []
    for present, past in _PAST_TENSE_SUBSTITUTIONS:
        if present in text:
            variants.append(text.replace(present, past, 1))
    return variants


def generate_variants(prototype: str) -> list[str]:
    """Generate offline text transformation variants for a prototype query.

    Targets 8-12 variants per prototype.
    """
    p = prototype.lower().strip()
    variants: list[str] = [p]  # original as-is

    seen: set[str] = {p}

    def _add(v: str) -> None:
        v = v.strip()
        if v and v not in seen and len(v) >= 3:
            seen.add(v)
            variants.append(v)

    # Subject pronoun swaps
    for v in _apply_subject_swap(p):
        _add(v)

    # Urgency prefixes (pick first 2 to avoid bloat)
    for v in _apply_urgency_prefix(p)[:2]:
        _add(v)

    # Filler suffixes (pick first 2)
    for v in _apply_filler_suffix(p)[:2]:
        _add(v)

    # Question variants (pick first 2)
    for v in _apply_question_variant(p)[:2]:
        _add(v)

    # Past tense
    for v in _apply_past_tense(p):
        _add(v)

    # Combination: urgency prefix + filler suffix
    combo = _URGENCY_PREFIXES[0] + p + _FILLER_SUFFIXES[0]
    _add(combo)

    # Combination: question starter + past tense
    for past_v in _apply_past_tense(p)[:1]:
        _add(_QUESTION_STARTERS[0] + past_v)

    return variants


# ---------------------------------------------------------------------------
# Optional LLM augmentation via Ollama
# ---------------------------------------------------------------------------

def _augment_with_llm(prototype: str, category: str, model: str = "gemma3:1b") -> list[str]:
    """Call Ollama to generate paraphrases. Returns empty list on failure."""
    try:
        import urllib.request
        import urllib.error

        prompt = (
            f"Generate 5 different natural-language phrasings of this first-aid emergency query. "
            f"Each phrasing should convey the same situation but use different words. "
            f"Output one phrasing per line, no numbering, no explanations.\n\n"
            f"Original: {prototype}\n\nPhrasings:"
        )

        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.8, "num_predict": 200},
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            text = data.get("response", "")

        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        # Filter out lines that are too long or look like explanations
        lines = [l for l in lines if 3 <= len(l) <= 200 and not l.startswith(("Note:", "Here", "These"))]
        return lines[:5]

    except Exception as exc:
        print(f"  [LLM] Warning: could not augment '{prototype[:40]}': {exc}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate_dataset(
    prototypes_path: str,
    output_path: str,
    use_llm: bool = False,
    llm_model: str = "gemma3:1b",
) -> int:
    """Generate synthetic dataset. Returns number of rows written."""

    with open(prototypes_path, encoding="utf-8") as fh:
        prototypes: dict[str, list[str]] = json.load(fh)

    rows: list[tuple[str, str]] = []

    total_categories = len(prototypes)
    for idx, (category, proto_list) in enumerate(prototypes.items(), 1):
        print(f"  [{idx}/{total_categories}] {category}: {len(proto_list)} prototypes", file=sys.stderr)
        category_rows: set[str] = set()

        for proto in proto_list:
            proto = proto.strip()
            if not proto:
                continue

            # Offline variants
            variants = generate_variants(proto)
            for v in variants:
                category_rows.add(v.lower().strip())

            # Optional LLM augmentation
            if use_llm:
                llm_variants = _augment_with_llm(proto, category, llm_model)
                for v in llm_variants:
                    category_rows.add(v.lower().strip())

        for query in sorted(category_rows):
            rows.append((query, category))

        print(f"    -> {len(category_rows)} unique queries", file=sys.stderr)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query", "category"])
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_path}", file=sys.stderr)
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for oasis-classify."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--prototypes",
        default=PROTOTYPES_PATH,
        help=f"Path to prototypes.json (default: {PROTOTYPES_PATH})",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help="Augment with Ollama paraphrases (requires Ollama running locally)",
    )
    parser.add_argument(
        "--llm-model",
        default="gemma3:1b",
        help="Ollama model to use for LLM augmentation (default: gemma3:1b)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not os.path.exists(args.prototypes):
        print(f"ERROR: prototypes file not found: {args.prototypes}", file=sys.stderr)
        sys.exit(1)

    print(f"Generating synthetic training data from {args.prototypes}", file=sys.stderr)
    if args.llm:
        print(f"LLM augmentation enabled (model: {args.llm_model})", file=sys.stderr)
    else:
        print("Offline mode (no LLM). Use --llm to enable Ollama augmentation.", file=sys.stderr)

    count = generate_dataset(
        prototypes_path=args.prototypes,
        output_path=args.output,
        use_llm=args.llm,
        llm_model=args.llm_model,
    )

    print(f"Done. {count} total samples generated.")
