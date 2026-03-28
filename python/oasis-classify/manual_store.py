"""
manual_store.py — Loads pre-generated manual text files at startup.

Serves manual content by category ID. All medical content comes from
data/manuals/*.txt — never generated at runtime.
"""

from __future__ import annotations

import os

from config import MANUALS_DIR


# ---------------------------------------------------------------------------
# Load all manuals at import time
# ---------------------------------------------------------------------------

def _load_manuals() -> dict[str, str]:
    """Load all .txt files in MANUALS_DIR into a dict keyed by category ID."""
    manuals: dict[str, str] = {}
    if not os.path.isdir(MANUALS_DIR):
        return manuals
    for filename in os.listdir(MANUALS_DIR):
        if filename.endswith(".txt"):
            category_id = filename[:-4]  # strip .txt
            filepath = os.path.join(MANUALS_DIR, filename)
            with open(filepath, encoding="utf-8") as fh:
                manuals[category_id] = fh.read().strip()
    return manuals


MANUALS: dict[str, str] = _load_manuals()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_manual(category_id: str) -> str | None:
    """Return the manual text for a given category ID, or None if not found."""
    return MANUALS.get(category_id)


def list_categories() -> list[str]:
    """Return sorted list of all loaded category IDs."""
    return sorted(MANUALS.keys())
