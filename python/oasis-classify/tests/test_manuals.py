"""
tests/test_manuals.py — Manual format validation.

For each .txt file in data/manuals/:
  - Verify "STEPS:" is present
  - Verify "NEVER DO:" is present
  - Verify token count is between 80 and 140 (tiktoken cl100k_base)
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tiktoken
from config import MANUALS_DIR


# ---------------------------------------------------------------------------
# Discover all manual files
# ---------------------------------------------------------------------------

def _get_manual_files() -> list[tuple[str, str]]:
    """Return list of (category_id, filepath) for all .txt files in MANUALS_DIR."""
    if not os.path.isdir(MANUALS_DIR):
        return []
    results = []
    for filename in sorted(os.listdir(MANUALS_DIR)):
        if filename.endswith(".txt"):
            category_id = filename[:-4]
            filepath = os.path.join(MANUALS_DIR, filename)
            results.append((category_id, filepath))
    return results


_MANUAL_FILES = _get_manual_files()
_ENCODING = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Parametrize over all manuals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category_id,filepath", _MANUAL_FILES)
def test_manual_has_steps_section(category_id: str, filepath: str):
    """Every manual must contain a 'STEPS:' section."""
    with open(filepath, encoding="utf-8") as fh:
        content = fh.read()
    assert "STEPS:" in content, (
        f"Manual '{category_id}' ({filepath}) is missing 'STEPS:' section"
    )


@pytest.mark.parametrize("category_id,filepath", _MANUAL_FILES)
def test_manual_has_never_do_section(category_id: str, filepath: str):
    """Every manual must contain a 'NEVER DO:' section."""
    with open(filepath, encoding="utf-8") as fh:
        content = fh.read()
    assert "NEVER DO:" in content, (
        f"Manual '{category_id}' ({filepath}) is missing 'NEVER DO:' section"
    )


@pytest.mark.parametrize("category_id,filepath", _MANUAL_FILES)
def test_manual_token_count_in_range(category_id: str, filepath: str):
    """Manual token count must be between 80 and 140 (tiktoken cl100k_base)."""
    with open(filepath, encoding="utf-8") as fh:
        content = fh.read()
    token_count = len(_ENCODING.encode(content))
    assert 80 <= token_count <= 140, (
        f"Manual '{category_id}' has {token_count} tokens — must be 80-140. "
        f"File: {filepath}"
    )


@pytest.mark.parametrize("category_id,filepath", _MANUAL_FILES)
def test_manual_no_markdown(category_id: str, filepath: str):
    """Manuals should not contain markdown headers, bold markers, or bullet asterisks."""
    with open(filepath, encoding="utf-8") as fh:
        content = fh.read()
    # Check for common markdown patterns (# headings, ** bold, ``` code)
    assert "```" not in content, f"Manual '{category_id}' contains markdown code block"
    # Allow * in "NEVER DO" lists as hyphens are used there, just check for ** bold
    assert "**" not in content, f"Manual '{category_id}' contains markdown bold (**)"


def test_all_expected_manuals_present():
    """All 32 expected category manuals must be present."""
    from categories import CATEGORY_IDS
    loaded = {cat_id for cat_id, _ in _MANUAL_FILES}
    # out_of_domain does not need a manual
    expected = {cat_id for cat_id in CATEGORY_IDS if cat_id != "out_of_domain"}
    missing = expected - loaded
    assert not missing, f"Missing manual files for categories: {sorted(missing)}"


def test_no_extra_manual_files():
    """No manual files should exist for undefined category IDs."""
    from categories import CATEGORY_IDS
    valid_ids = set(CATEGORY_IDS) | {"out_of_domain"}
    loaded = {cat_id for cat_id, _ in _MANUAL_FILES}
    extra = loaded - valid_ids
    assert not extra, f"Extra manual files with unknown category IDs: {sorted(extra)}"
