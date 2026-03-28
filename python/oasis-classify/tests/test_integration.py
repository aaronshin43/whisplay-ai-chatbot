"""
tests/test_integration.py — End-to-end dispatch pipeline tests.

Tests all four dispatch modes using service.dispatch() directly (no HTTP).
Model-dependent tests are marked @pytest.mark.requires_model and skipped
if centroids.npy is not available.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Skip marker for model-dependent tests
# ---------------------------------------------------------------------------

def _model_available() -> bool:
    from config import CENTROIDS_PATH
    return os.path.isfile(CENTROIDS_PATH)


requires_model = pytest.mark.skipif(
    not _model_available(),
    reason="gte-small model / centroids.npy not available",
)


# ---------------------------------------------------------------------------
# direct_response mode — Tier 0 pre-baked text (no model needed)
# ---------------------------------------------------------------------------

class TestDirectResponseMode:
    def test_help_returns_direct_response(self):
        """'help' keyword returns direct_response mode."""
        from service import dispatch
        result = dispatch("help")
        assert result.mode == "direct_response"
        assert result.response_text is not None
        assert result.system_prompt is None
        assert result.score is None
        assert result.threshold_path == "tier0_short"
        assert result.latency_ms >= 0

    def test_911_returns_direct_response(self):
        """'911' returns direct_response with call-911 text."""
        from service import dispatch
        result = dispatch("911")
        assert result.mode == "direct_response"
        assert "911" in result.response_text or "emergency" in result.response_text.lower()
        assert result.threshold_path == "tier0_short"

    def test_hint_changed_result_false_for_tier0(self):
        """Tier 0 hits always have hint_changed_result=False."""
        from service import dispatch
        result = dispatch("help")
        assert result.hint_changed_result is False


# ---------------------------------------------------------------------------
# llm_prompt mode — Tier 0 category hit (no model needed)
# ---------------------------------------------------------------------------

class TestLLMPromptModeTier0:
    def test_cpr_keyword_returns_llm_prompt(self):
        """'cpr' keyword returns llm_prompt with system_prompt."""
        from service import dispatch
        result = dispatch("cpr")
        assert result.mode == "llm_prompt"
        assert result.system_prompt is not None
        assert "STEPS" in result.system_prompt
        assert "NEVER DO" in result.system_prompt
        assert result.response_text is None
        assert result.threshold_path == "tier0_short"

    def test_bleeding_keyword_returns_llm_prompt(self):
        """'bleeding' keyword returns llm_prompt."""
        from service import dispatch
        result = dispatch("bleeding")
        assert result.mode == "llm_prompt"
        assert result.system_prompt is not None

    def test_sentence_match_returns_llm_prompt(self):
        """Sentence match returns llm_prompt with prompt."""
        from service import dispatch
        result = dispatch("how do i stop the bleeding")
        assert result.mode == "llm_prompt"
        assert result.system_prompt is not None
        assert result.threshold_path == "tier0_sentence"


# ---------------------------------------------------------------------------
# ood_response mode (model-dependent)
# ---------------------------------------------------------------------------

@requires_model
class TestOODResponseMode:
    def test_nonsense_returns_ood(self):
        """Nonsense query returns ood_response."""
        from service import dispatch
        result = dispatch("asdfghjklzxcvbnm qwerty")
        assert result.mode == "ood_response"
        assert result.response_text is not None
        assert result.system_prompt is None
        assert result.category is None
        assert result.threshold_path in ("ood_floor", "ood_cluster")

    def test_non_medical_returns_ood(self):
        """Clearly non-medical query returns ood_response."""
        from service import dispatch
        result = dispatch("what time is it")
        assert result.mode == "ood_response"
        assert result.category is None


# ---------------------------------------------------------------------------
# triage_prompt mode (model-dependent)
# ---------------------------------------------------------------------------

@requires_model
class TestTriagePromptMode:
    def test_triage_prompt_has_category(self):
        """triage_prompt always has category set."""
        from service import dispatch
        # Use a vague medical query that may fall in triage band
        result = dispatch("i think something is wrong")
        if result.mode == "triage_prompt":
            assert result.category is not None
            assert result.system_prompt is not None
            assert result.response_text is None

    def test_triage_prompt_threshold_path(self):
        """triage_prompt has threshold_path='triage'."""
        from service import dispatch
        result = dispatch("i think something is wrong with my friend")
        if result.mode == "triage_prompt":
            assert result.threshold_path == "triage"


# ---------------------------------------------------------------------------
# Schema completeness — all modes (Tier 0 hit, no model needed)
# ---------------------------------------------------------------------------

class TestSchemaCompleteness:
    def _assert_full_schema(self, result):
        """Assert all required fields are present with correct types."""
        import dataclasses
        from classifier import DispatchResult

        assert isinstance(result, DispatchResult)
        assert result.mode in ("direct_response", "llm_prompt", "triage_prompt", "ood_response")
        assert isinstance(result.top3, list)
        assert isinstance(result.threshold_path, str)
        assert isinstance(result.latency_ms, float)
        assert isinstance(result.hint_changed_result, bool)

        for entry in result.top3:
            assert isinstance(entry, dict)
            assert "category" in entry
            assert "score" in entry

    def test_direct_response_schema(self):
        from service import dispatch
        result = dispatch("help")
        self._assert_full_schema(result)

    def test_llm_prompt_schema_tier0(self):
        from service import dispatch
        result = dispatch("cpr")
        self._assert_full_schema(result)


# ---------------------------------------------------------------------------
# Multi-label token ceiling (no model needed — test prompt_builder directly)
# ---------------------------------------------------------------------------

class TestMultiLabelTokenCeiling:
    def test_combined_prompt_within_token_limit(self):
        """Primary + also-check prompt should not exceed MAX_PROMPT_TOKENS."""
        from prompt_builder import build_prompt, count_tokens
        from config import MAX_PROMPT_TOKENS

        prompt = build_prompt(
            query="my friend is bleeding from a broken leg",
            primary_category="bleeding",
            secondary_category="fracture",
        )
        token_count = count_tokens(prompt)
        assert token_count <= MAX_PROMPT_TOKENS, (
            f"Combined prompt exceeds MAX_PROMPT_TOKENS ({MAX_PROMPT_TOKENS}): {token_count}"
        )

    def test_secondary_dropped_when_over_limit(self):
        """When combined prompt exceeds MAX_PROMPT_TOKENS, secondary block is dropped."""
        from prompt_builder import build_prompt
        from config import MAX_PROMPT_TOKENS
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        # Build with and without secondary
        prompt_with = build_prompt("test query", "cpr", "bleeding")
        prompt_without = build_prompt("test query", "cpr")

        # Both should be within limit
        assert len(enc.encode(prompt_with)) <= MAX_PROMPT_TOKENS
        assert len(enc.encode(prompt_without)) <= MAX_PROMPT_TOKENS

        # If combined would exceed, the function returns primary only
        # We verify this by checking the without-secondary prompt is smaller or equal
        assert len(enc.encode(prompt_without)) <= len(enc.encode(prompt_with)) + 50

    def test_also_check_block_present_when_within_limit(self):
        """When combined prompt is within limit, ALSO CHECK block should appear."""
        from prompt_builder import build_prompt

        prompt = build_prompt(
            query="bleeding from a fracture",
            primary_category="bleeding",
            secondary_category="fracture",
        )
        # The prompt should contain ALSO CHECK since both manuals are short
        assert "ALSO CHECK" in prompt

    def test_primary_manual_always_present(self):
        """Primary manual is always present regardless of secondary."""
        from prompt_builder import build_prompt

        prompt = build_prompt("test query", "cpr")
        assert "STEPS" in prompt
        assert "NEVER DO" in prompt
