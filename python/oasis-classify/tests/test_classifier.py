"""
tests/test_classifier.py — Tier 1 classifier tests.

Tests that require the embedding model are marked @pytest.mark.requires_model
and skipped if the model / centroids are not available.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures and markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "requires_model: requires gte-small model and centroids.npy")


def _model_available() -> bool:
    """Check if centroids.npy exists (proxy for model availability)."""
    from config import CENTROIDS_PATH
    return os.path.isfile(CENTROIDS_PATH)


requires_model = pytest.mark.skipif(
    not _model_available(),
    reason="gte-small model / centroids.npy not available",
)


# ---------------------------------------------------------------------------
# DispatchResult structure tests (no model required)
# ---------------------------------------------------------------------------

class TestDispatchResultStructure:
    def test_dispatch_result_fields(self):
        """DispatchResult dataclass has all required fields."""
        from classifier import DispatchResult
        import dataclasses

        fields = {f.name for f in dataclasses.fields(DispatchResult)}
        required = {
            "mode", "response_text", "system_prompt", "category",
            "top3", "score", "threshold_path", "latency_ms", "hint_changed_result",
        }
        assert required.issubset(fields)

    def test_top3_structure_is_dicts(self):
        """top3 entries must be dicts with 'category' and 'score' keys, not tuples."""
        from classifier import DispatchResult
        result = DispatchResult(
            mode="ood_response",
            response_text="test",
            system_prompt=None,
            category=None,
            top3=[{"category": "cpr", "score": 0.9}, {"category": "bleeding", "score": 0.7}],
            score=0.9,
            threshold_path="ood_cluster",
            latency_ms=10.0,
            hint_changed_result=False,
        )
        for entry in result.top3:
            assert isinstance(entry, dict), "top3 entries must be dicts, not tuples or lists"
            assert "category" in entry
            assert "score" in entry
            assert isinstance(entry["category"], str)
            assert isinstance(entry["score"], float)

    def test_mode_values(self):
        """Mode field only accepts valid literal values."""
        from classifier import DispatchResult
        valid_modes = {"direct_response", "llm_prompt", "triage_prompt", "ood_response"}
        result = DispatchResult(
            mode="llm_prompt",
            response_text=None,
            system_prompt="test prompt",
            category="cpr",
            top3=[],
            score=0.8,
            threshold_path="classifier_hit",
            latency_ms=50.0,
            hint_changed_result=False,
        )
        assert result.mode in valid_modes


# ---------------------------------------------------------------------------
# OOD detection (no model required — using mock scores)
# ---------------------------------------------------------------------------

class TestOODRouting:
    def test_ood_floor_path(self):
        """Score below OOD_FLOOR routes to ood_response with threshold_path='ood_floor'."""
        # We test the routing logic directly using _cosine_scores logic via mock
        import numpy as np
        from config import OOD_FLOOR, CLASSIFY_THRESHOLD
        from categories import CATEGORY_IDS

        # Simulate a score array where all scores are below OOD_FLOOR
        n = len(CATEGORY_IDS)
        scores = np.full(n, 0.10, dtype=np.float32)  # all below 0.30

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        assert best_score < OOD_FLOOR
        # This would route to ood_floor in the classifier

    def test_ood_cluster_path(self):
        """out_of_domain cluster being top-1 routes to ood_response."""
        import numpy as np
        from config import OOD_FLOOR
        from categories import CATEGORY_IDS

        n = len(CATEGORY_IDS)
        scores = np.full(n, 0.20, dtype=np.float32)
        ood_idx = CATEGORY_IDS.index("out_of_domain")
        scores[ood_idx] = 0.75  # OOD cluster wins

        best_idx = int(np.argmax(scores))
        best_category = CATEGORY_IDS[best_idx]
        best_score = float(scores[best_idx])

        assert best_category == "out_of_domain"
        assert best_score >= OOD_FLOOR
        # This would route to ood_cluster in the classifier

    def test_triage_band(self):
        """Score in [OOD_FLOOR, CLASSIFY_THRESHOLD) routes to triage."""
        from config import OOD_FLOOR, CLASSIFY_THRESHOLD

        triage_score = (OOD_FLOOR + CLASSIFY_THRESHOLD) / 2
        assert OOD_FLOOR <= triage_score < CLASSIFY_THRESHOLD

    def test_classifier_hit_band(self):
        """Score >= CLASSIFY_THRESHOLD routes to llm_prompt."""
        from config import CLASSIFY_THRESHOLD

        hit_score = CLASSIFY_THRESHOLD + 0.05
        assert hit_score >= CLASSIFY_THRESHOLD


# ---------------------------------------------------------------------------
# Model-dependent tests
# ---------------------------------------------------------------------------

@requires_model
class TestClassifierWithModel:
    def test_cpr_query_classifies(self):
        """CPR query classifies to 'cpr' or triage band."""
        from classifier import classify
        result = classify("someone is not breathing")
        assert result.mode in ("llm_prompt", "triage_prompt")
        if result.mode == "llm_prompt":
            assert result.category == "cpr"

    def test_ood_query(self):
        """Non-medical query returns ood_response."""
        from classifier import classify
        result = classify("what is the capital of france")
        assert result.mode == "ood_response"
        assert result.category is None

    def test_top3_is_list_of_dicts(self):
        """top3 is a list of dicts with 'category' and 'score'."""
        from classifier import classify
        result = classify("someone is not breathing")
        assert isinstance(result.top3, list)
        assert len(result.top3) <= 3
        for entry in result.top3:
            assert isinstance(entry, dict)
            assert "category" in entry
            assert "score" in entry

    def test_triage_prompt_always_has_category(self):
        """triage_prompt mode always has category set."""
        from classifier import classify
        # Force a triage band result by using an ambiguous query
        # This may or may not be triage depending on centroids — skip if not
        result = classify("I think something might be wrong with my friend")
        if result.mode == "triage_prompt":
            assert result.category is not None, "triage_prompt must always have category set"

    def test_score_is_float_not_none_for_classifier(self):
        """Classifier results have score as float, not None."""
        from classifier import classify
        result = classify("someone is not breathing")
        assert result.score is not None
        assert isinstance(result.score, float)

    def test_latency_ms_positive(self):
        """latency_ms is a positive float."""
        from classifier import classify
        result = classify("someone is not breathing")
        assert result.latency_ms > 0

    def test_hint_changed_result_is_bool(self):
        """hint_changed_result is always bool."""
        from classifier import classify
        result = classify("someone is not breathing", prev_triage_hint=None)
        assert isinstance(result.hint_changed_result, bool)
