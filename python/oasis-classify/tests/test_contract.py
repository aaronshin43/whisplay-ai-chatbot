"""
tests/test_contract.py — /dispatch JSON schema stability.

Ensures the TypeScript contract never silently breaks.
Tests that client-side threshold_path constants are importable.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Required schema fields and types
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "mode":               str,
    "response_text":      (str, type(None)),
    "system_prompt":      (str, type(None)),
    "category":           (str, type(None)),
    "top3":               list,
    "score":              (float, int, type(None)),
    "threshold_path":     str,
    "latency_ms":         (float, int),
    "hint_changed_result": bool,
}

VALID_MODES = {"direct_response", "llm_prompt", "triage_prompt", "ood_response"}

VALID_SERVER_THRESHOLD_PATHS = {
    "tier0_short",
    "tier0_sentence",
    "classifier_hit",
    "triage",
    "ood_floor",
    "ood_cluster",
}

VALID_CLIENT_THRESHOLD_PATHS = {
    "network_error",
    "service_error",
    "invalid_schema",
}

ALL_VALID_THRESHOLD_PATHS = VALID_SERVER_THRESHOLD_PATHS | VALID_CLIENT_THRESHOLD_PATHS


# ---------------------------------------------------------------------------
# Helper: convert DispatchResult to JSON dict (same as Flask would)
# ---------------------------------------------------------------------------

def _result_to_dict(result) -> dict:
    import dataclasses
    return dataclasses.asdict(result)


# ---------------------------------------------------------------------------
# Schema validation helper
# ---------------------------------------------------------------------------

def _assert_schema(d: dict):
    """Assert a dict matches the /dispatch response schema."""
    for field, expected_type in REQUIRED_FIELDS.items():
        assert field in d, f"Required field '{field}' missing from response"
        if isinstance(expected_type, tuple):
            assert isinstance(d[field], expected_type), (
                f"Field '{field}' has wrong type: {type(d[field])} (expected {expected_type})"
            )
        else:
            assert isinstance(d[field], expected_type), (
                f"Field '{field}' has wrong type: {type(d[field])} (expected {expected_type})"
            )

    assert d["mode"] in VALID_MODES, f"Invalid mode: {d['mode']}"

    assert isinstance(d["top3"], list), "top3 must be a list"
    for entry in d["top3"]:
        assert isinstance(entry, dict), f"top3 entry must be a dict, got {type(entry)}"
        assert "category" in entry, "top3 entry missing 'category'"
        assert "score" in entry, "top3 entry missing 'score'"
        assert isinstance(entry["category"], str), "top3 entry 'category' must be str"
        assert isinstance(entry["score"], (float, int)), "top3 entry 'score' must be numeric"


# ---------------------------------------------------------------------------
# Test client-side threshold_path constants are importable
# ---------------------------------------------------------------------------

class TestClientConstants:
    def test_network_error_constant(self):
        """THRESHOLD_PATH_NETWORK_ERROR is importable from config."""
        from config import THRESHOLD_PATH_NETWORK_ERROR
        assert THRESHOLD_PATH_NETWORK_ERROR == "network_error"

    def test_service_error_constant(self):
        """THRESHOLD_PATH_SERVICE_ERROR is importable from config."""
        from config import THRESHOLD_PATH_SERVICE_ERROR
        assert THRESHOLD_PATH_SERVICE_ERROR == "service_error"

    def test_invalid_schema_constant(self):
        """THRESHOLD_PATH_INVALID_SCHEMA is importable from config."""
        from config import THRESHOLD_PATH_INVALID_SCHEMA
        assert THRESHOLD_PATH_INVALID_SCHEMA == "invalid_schema"

    def test_all_client_constants_are_distinct(self):
        """All client-side threshold_path constants are distinct strings."""
        from config import (
            THRESHOLD_PATH_NETWORK_ERROR,
            THRESHOLD_PATH_SERVICE_ERROR,
            THRESHOLD_PATH_INVALID_SCHEMA,
        )
        values = [
            THRESHOLD_PATH_NETWORK_ERROR,
            THRESHOLD_PATH_SERVICE_ERROR,
            THRESHOLD_PATH_INVALID_SCHEMA,
        ]
        assert len(values) == len(set(values)), "Client threshold_path constants must be distinct"

    def test_client_constants_not_in_server_paths(self):
        """Client-synthesized paths must not collide with server-side paths."""
        from config import (
            THRESHOLD_PATH_NETWORK_ERROR,
            THRESHOLD_PATH_SERVICE_ERROR,
            THRESHOLD_PATH_INVALID_SCHEMA,
        )
        client_paths = {
            THRESHOLD_PATH_NETWORK_ERROR,
            THRESHOLD_PATH_SERVICE_ERROR,
            THRESHOLD_PATH_INVALID_SCHEMA,
        }
        collision = client_paths & VALID_SERVER_THRESHOLD_PATHS
        assert not collision, (
            f"Client threshold_path constants collide with server paths: {collision}"
        )


# ---------------------------------------------------------------------------
# Schema validation via Tier 0 (no model required)
# ---------------------------------------------------------------------------

class TestSchemaViaDispatch:
    def test_direct_response_schema(self):
        """direct_response mode matches full schema."""
        from service import dispatch
        result = dispatch("help")
        d = _result_to_dict(result)
        _assert_schema(d)
        assert d["mode"] == "direct_response"
        assert d["response_text"] is not None
        assert d["system_prompt"] is None
        assert d["category"] is None
        assert d["score"] is None

    def test_llm_prompt_tier0_schema(self):
        """llm_prompt mode (Tier 0 hit) matches full schema."""
        from service import dispatch
        result = dispatch("cpr")
        d = _result_to_dict(result)
        _assert_schema(d)
        assert d["mode"] == "llm_prompt"
        assert d["system_prompt"] is not None
        assert d["response_text"] is None
        assert d["score"] is None   # Tier 0 hit has no score

    def test_threshold_path_is_valid_server_path(self):
        """threshold_path from server dispatch is always a valid server path."""
        from service import dispatch
        for query in ["help", "cpr", "bleeding", "seizure"]:
            result = dispatch(query)
            assert result.threshold_path in VALID_SERVER_THRESHOLD_PATHS, (
                f"Query '{query}' returned unexpected threshold_path: {result.threshold_path}"
            )

    def test_top3_is_list_of_objects_not_tuples(self):
        """top3 must be a list of objects (dicts), never tuples or nested lists."""
        from service import dispatch
        result = dispatch("cpr")
        d = _result_to_dict(result)
        for entry in d["top3"]:
            assert isinstance(entry, dict), (
                f"top3 entry is {type(entry)}, expected dict"
            )
            assert not isinstance(entry, (list, tuple)), (
                "top3 entries must not be tuples or lists"
            )

    def test_hint_changed_result_is_bool(self):
        """hint_changed_result is always bool."""
        from service import dispatch
        result = dispatch("cpr")
        d = _result_to_dict(result)
        assert isinstance(d["hint_changed_result"], bool)

    def test_latency_ms_is_nonnegative(self):
        """latency_ms is always non-negative."""
        from service import dispatch
        result = dispatch("cpr")
        d = _result_to_dict(result)
        assert d["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# triage_prompt contract rules (no model — direct DispatchResult construction)
# ---------------------------------------------------------------------------

class TestTriagePromptContract:
    def test_triage_prompt_must_have_category(self):
        """Any triage_prompt result must have category set (rule #8)."""
        from classifier import DispatchResult
        # Simulate a triage result
        result = DispatchResult(
            mode="triage_prompt",
            response_text=None,
            system_prompt="ask clarifying question",
            category="cpr",           # MUST be set
            top3=[{"category": "cpr", "score": 0.45}],
            score=0.45,
            threshold_path="triage",
            latency_ms=50.0,
            hint_changed_result=False,
        )
        assert result.category is not None, (
            "triage_prompt mode MUST always have category set"
        )

    def test_ood_response_category_is_none(self):
        """ood_response and direct_response should have category=None."""
        from classifier import DispatchResult
        result = DispatchResult(
            mode="ood_response",
            response_text="I can only help with medical emergencies.",
            system_prompt=None,
            category=None,
            top3=[],
            score=0.1,
            threshold_path="ood_floor",
            latency_ms=10.0,
            hint_changed_result=False,
        )
        assert result.category is None


# ---------------------------------------------------------------------------
# All fields present in JSON serialization
# ---------------------------------------------------------------------------

class TestJSONSerialization:
    def test_dispatch_result_serializes_all_fields(self):
        """dataclasses.asdict includes all required schema fields."""
        import dataclasses
        from classifier import DispatchResult

        result = DispatchResult(
            mode="llm_prompt",
            response_text=None,
            system_prompt="test",
            category="cpr",
            top3=[{"category": "cpr", "score": 0.8}],
            score=0.8,
            threshold_path="classifier_hit",
            latency_ms=42.0,
            hint_changed_result=False,
        )
        d = dataclasses.asdict(result)
        for field in REQUIRED_FIELDS:
            assert field in d, f"Field '{field}' missing from serialized dict"

    def test_json_dumps_succeeds(self):
        """DispatchResult can be serialized to JSON without errors."""
        import json
        import dataclasses
        from classifier import DispatchResult

        result = DispatchResult(
            mode="ood_response",
            response_text="test response",
            system_prompt=None,
            category=None,
            top3=[],
            score=0.1,
            threshold_path="ood_floor",
            latency_ms=5.0,
            hint_changed_result=False,
        )
        d = dataclasses.asdict(result)
        json_str = json.dumps(d)
        assert len(json_str) > 0
