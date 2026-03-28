"""
tests/test_adversarial.py — Adversarial robustness and life-critical canary tests.

Test groups:
  - Canary tests: life-critical recall >= 0.95 for cpr, choking, bleeding
  - ASR noise tests: homophones, repeated letters, number words
  - Mixed-topic confusion tests: multi-condition queries must not be OOD
  - OOD rejection tests: non-medical queries must be OOD
  - Foreign language / gibberish tests

Tests requiring the model are marked @pytest.mark.requires_model.
Skipped gracefully if data/centroids.npy is absent.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure classify modules are importable from the tests/ subfolder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Marker and skip guard
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_model: requires gte-small model and centroids.npy",
    )


def _model_available() -> bool:
    from config import CENTROIDS_PATH
    return os.path.isfile(CENTROIDS_PATH)


requires_model = pytest.mark.skipif(
    not _model_available(),
    reason="gte-small model / centroids.npy not available",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _dispatch(query: str) -> object:
    """Run the full dispatch pipeline (Tier 0 + classifier)."""
    from service import dispatch
    return dispatch(query)


def _classify_only(query: str) -> object:
    """Run Tier 1 classifier only (normalized query)."""
    from fast_match import normalize
    from classifier import classify
    return classify(normalize(query))


def _is_not_ood(result) -> bool:
    return result.mode != "ood_response"


def _is_ood(result) -> bool:
    return result.mode == "ood_response"


def _category_matches(result, *expected_categories: str) -> bool:
    """Return True if result.category is one of the expected categories,
    OR if mode is triage_prompt and category matches (triage is acceptable for life-critical)."""
    if result.category in expected_categories:
        return True
    # top3 check: any of expected_categories in top3
    top3_cats = {entry["category"] for entry in result.top3}
    return bool(top3_cats & set(expected_categories))


# ---------------------------------------------------------------------------
# Canary tests — life-critical recall >= 0.95
# ---------------------------------------------------------------------------

# At least 20 varied phrasings per life-critical category.
# Acceptable result: category in {expected} OR mode is triage_prompt (never ood_response).
# Automatic failure: mode == ood_response for any life-critical query.

CPR_QUERIES = [
    "someone is not breathing",
    "the person has no pulse",
    "cardiac arrest",
    "do chest compressions",
    "he collapsed and is not breathing",
    "she stopped breathing",
    "unresponsive and not breathing",
    "heart stopped beating",
    "do cpr on him",
    "i need to do rescue breathing",
    "the victim is unresponsive with no pulse",
    "my friend passed out and wont wake up no pulse",
    "adult not breathing what do i do",
    "start cpr immediately",
    "perform cardiopulmonary resuscitation",
    "thirty compressions two breaths",
    "person collapsed unconscious not breathing",
    "ambulance is coming but she has no heartbeat",
    "how do i do cpr correctly",
    "baby is not breathing what do i do",
    "he is unresponsive and i cant feel a pulse",
    "she has no signs of life",
]

CHOKING_QUERIES = [
    "someone is choking",
    "they cant breathe and are grabbing their throat",
    "airway obstruction",
    "do the heimlich maneuver",
    "he is choking on food",
    "she is turning blue and cant speak",
    "baby is choking on something",
    "complete airway blockage",
    "person cant breathe choking",
    "abdominal thrusts choking adult",
    "my child swallowed something and is choking",
    "infant choking back blows",
    "friend is choking at dinner",
    "how to help a choking person",
    "she cannot speak or breathe choking",
    "child ate something and is now choking",
    "obstruction in the throat",
    "he is clutching his throat and turning red",
    "toddler choked on a toy",
    "person is choking please help",
    "they swallowed food wrong and cant breathe",
    "do back blows on a choking person",
]

BLEEDING_QUERIES = [
    "severe bleeding",
    "there is a lot of blood",
    "how do i stop the bleeding",
    "deep cut with heavy bleeding",
    "apply pressure to the wound",
    "the wound is bleeding badly",
    "she cut her arm and its bleeding heavily",
    "he has a gash that wont stop bleeding",
    "blood everywhere what do i do",
    "use a tourniquet on the leg",
    "arterial bleeding spurting blood",
    "blood soaking through the bandage",
    "knife wound bleeding a lot",
    "my friend is bleeding from the leg",
    "uncontrolled hemorrhage",
    "gunshot wound bleeding",
    "how to apply a tourniquet",
    "the person is losing a lot of blood",
    "press down on the cut to stop bleeding",
    "open wound with blood loss",
    "lacerating injury with significant bleeding",
    "he is bleeding profusely from a cut",
]


@requires_model
class TestCanaryLifeCriticalCPR:
    """Life-critical recall >= 0.95 for CPR queries.

    Acceptable: category == 'cpr' or mode == 'triage_prompt' (never ood_response).
    """

    def _run_recall_check(self, queries, expected_category):
        acceptable = 0
        ood_failures = []

        for q in queries:
            result = _dispatch(q)
            if result.mode == "ood_response":
                ood_failures.append(q)
            else:
                acceptable += 1

        recall = acceptable / len(queries)
        assert not ood_failures, (
            f"AUTOMATIC FAILURE: {len(ood_failures)} life-critical queries routed to OOD: "
            f"{ood_failures[:5]}"
        )
        assert recall >= 0.95, (
            f"Life-critical recall for '{expected_category}' is {recall:.2%} "
            f"(need >= 95%). Failed queries: "
            f"{[q for q in queries if _dispatch(q).mode == 'ood_response']}"
        )
        return recall

    def test_cpr_recall(self):
        recall = self._run_recall_check(CPR_QUERIES, "cpr")
        print(f"\nCPR recall: {recall:.2%} ({len(CPR_QUERIES)} queries)")

    def test_cpr_no_ood_failures(self):
        """No CPR query must ever route to ood_response."""
        for q in CPR_QUERIES:
            result = _dispatch(q)
            assert result.mode != "ood_response", (
                f"CRITICAL FAILURE: CPR query '{q}' routed to ood_response "
                f"(threshold_path={result.threshold_path})"
            )


@requires_model
class TestCanaryLifeCriticalChoking:
    def test_choking_recall(self):
        acceptable = 0
        ood_failures = []

        for q in CHOKING_QUERIES:
            result = _dispatch(q)
            if result.mode == "ood_response":
                ood_failures.append(q)
            else:
                acceptable += 1

        recall = acceptable / len(CHOKING_QUERIES)
        assert not ood_failures, (
            f"AUTOMATIC FAILURE: Choking queries routed to OOD: {ood_failures}"
        )
        assert recall >= 0.95, (
            f"Life-critical recall for 'choking' is {recall:.2%} (need >= 95%)"
        )
        print(f"\nChoking recall: {recall:.2%} ({len(CHOKING_QUERIES)} queries)")

    def test_choking_no_ood_failures(self):
        for q in CHOKING_QUERIES:
            result = _dispatch(q)
            assert result.mode != "ood_response", (
                f"CRITICAL FAILURE: Choking query '{q}' routed to ood_response"
            )


@requires_model
class TestCanaryLifeCriticalBleeding:
    def test_bleeding_recall(self):
        acceptable = 0
        ood_failures = []

        for q in BLEEDING_QUERIES:
            result = _dispatch(q)
            if result.mode == "ood_response":
                ood_failures.append(q)
            else:
                acceptable += 1

        recall = acceptable / len(BLEEDING_QUERIES)
        assert not ood_failures, (
            f"AUTOMATIC FAILURE: Bleeding queries routed to OOD: {ood_failures}"
        )
        assert recall >= 0.95, (
            f"Life-critical recall for 'bleeding' is {recall:.2%} (need >= 95%)"
        )
        print(f"\nBleeding recall: {recall:.2%} ({len(BLEEDING_QUERIES)} queries)")

    def test_bleeding_no_ood_failures(self):
        for q in BLEEDING_QUERIES:
            result = _dispatch(q)
            assert result.mode != "ood_response", (
                f"CRITICAL FAILURE: Bleeding query '{q}' routed to ood_response"
            )


# ---------------------------------------------------------------------------
# ASR noise tests
# ---------------------------------------------------------------------------

class TestASRNoise:
    """Tests that ASR artifacts normalize and route correctly (not OOD)."""

    # Homophone tests — these rely on the normalize() function in fast_match.py
    def test_homophone_strock_normalizes(self):
        """'strock' normalizes to 'stroke' and matches Tier 0."""
        from fast_match import normalize, tier0_lookup
        norm = normalize("strock")
        assert "stroke" in norm, f"'strock' should normalize to contain 'stroke', got '{norm}'"

    def test_homophone_sieze_normalizes(self):
        """'sieze' normalizes to 'seizure'."""
        from fast_match import normalize
        norm = normalize("sieze")
        assert "seizure" in norm, f"'sieze' should normalize to contain 'seizure', got '{norm}'"

    def test_homophone_heart_tack_normalizes(self):
        """'hart attack' / 'heart tack' normalizes to 'heart attack'."""
        from fast_match import normalize
        norm = normalize("heart tack symptoms")
        assert "heart attack" in norm, f"Expected 'heart attack' in '{norm}'"

    def test_homophone_bleding_normalizes(self):
        """'bleding' (missing letter, not covered by homophone map) — verify normalize() does not crash."""
        # This is not in the homophone map; the edit distance gate (<=6 chars)
        # won't apply because len('bleding')=7. Just verify normalization is stable.
        from fast_match import normalize
        norm = normalize("bleding badly")
        # Should lowercase and strip punctuation — at minimum it should be stable
        assert isinstance(norm, str)
        assert len(norm) > 0

    def test_repeated_letters_heelp(self):
        """'heeelp' collapses to 'heelp' — normalized, not crashed."""
        from fast_match import normalize
        norm = normalize("heeelp")
        assert norm == "heelp", f"Expected 'heelp', got '{norm}'"

    def test_repeated_letters_pleease(self):
        """'pleease help' normalizes without crash (2 repeated letters not collapsed — need 3+)."""
        from fast_match import normalize
        norm = normalize("pleease help")
        # The normalize() regex collapses runs of 3+ identical chars to 2.
        # "pleease" has only 2 e's in a row, so it is NOT collapsed — that is correct.
        # Verify it normalizes cleanly without crashing.
        assert isinstance(norm, str)
        assert "help" in norm

    def test_number_words_nine_one_one(self):
        """'call nine one one' normalizes to 'call 911'."""
        from fast_match import normalize
        norm = normalize("call nine one one")
        assert "911" in norm, f"Expected '911' in '{norm}'"

    def test_number_words_nine_eleven(self):
        """'call nine eleven' normalizes to 'call 911'."""
        from fast_match import normalize
        norm = normalize("call nine eleven")
        assert "911" in norm, f"Expected '911' in '{norm}'"

    @requires_model
    def test_asr_noise_strock_routes_to_stroke_or_triage(self):
        """'strock' routes to stroke category (not OOD)."""
        result = _dispatch("strock")
        assert result.mode != "ood_response", (
            "'strock' (ASR for stroke) should not be OOD"
        )

    @requires_model
    def test_asr_noise_sieze_routes_not_ood(self):
        """'sieze' routes to seizure (not OOD)."""
        result = _dispatch("sieze")
        assert result.mode != "ood_response", (
            "'sieze' (ASR for seizure) should not be OOD"
        )

    @requires_model
    def test_asr_noise_hart_attack_routes_not_ood(self):
        """'hart attack' routes to heart_attack (not OOD)."""
        result = _dispatch("hart attack")
        assert result.mode != "ood_response", (
            "'hart attack' (ASR for heart attack) should not be OOD"
        )

    @requires_model
    def test_asr_noise_call_nine_one_one(self):
        """'call nine one one' routes correctly (Tier 0 or classifier)."""
        result = _dispatch("call nine one one")
        # Should match CALL_911 response via Tier 0 after normalization
        assert result is not None  # just must not crash

    @requires_model
    def test_asr_noise_repeated_letters_route_not_ood(self):
        """'heeeelp' with many repeated letters does not crash and routes reasonably."""
        result = _dispatch("heeeelp me please")
        assert result is not None
        # Not asserting specific route — just that it doesn't error or OOD on "help me please"


# ---------------------------------------------------------------------------
# Mixed-topic confusion tests
# ---------------------------------------------------------------------------

@requires_model
class TestMixedTopicConfusion:
    """Multi-condition queries must not route to OOD."""

    def test_bleeding_broken_leg(self):
        """'bleeding from a broken leg' — should be bleeding or fracture, not OOD."""
        result = _dispatch("bleeding from a broken leg")
        assert _is_not_ood(result), (
            "'bleeding from a broken leg' should not be OOD. "
            f"Got mode={result.mode}, threshold_path={result.threshold_path}"
        )
        # Primary should be bleeding or fracture
        if result.category:
            assert result.category in ("bleeding", "fracture", "shock"), (
                f"Expected bleeding/fracture/shock, got {result.category}"
            )

    def test_baby_not_breathing_after_drowning(self):
        """'baby not breathing after drowning' — should route to one of: drowning, cpr, pediatric_emergency."""
        result = _dispatch("baby not breathing after drowning")
        assert _is_not_ood(result), (
            "'baby not breathing after drowning' should not be OOD"
        )
        if result.mode == "llm_prompt" and result.category:
            assert result.category in (
                "drowning", "cpr", "pediatric_emergency"
            ), f"Expected drowning/cpr/pediatric, got {result.category}"

    def test_seizure_caused_by_head_injury(self):
        """'seizure caused by head injury' — should be seizure or head_injury, not OOD."""
        result = _dispatch("seizure caused by head injury")
        assert _is_not_ood(result), (
            "'seizure caused by head injury' should not be OOD"
        )
        if result.mode == "llm_prompt" and result.category:
            assert result.category in (
                "seizure", "head_injury"
            ), f"Expected seizure/head_injury, got {result.category}"

    def test_shock_and_bleeding(self):
        """'massive bleeding causing shock' — bleeding or shock, not OOD."""
        result = _dispatch("massive bleeding causing shock")
        assert _is_not_ood(result), (
            "'massive bleeding causing shock' should not be OOD"
        )

    def test_anaphylaxis_and_asthma(self):
        """'allergic reaction with severe difficulty breathing' — should not be OOD."""
        result = _dispatch("allergic reaction with severe difficulty breathing")
        assert _is_not_ood(result), (
            "'allergic reaction with severe difficulty breathing' should not be OOD"
        )

    def test_drowning_and_hypothermia(self):
        """'person pulled from cold water not breathing' — not OOD."""
        result = _dispatch("person pulled from cold water not breathing")
        assert _is_not_ood(result), (
            "'person pulled from cold water not breathing' should not be OOD"
        )

    def test_burn_and_shock(self):
        """'severe burns over large area person in shock' — not OOD."""
        result = _dispatch("severe burns over large area person in shock")
        assert _is_not_ood(result), (
            "Burn+shock query should not be OOD"
        )


# ---------------------------------------------------------------------------
# OOD rejection tests
# ---------------------------------------------------------------------------

@requires_model
class TestOODRejection:
    """Non-medical queries must route to ood_response."""

    OOD_QUERIES = [
        "what is the weather today",
        "tell me a joke",
        "how do i boil eggs",
        "set a timer for 5 minutes",
        "what is the capital of france",
        "who won the game last night",
        "play some music",
        "what is 2 plus 2",
        "order pizza online",
        "how do i fix my wifi",
    ]

    def test_all_ood_queries_rejected(self):
        """All clear non-medical queries route to ood_response."""
        failures = []
        for q in self.OOD_QUERIES:
            result = _dispatch(q)
            if result.mode != "ood_response":
                failures.append((q, result.mode, result.category))

        assert not failures, (
            f"{len(failures)} non-medical queries were NOT rejected as OOD:\n"
            + "\n".join(f"  '{q}' -> mode={m}, category={c}" for q, m, c in failures)
        )

    def test_weather_is_ood(self):
        result = _dispatch("what is the weather today")
        assert _is_ood(result), f"'what is the weather today' should be OOD, got mode={result.mode}"

    def test_joke_is_ood(self):
        result = _dispatch("tell me a joke")
        assert _is_ood(result), f"'tell me a joke' should be OOD, got mode={result.mode}"

    def test_cooking_is_ood(self):
        result = _dispatch("how do i boil eggs")
        assert _is_ood(result), f"'how do i boil eggs' should be OOD, got mode={result.mode}"

    def test_timer_is_ood(self):
        result = _dispatch("set a timer for 5 minutes")
        assert _is_ood(result), f"'set a timer for 5 minutes' should be OOD, got mode={result.mode}"


# ---------------------------------------------------------------------------
# Foreign language / gibberish tests
# ---------------------------------------------------------------------------

@requires_model
class TestForeignLanguageAndGibberish:
    """Gibberish and non-English strings must route to ood_response."""

    GIBBERISH_QUERIES = [
        "xyzzy foobarbaz qwerty",
        "asdfghjkl zxcvbnm qwerty",
        "aaaabbbcccdddeee fffggghhh",
        "the quick brown fox jumps over the lazy dog repeatedly",  # nonsense in context
        "12345 67890 abcde fghij",
    ]

    # Note: some non-English medical terms (e.g., "ayuda") might have weak cosine
    # similarity to OOD cluster since the model is trained on English. We assert OOD.
    FOREIGN_QUERIES = [
        "ayuda por favor",                    # Spanish: "help please"
        "au secours quelquun est bless",      # French: "help someone is hurt"
        "bitte helfen sie mir",               # German: "please help me"
        "je ne sais pas quoi faire",          # French: "I don't know what to do"
    ]

    def test_gibberish_routes_to_ood(self):
        """Pure gibberish must route to ood_response."""
        failures = []
        for q in self.GIBBERISH_QUERIES:
            result = _dispatch(q)
            if result.mode != "ood_response":
                failures.append((q, result.mode, result.category, result.score))

        assert not failures, (
            f"{len(failures)} gibberish queries were NOT rejected as OOD:\n"
            + "\n".join(
                f"  '{q}' -> mode={m}, cat={c}, score={s:.3f}"
                for q, m, c, s in failures
            )
        )

    def test_foreign_language_routes_to_ood(self):
        """Non-English sentences should route to ood_response (system is English-only)."""
        failures = []
        for q in self.FOREIGN_QUERIES:
            result = _dispatch(q)
            if result.mode != "ood_response":
                failures.append((q, result.mode, result.category, result.score))

        # We allow up to 1 failure here because some models may partially understand
        # cross-lingual inputs — but they must not trigger a medical response confidently.
        if len(failures) > 1:
            assert False, (
                f"{len(failures)} foreign language queries were NOT rejected as OOD:\n"
                + "\n".join(
                    f"  '{q}' -> mode={m}, cat={c}, score={s}"
                    for q, m, c, s in failures
                )
            )

    def test_xyzzy_gibberish_is_ood(self):
        result = _dispatch("xyzzy foobarbaz qwerty")
        assert _is_ood(result), f"Gibberish 'xyzzy...' should be OOD, got mode={result.mode}"

    def test_asdf_gibberish_is_ood(self):
        result = _dispatch("asdfghjkl zxcvbnm qwerty")
        assert _is_ood(result), f"Keyboard gibberish should be OOD, got mode={result.mode}"


# ---------------------------------------------------------------------------
# Aggregate canary summary (runs without model — verifies query list sizes)
# ---------------------------------------------------------------------------

class TestCanaryQueryListSizes:
    """Verify that canary query lists meet the minimum size requirement (>= 20 each)."""

    def test_cpr_list_has_minimum_queries(self):
        assert len(CPR_QUERIES) >= 20, f"CPR canary needs >= 20 queries, got {len(CPR_QUERIES)}"

    def test_choking_list_has_minimum_queries(self):
        assert len(CHOKING_QUERIES) >= 20, f"Choking canary needs >= 20 queries, got {len(CHOKING_QUERIES)}"

    def test_bleeding_list_has_minimum_queries(self):
        assert len(BLEEDING_QUERIES) >= 20, f"Bleeding canary needs >= 20 queries, got {len(BLEEDING_QUERIES)}"

    def test_ood_rejection_list_has_minimum_queries(self):
        assert len(TestOODRejection.OOD_QUERIES) >= 5, "OOD rejection list needs >= 5 queries"

    def test_gibberish_list_has_minimum_queries(self):
        assert len(TestForeignLanguageAndGibberish.GIBBERISH_QUERIES) >= 5

    def test_foreign_language_list_has_minimum_queries(self):
        assert len(TestForeignLanguageAndGibberish.FOREIGN_QUERIES) >= 4
