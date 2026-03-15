"""Part 3: Scenario Coverage — every scenario must return score >= 0.70."""
from __future__ import annotations
from _shared import TestResult, top_score, top_cosine

# hybrid_score for gte-small tops out ~0.60-0.68 even for perfect matches.
# Use cosine_score (range 0-1, normalised) which reliably reaches 0.80+ for
# correct retrievals. Threshold 0.75 separates good hits from noise.
MIN_COSINE = 0.75

COVERAGE_SCENARIOS = [
    {"id": "COV-001", "scenario": "bleeding",               "query": "severe bleeding"},
    {"id": "COV-002", "scenario": "fractures",              "query": "broken bone fracture"},
    {"id": "COV-003", "scenario": "burns",                  "query": "burn injury"},
    {"id": "COV-004", "scenario": "choking",                "query": "choking obstructed airway"},
    {"id": "COV-005", "scenario": "allergic reactions",     "query": "severe allergic reaction anaphylaxis"},
    {"id": "COV-006", "scenario": "shock stabilization",    "query": "shock treatment"},
    {"id": "COV-007", "scenario": "CPR",                    "query": "cardiac arrest CPR"},
    {"id": "COV-008", "scenario": "ABCDE approach",         "query": "ABCDE patient assessment"},
    {"id": "COV-009", "scenario": "trauma",                 "query": "major trauma injury"},
    {"id": "COV-010", "scenario": "breathing difficulty",   "query": "difficulty breathing respiratory distress"},
    {"id": "COV-011", "scenario": "shock types",            "query": "hypovolaemic shock blood loss"},
    {"id": "COV-012", "scenario": "altered mental status",  "query": "unconscious altered mental status"},
    {"id": "COV-013", "scenario": "stroke",                 "query": "stroke symptoms face drooping"},
    {"id": "COV-014", "scenario": "poisoning",              "query": "poisoning ingested toxic substance"},
    {"id": "COV-015", "scenario": "spinal injury",          "query": "spinal cord injury neck"},
    {"id": "COV-016", "scenario": "head injury",            "query": "head injury concussion"},
    {"id": "COV-017", "scenario": "chest injury",           "query": "chest wound pneumothorax"},
    {"id": "COV-018", "scenario": "abdominal injury",       "query": "abdominal injury internal bleeding"},
    {"id": "COV-019", "scenario": "snake bite",             "query": "snake bite venom"},
    {"id": "COV-020", "scenario": "hypothermia",            "query": "hypothermia cold exposure"},
    {"id": "COV-021", "scenario": "heat stroke",            "query": "heat stroke hyperthermia"},
    {"id": "COV-022", "scenario": "lightning",              "query": "lightning injury strike"},
    {"id": "COV-023", "scenario": "altitude sickness",      "query": "altitude sickness mountain"},
    {"id": "COV-024", "scenario": "frostbite",              "query": "frostbite frozen skin"},
    {"id": "COV-025", "scenario": "drowning",               "query": "drowning submersion rescue"},
    {"id": "COV-026", "scenario": "wound infection",        "query": "wound infection signs"},
    {"id": "COV-027", "scenario": "diabetic emergency",     "query": "diabetic emergency low blood sugar"},
    {"id": "COV-028", "scenario": "seizure",                "query": "seizure convulsion"},
    {"id": "COV-029", "scenario": "asthma",                 "query": "asthma attack inhaler"},
    {"id": "COV-030", "scenario": "eye injury",             "query": "eye injury chemical splash"},
]


def run(retriever) -> list[TestResult]:
    results = []
    for sc in COVERAGE_SCENARIOS:
        try:
            ret = retriever.retrieve(sc["query"])
            cosine = top_cosine(ret)
            hybrid = top_score(ret)
            has_context = bool(ret.context and ret.context.strip())
            passed = has_context and cosine >= MIN_COSINE
            note = f"cosine={cosine:.4f} hybrid={hybrid:.4f}"
            if not has_context:
                note += " | NO CONTEXT returned"
            elif cosine < MIN_COSINE:
                note += f" | cosine below threshold {MIN_COSINE}"
            results.append(TestResult(
                sc["id"], passed,
                f"[{sc['scenario']}] {note}",
                {"cosine": cosine, "hybrid": hybrid,
                 "has_context": has_context, "scenario": sc["scenario"]}
            ))
        except Exception as e:
            results.append(TestResult(sc["id"], False, f"EXCEPTION: {e}"))
    return results
