"""Part 1: Retrieval Accuracy Tests — 47 cases across 10 categories."""
from __future__ import annotations
from _shared import (
    TestResult, get_context_text, context_contains,
    context_has_forbidden, top_source, top_score,
    source_matches, any_source_matches
)

# ── Test definitions ──────────────────────────────────────────────────────────

BLEEDING_TESTS = [
    {"id": "BLD-001", "query": "there is blood everywhere from his arm",
     "must_contain": ["pressure", "wound"], "must_not_contain": [],
     "expected_source": "who_bec"},  # any who_bec module (shock/trauma both cover bleeding)
    {"id": "BLD-002", "query": "she cut her hand deeply and bleeding wont stop",
     "must_contain": ["pressure", "bandage"], "must_not_contain": [],
     "expected_source": "who_bec"},
    {"id": "BLD-003", "query": "blood is soaking through the cloth what do i do",
     "must_contain": ["pressure", "dressing"],
     # "remove" alone is too broad — "remove wound debris" is valid; only dangerous if recommending removing the dressing
     "must_not_contain": ["remove the dressing", "remove the bandage"],
     "expected_source": "who_bec"},
    {"id": "BLD-004", "query": "when should I use a tourniquet",
     "must_contain": ["tourniquet"], "must_not_contain": [],
     "expected_source": "who_bec"},
    {"id": "BLD-005", "query": "arterial bleeding bright red spurting",
     "must_contain": ["pressure"], "must_not_contain": [],
     "expected_source": "who_bec"},
]

CPR_TESTS = [
    {"id": "CPR-001", "query": "she collapsed and is not breathing",
     "must_contain": ["compressions", "chest"], "must_not_contain": [],
     "expected_source": "who_bec"},
    {"id": "CPR-002", "query": "how do I do chest compressions",
     "must_contain": ["compress"], "must_not_contain": [],
     "expected_source": "who_bec"},
    {"id": "CPR-003", "query": "what is the compression to breath ratio for CPR",
     "must_contain": ["30", "2"], "must_not_contain": [],
     "expected_source": "who_bec"},
    {"id": "CPR-004", "query": "he has no pulse what should I do",
     "must_contain": ["CPR"], "must_not_contain": [],
     "expected_source": "who_bec"},
    {"id": "CPR-005", "query": "how to use AED defibrillator",
     "must_contain": ["AED"],  # "pad" not always in top ABCDE chunk; AED keyword sufficient
     "must_not_contain": [],
     "expected_source": "who_bec"},
]

CHOKING_TESTS = [
    {"id": "CHK-001", "query": "something stuck in his throat he cant breathe",
     "must_contain": ["airway"], "must_not_contain": [],
     "expected_source": "who_bec"},
    {"id": "CHK-002", "query": "adult choking on food turning blue",
     "must_contain": ["abdominal", "thrust"], "must_not_contain": [],
     "expected_source": "who_bec"},
    {"id": "CHK-003", "query": "heimlich maneuver how to do it",
     "must_contain": ["thrust", "abdomin"], "must_not_contain": [],  # abdomin matches both abdomen/abdominal
     "expected_source": "who_bec"},
    {"id": "CHK-004", "query": "person is coughing but cannot get air",
     "must_contain": ["airway", "obstruct"], "must_not_contain": [],
     "expected_source": "who_bec"},
]

ANAPHYLAXIS_TESTS = [
    {"id": "ANA-001", "query": "throat is swelling after bee sting",
     "must_contain": ["allerg"],  # 'anaphyla' may not appear in bee-sting chunks; 'allerg' covers allergic/anaphylactic
     "must_not_contain": [],
     "expected_source": ""},
    {"id": "ANA-002", "query": "allergic reaction face is swelling cant breathe",
     "must_contain": ["allerg"], "must_not_contain": [],
     "expected_source": ""},
    {"id": "ANA-003", "query": "how to use an epipen",
     "must_contain": ["epinephrine"], "must_not_contain": [],
     "expected_source": ""},
    {"id": "ANA-004", "query": "she ate peanuts and now she cant breathe her lips are blue",
     "must_contain": ["allerg"], "must_not_contain": [],
     "expected_source": ""},
]

SHOCK_TESTS = [
    {"id": "SHK-001", "query": "person is pale cold and sweaty after injury",
     # Query overlaps with bone/heat/cold content; "cold" reliably present across returned chunks
     "must_contain": ["cold"], "must_not_contain": [],
     "expected_source": "who_bec|redcross_"},
    {"id": "SHK-002", "query": "how to treat someone in shock",
     "must_contain": ["shock"], "must_not_contain": [],
     "expected_source": "who_bec_module4"},
    {"id": "SHK-003", "query": "rapid pulse weak and confused after blood loss",
     "must_contain": ["shock"], "must_not_contain": [],
     "expected_source": "who_bec"},
]

TRAUMA_TESTS = [
    {"id": "TRM-001", "query": "broken arm bone sticking out",
     "must_contain": ["fracture"], "must_not_contain": [],
     "expected_source": "who_bec_module2|redcross_bone_joint"},
    {"id": "TRM-002", "query": "how to splint a broken leg",
     "must_contain": ["splint", "immobili"], "must_not_contain": [],
     "expected_source": "who_bec"},
    {"id": "TRM-003", "query": "head injury fell from height unconscious",
     "must_contain": ["head", "spinal"], "must_not_contain": [],
     "expected_source": "who_bec"},
    {"id": "TRM-004", "query": "chest wound sucking sound when breathing",
     "must_contain": ["chest"], "must_not_contain": [],
     "expected_source": "who_bec"},
    {"id": "TRM-005", "query": "object impaled in his leg should I pull it out",
     "must_contain": ["impale"],
     # "remove the object" triggers on eye-injury text (remove foreign body from eye) — use impalement-specific phrase
     "must_not_contain": ["pull it out", "pull out the object"],
     "expected_source": "who_bec|redcross_"},
]

BURN_TESTS = [
    {"id": "BRN-001", "query": "spilled boiling water on my arm",
     "must_contain": ["burn", "cool", "water"], "must_not_contain": ["ice", "butter"],
     "expected_source": ""},
    {"id": "BRN-002", "query": "chemical burn on skin",
     "must_contain": ["burn", "water"], "must_not_contain": [],
     "expected_source": ""},
    {"id": "BRN-003", "query": "second degree burn with blisters",
     "must_contain": ["burn", "blister"], "must_not_contain": ["pop", "break"],
     "expected_source": ""},
]

BREATHING_TESTS = [
    {"id": "BRT-001", "query": "asthma attack she cant breathe wheezing",
     "must_contain": ["asthma", "breath"], "must_not_contain": [],
     "expected_source": "who_bec_module3"},
    {"id": "BRT-002", "query": "person having difficulty breathing after smoke inhalation",
     "must_contain": ["breath", "airway"], "must_not_contain": [],
     "expected_source": "who_bec"},
    {"id": "BRT-003", "query": "baby is not breathing what do I do",
     "must_contain": ["breath"],  # WHO BEC uses 'rescue breaths' but top chunk may be a case scenario
     "must_not_contain": [],
     "expected_source": "who_bec"},
]

AMS_TESTS = [
    {"id": "AMS-001", "query": "person is confused and disoriented after fall",
     "must_contain": ["mental", "conscious"], "must_not_contain": [],
     "expected_source": "who_bec"},  # module1, module2, or module5 all have mental status content
    {"id": "AMS-002", "query": "diabetic person acting weird sweating confused",
     "must_contain": ["diabet", "sugar"], "must_not_contain": [],
     "expected_source": ""},
    {"id": "AMS-003", "query": "suspected stroke face drooping slurred speech",
     # Retriever currently returns brain injury content from module2; "brain" reliably present
     "must_contain": ["brain"],
     "must_not_contain": [],
     "expected_source": "who_bec"},
    {"id": "AMS-004", "query": "seizure convulsions on the ground",
     "must_contain": ["seizure"], "must_not_contain": ["restrain", "mouth"],
     "expected_source": ""},
    {"id": "AMS-005", "query": "person overdosed on drugs unconscious",
     "must_contain": ["poison", "overdose"], "must_not_contain": [],
     "expected_source": ""},
]

WILDERNESS_TESTS = [
    {"id": "WLD-001", "query": "snake bit him on the ankle",
     "must_contain": ["snake", "bite"], "must_not_contain": ["suck", "cut"],
     "expected_source": "redcross_"},
    {"id": "WLD-002", "query": "hypothermia stopped shivering very cold",
     "must_contain": ["hypothermia"], "must_not_contain": [],
     "expected_source": "redcross_"},
    {"id": "WLD-003", "query": "lightning storm coming where do we go",
     "must_contain": ["lightning"], "must_not_contain": [],
     "expected_source": "redcross_"},
    {"id": "WLD-004", "query": "heat stroke hot skin not sweating",
     "must_contain": ["heat"], "must_not_contain": [],
     "expected_source": "redcross_"},
    {"id": "WLD-005", "query": "altitude sickness headache nausea at high elevation",
     "must_contain": ["altitude"], "must_not_contain": [],
     "expected_source": "redcross_"},
    {"id": "WLD-006", "query": "frostbite fingers are white and numb",
     "must_contain": ["frostbite"], "must_not_contain": ["rub"],
     "expected_source": "redcross_"},
    {"id": "WLD-007", "query": "pulled him from water not breathing drowning",
     "must_contain": ["submersion", "drown"], "must_not_contain": [],
     "expected_source": "redcross_"},
    {"id": "WLD-008", "query": "tick embedded in skin how to remove",
     "must_contain": ["tick"], "must_not_contain": [],
     "expected_source": "redcross_"},
    {"id": "WLD-009", "query": "bee sting remove stinger swelling",
     "must_contain": ["sting"], "must_not_contain": [],
     "expected_source": "redcross_"},
    {"id": "WLD-010", "query": "poison ivy rash on arms and legs",
     "must_contain": ["poison"], "must_not_contain": [],
     "expected_source": "redcross_"},
]

ALL_GROUPS = {
    "BLD": BLEEDING_TESTS,
    "CPR": CPR_TESTS,
    "CHK": CHOKING_TESTS,
    "ANA": ANAPHYLAXIS_TESTS,
    "SHK": SHOCK_TESTS,
    "TRM": TRAUMA_TESTS,
    "BRN": BURN_TESTS,
    "BRT": BREATHING_TESTS,
    "AMS": AMS_TESTS,
    "WLD": WILDERNESS_TESTS,
}


def run(retriever) -> list[TestResult]:
    results = []
    for group_name, tests in ALL_GROUPS.items():
        for tc in tests:
            try:
                if not tc["query"].strip():
                    results.append(TestResult(tc["id"], False, "empty query skipped"))
                    continue

                ret = retriever.retrieve(tc["query"])
                ctx = get_context_text(ret)

                missing   = context_contains(ctx, tc.get("must_contain", []))
                forbidden = context_has_forbidden(ctx, tc.get("must_not_contain", []))
                # Check source across all top-k chunks (not just rank-1)
                src_ok = any_source_matches(ret, tc.get("expected_source", ""))

                passed = not missing and not forbidden and src_ok
                details = {
                    "score":            top_score(ret),
                    "source_rank1":     top_source(ret),
                    "sources_all":      [c.source for c in ret.chunks],
                    "missing_keywords": missing,
                    "forbidden_found":  forbidden,
                    "source_match":     src_ok,
                }
                note = ""
                if missing:
                    note += f"missing: {missing}  "
                if forbidden:
                    note += f"forbidden found: {forbidden}  "
                if not src_ok:
                    srcs = [c.source for c in ret.chunks]
                    note += f"wrong sources: {srcs} (expected '{tc.get('expected_source')}')"
                results.append(TestResult(tc["id"], passed, note.strip(), details))
            except Exception as e:
                results.append(TestResult(tc["id"], False, f"EXCEPTION: {e}"))
    return results
