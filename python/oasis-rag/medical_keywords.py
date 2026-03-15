"""
medical_keywords.py — O.A.S.I.S. RAG Phase 1

Curated medical keyword dictionary for query expansion and relevance boosting.
Organized into clinical categories aligned with WHO Basic Emergency Care (BEC)
and IFRC First Aid guidelines.

Usage:
    from medical_keywords import MEDICAL_KEYWORDS, get_category, expand_query

    # All 200+ terms as a flat set
    all_terms = MEDICAL_KEYWORDS

    # Category for a term
    cat = get_category("tourniquet")   # → "hemorrhage_control"

    # Expand a user query with related synonyms
    expanded = expand_query("he is bleeding a lot")
    # → ["bleeding", "hemorrhage", "tourniquet", "hemostasis", ...]
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Keyword taxonomy
# Each category maps to a list of (term, synonyms/related_terms) pairs.
# ─────────────────────────────────────────────────────────────────────────────

_TAXONOMY: dict[str, list[str]] = {

    # ── 1. Hemorrhage / Bleeding Control ──────────────────────────────────
    "hemorrhage_control": [
        "bleeding", "hemorrhage", "haemorrhage", "blood loss", "exsanguination",
        "tourniquet", "hemostasis", "haemostasis", "wound packing", "direct pressure",
        "pressure dressing", "hemostatic", "hemostatic dressing", "clotting",
        "coagulation", "platelet", "blood clot", "arterial bleeding",
        "venous bleeding", "capillary bleeding", "internal bleeding",
        "laceration", "puncture wound", "penetrating wound", "wound",
        "junctional hemorrhage", "wound gauze", "combat gauze",
        "elevation", "pressure point", "femoral artery", "brachial artery",
        "carotid artery", "subclavian artery", "aorta",
    ],

    # ── 2. Airway Management ──────────────────────────────────────────────
    "airway_management": [
        "airway", "airway obstruction", "choking", "foreign body",
        "heimlich maneuver", "abdominal thrust", "back blow", "chest thrust",
        "jaw thrust", "head tilt", "chin lift", "recovery position",
        "unconscious", "unresponsive", "gurgling", "stridor", "wheeze",
        "aspiration", "tongue obstruction", "oropharyngeal", "nasopharyngeal",
        "supraglottic", "intubation", "bag-valve-mask", "BVM",
        "tracheotomy", "cricothyrotomy", "snoring", "apnea",
    ],

    # ── 3. Breathing / Respiratory ────────────────────────────────────────
    "respiratory": [
        "breathing", "respiration", "respiratory arrest", "respiratory distress",
        "dyspnea", "shortness of breath", "hyperventilation", "hypoventilation",
        "rescue breathing", "mouth-to-mouth", "ventilation",
        "pneumothorax", "tension pneumothorax", "hemothorax",
        "flail chest", "rib fracture", "chest injury", "open chest wound",
        "sucking chest wound", "occlusive dressing", "needle decompression",
        "cyanosis", "hypoxia", "oxygen saturation", "SpO2",
        "asthma", "asthma attack", "bronchospasm", "anaphylaxis",
    ],

    # ── 4. Circulation / Cardiac ──────────────────────────────────────────
    "circulation_cardiac": [
        "CPR", "cardiopulmonary resuscitation", "cardiac arrest",
        "chest compression", "defibrillation", "AED", "automated external defibrillator",
        "ventricular fibrillation", "VF", "ventricular tachycardia", "VT",
        "pulseless", "no pulse", "pulse check", "carotid pulse", "radial pulse",
        "shock", "hypovolemic shock", "cardiogenic shock", "septic shock",
        "anaphylactic shock", "neurogenic shock", "obstructive shock",
        "hypotension", "tachycardia", "bradycardia", "arrhythmia",
        "heart attack", "myocardial infarction", "MI", "STEMI",
        "chest pain", "angina", "palpitation",
        # Collapse / arrest presentation terms
        "collapse", "collapsed", "sudden collapse", "unresponsive",
        "not breathing", "no breathing", "stopped breathing",
        "rescue breathing", "agonal breathing", "gasping",
        "chain of survival", "ROSC", "return of spontaneous circulation",
    ],

    # ── 5. Burns ──────────────────────────────────────────────────────────
    "burns": [
        "burn", "thermal burn", "chemical burn", "electrical burn",
        "radiation burn", "inhalation injury", "smoke inhalation",
        "first degree burn", "second degree burn", "third degree burn",
        "superficial burn", "partial thickness", "full thickness",
        "rule of nines", "TBSA", "total body surface area",
        "blister", "eschar", "debridement", "burn dressing",
        "cool water", "do not ice", "burn center",
        "flash burn", "contact burn", "scalding",
    ],

    # ── 6. Fractures / Musculoskeletal ────────────────────────────────────
    "fractures_ortho": [
        "fracture", "broken bone", "open fracture", "compound fracture",
        "closed fracture", "greenstick fracture", "comminuted fracture",
        "splint", "splinting", "immobilization", "traction",
        "spinal injury", "cervical spine", "spinal cord injury",
        "log roll", "cervical collar", "c-collar",
        "dislocation", "subluxation", "reduction",
        "femur fracture", "tibia fracture", "radius fracture",
        "clavicle fracture", "rib fracture", "pelvic fracture",
        "skull fracture", "crepitus", "neurovascular status",
    ],

    # ── 7. Neurological / Head Trauma ─────────────────────────────────────
    "neuro_head": [
        "head injury", "traumatic brain injury", "TBI", "concussion",
        "intracranial pressure", "ICP", "cerebral hemorrhage",
        "subdural hematoma", "epidural hematoma", "subarachnoid hemorrhage",
        "Glasgow Coma Scale", "GCS", "pupils", "pupil response",
        "altered consciousness", "loss of consciousness", "LOC",
        "seizure", "convulsion", "epilepsy", "status epilepticus",
        "stroke", "FAST", "facial droop", "arm weakness", "speech",
        "dizziness", "vertigo", "headache", "vomiting after head injury",
    ],

    # ── 8. Hypothermia / Hyperthermia ─────────────────────────────────────
    "temperature_emergencies": [
        "hypothermia", "frostbite", "cold exposure", "immersion foot",
        "trench foot", "rewarming", "active rewarming", "passive rewarming",
        "heat stroke", "heat exhaustion", "hyperthermia", "heat cramps",
        "sunstroke", "dehydration", "heat index",
        "core temperature", "body temperature", "shivering",
        "altered mental status from cold", "after-drop",
    ],

    # ── 9. Poisoning / Toxicology ─────────────────────────────────────────
    "poisoning_tox": [
        "poisoning", "overdose", "intoxication", "toxic ingestion",
        "activated charcoal", "antidote", "naloxone", "Narcan",
        "opioid overdose", "carbon monoxide poisoning", "CO poisoning",
        "cyanide", "organophosphate", "snake bite", "envenomation",
        "insect sting", "bee sting", "spider bite", "jellyfish sting",
        "corrosive ingestion", "acid", "alkali",
    ],

    # ── 10. Obstetric / Pediatric ─────────────────────────────────────────
    "obstetric_pediatric": [
        "childbirth", "delivery", "emergency delivery", "precipitous labor",
        "cord prolapse", "placenta", "postpartum hemorrhage",
        "eclampsia", "pre-eclampsia", "neonatal resuscitation",
        "pediatric CPR", "infant CPR", "child CPR",
        "febrile seizure", "croup", "epiglottitis",
        "SIDS", "sudden infant death",
    ],

    # ── 11. Triage / Assessment ───────────────────────────────────────────
    "triage_assessment": [
        "triage", "START triage", "SALT triage", "mass casualty",
        "MCI", "immediate", "delayed", "minimal", "expectant",
        "primary survey", "secondary survey", "ABCDE", "SAMPLE history",
        "vital signs", "blood pressure", "heart rate", "respiratory rate",
        "capillary refill", "skin color", "skin temperature",
        "mental status", "AVPU", "Glasgow Coma Scale",
        "mechanism of injury", "MOI",
    ],

    # ── 12. WHO BEC / IFRC Protocol Terms ────────────────────────────────
    "who_bec_protocol": [
        "basic emergency care", "BEC", "WHO BEC",
        "WHO first aid", "IFRC", "Red Cross",
        "essential emergency care", "emergency health system",
        "point-of-care", "resource-limited", "low-resource setting",
        "standard precautions", "universal precautions",
        "personal protective equipment", "PPE", "gloves", "mask",
        "scene safety", "danger assessment", "call for help",
        "do no further harm", "informed consent",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Flat set of all keywords (for fast membership testing)
# ─────────────────────────────────────────────────────────────────────────────
MEDICAL_KEYWORDS: frozenset[str] = frozenset(
    term.lower()
    for terms in _TAXONOMY.values()
    for term in terms
)

# Reverse index: term → category
_TERM_TO_CATEGORY: dict[str, str] = {
    term.lower(): category
    for category, terms in _TAXONOMY.items()
    for term in terms
}


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_category(term: str) -> str | None:
    """Return the taxonomy category for a given term, or None if unknown."""
    return _TERM_TO_CATEGORY.get(term.lower())


def get_category_terms(category: str) -> list[str]:
    """Return all terms in a taxonomy category."""
    return list(_TAXONOMY.get(category, []))


def detect_keywords(text: str) -> list[tuple[str, str]]:
    """
    Scan *text* and return (keyword, category) pairs that appear in it.
    Case-insensitive. Results are deduplicated and ordered by position.
    """
    text_lower  = text.lower()
    found: dict[str, str] = {}

    for term, category in _TERM_TO_CATEGORY.items():
        if term in text_lower and term not in found:
            found[term] = category

    # Sort by occurrence position for deterministic output
    return sorted(found.items(), key=lambda kv: text_lower.index(kv[0]))


def expand_query(query: str) -> list[str]:
    """
    Given a user query, detect matching keywords and return all related terms
    from the same categories — useful for query expansion before retrieval.
    """
    detected  = detect_keywords(query)
    seen_cats: set[str] = set()
    expanded:  list[str] = []

    for _term, category in detected:
        if category not in seen_cats:
            seen_cats.add(category)
            expanded.extend(get_category_terms(category))

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for t in expanded:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            result.append(t)

    return result


def keyword_score(text: str) -> float:
    """
    Return a normalized relevance score [0, 1] based on how many distinct
    medical keywords appear in *text*. Useful for pre-filtering chunks.
    """
    found = detect_keywords(text)
    if not found:
        return 0.0
    # Cap at 20 unique keywords → score 1.0
    return min(len(found) / 20.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Total keywords: {len(MEDICAL_KEYWORDS)}")
    print(f"Categories    : {list(_TAXONOMY.keys())}")

    test_query = "patient is bleeding heavily from the leg and is going into shock"
    print(f"\nQuery: {test_query!r}")

    detected = detect_keywords(test_query)
    print(f"Detected : {detected}")

    expanded = expand_query(test_query)
    print(f"Expanded ({len(expanded)} terms): {expanded[:10]}...")

    score = keyword_score(test_query)
    print(f"Score    : {score:.2f}")
