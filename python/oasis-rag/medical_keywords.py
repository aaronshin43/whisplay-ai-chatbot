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
        "hemorrhage control", "haemorrhage control",   # protocol language
        # ── Colloquial user language ──────────────────────────────────────
        "blood",                                        # "there is blood everywhere"
        "wont stop", "won't stop", "cant stop", "can't stop",  # bleeding that won't stop
        "gushing", "soaking through", "soaked through", "soaked",  # severity descriptors
        "cut", "cuts", "deep cut", "gash",             # wound descriptions
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
        # ── Colloquial user language ──────────────────────────────────────
        "cant breathe", "can't breathe",               # "I cant breathe"
        "cant cough", "can't cough",                   # choking: unable to cough
        "cannot get air", "can't get air", "cannot breathe", "no air",  # partial obstruction
        "turning blue", "going blue", "lips blue",     # cyanosis descriptors
        "something stuck", "stuck in throat",          # foreign body
    ],

    # ── 3. Breathing / Respiratory ────────────────────────────────────────
    "respiratory": [
        "breathing", "respiration", "respiratory arrest", "respiratory distress",
        "dyspnea", "shortness of breath", "hyperventilation", "hypoventilation",
        "rescue breathing", "mouth-to-mouth", "ventilation",
        "pneumothorax", "tension pneumothorax", "hemothorax",
        "flail chest", "rib fracture", "chest injury", "open chest wound",
        "sucking chest wound", "sucking wound", "occlusive dressing", "needle decompression",
        "cyanosis", "hypoxia", "oxygen saturation", "SpO2",
        "asthma", "asthma attack", "bronchospasm", "anaphylaxis",
        # ── Penetrating chest trauma ───────────────────────────────────────
        "knife", "stab", "stabbing", "stabbed",
        "penetrating", "object in chest", "chest wound",
        "pull the knife", "knife in chest",
    ],

    # ── 4a. Cardiac Arrest / CPR ──────────────────────────────────────────
    "circulation_cardiac": [
        "CPR", "cardiopulmonary resuscitation", "cardiac arrest",
        "chest compression", "defibrillation", "AED", "automated external defibrillator",
        "ventricular fibrillation", "VF", "ventricular tachycardia", "VT",
        "pulseless", "no pulse", "pulse check", "carotid pulse", "radial pulse",
        "heart attack", "myocardial infarction", "STEMI",
        "chest pain", "angina", "palpitation",
        # Collapse / arrest presentation terms
        "collapse", "collapsed", "sudden collapse", "unresponsive",
        "not breathing", "no breathing", "stopped breathing",
        "rescue breathing", "agonal breathing", "gasping",
        "chain of survival", "ROSC", "return of spontaneous circulation",
        # ── Colloquial user language ──────────────────────────────────────
        "dying", "he is dying", "she is dying",
        "not moving", "wont move", "won't move",
        "not waking", "wont wake", "won't wake up",
        "heart stopped", "stopped his heart",
    ],

    # ── 4b. Shock / Circulatory Failure ───────────────────────────────────
    # Kept separate so "shock" queries expand to fluid/perfusion terms,
    # not CPR terms — avoids retrieving CPR chunks for hypovolemic shock queries.
    "shock_management": [
        "shock", "hypovolemic shock", "cardiogenic shock", "septic shock",
        "anaphylactic shock", "neurogenic shock", "obstructive shock",
        "distributive shock", "circulatory failure", "circulatory collapse",
        "hypotension", "tachycardia", "bradycardia", "arrhythmia",
        "poor perfusion", "capillary refill", "mottled skin",
        "IV fluid", "IV access", "fluid resuscitation", "fluid bolus",
        "normal saline", "Ringer's lactate", "intraosseous", "IO access",
        "pale cold sweaty", "weak pulse", "rapid pulse", "low blood pressure",
        "in shock", "going into shock", "signs of shock",
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
        "scald", "boiling water", "hot water", "steam burn",
        "burned", "burning skin", "blistering",
    ],

    # ── 6. Fractures / Musculoskeletal ────────────────────────────────────
    "fractures_ortho": [
        "fracture", "broken bone", "open fracture", "compound fracture",
        "closed fracture", "greenstick fracture", "comminuted fracture",
        "splint", "splinting", "immobilization", "traction",
        "spinal injury", "cervical spine", "spinal cord injury",
        "spinal", "spine", "neck injury",
        "cant feel", "cannot feel", "no feeling", "no sensation",
        "paralysis", "paralyzed", "paralysed",
        "numb legs", "numb feet", "numbness below",
        "dont move", "do not move", "keep still",
        "log roll", "cervical collar", "c-collar",
        "dislocation", "subluxation", "reduction",
        "femur fracture", "tibia fracture", "radius fracture",
        "clavicle fracture", "rib fracture", "pelvic fracture",
        "skull fracture", "crepitus", "neurovascular status",
        "bone sticking out", "deformed limb", "angulated",
        "sprain", "strain", "joint injury", "sling", "swathe",
        # ── Trauma mechanism / colloquial triggers ────────────────────────
        "fell from height", "fell from", "fall from height", "fell off",
        "hit head", "head trauma",          # biases toward trauma.md head injury section
        # ── Impaled / penetrating object (non-chest) ─────────────────────
        "impaled", "impalement",            # moved from respiratory — routes to trauma.md
        "object impaled", "object stuck in leg", "object stuck in arm",
        "should i pull", "pull it out", "pull out the object",
        "remove the object", "take it out",
    ],

    # ── 7. Neurological / Head Trauma ─────────────────────────────────────
    "neuro_head": [
        "head injury", "traumatic brain injury", "concussion",  # removed "TBI" — false-positive in "frostbite"
        "intracranial pressure", "cerebral hemorrhage",         # removed "ICP" — risky short abbreviation
        "subdural hematoma", "epidural hematoma", "subarachnoid hemorrhage",
        "Glasgow Coma Scale", "GCS", "pupils", "pupil response",
        "altered consciousness", "loss of consciousness", "LOC",
        "consciousness", "level of consciousness", "unconscious",
        "seizure", "convulsion", "epilepsy", "status epilepticus",
        "stroke", "CVA", "cerebrovascular", "FAST assessment",
        "facial droop", "face drooping", "slurred speech", "arm weakness",
        "speech difficulty", "sudden weakness", "sudden numbness",
        "AVPU", "alert", "voice", "pain response", "unresponsive",
        "dizziness", "vertigo", "headache", "vomiting after head injury",
        "altered mental status", "AMS", "confused", "disoriented",
    ],

    # ── 8. Hypothermia / Hyperthermia ─────────────────────────────────────
    "temperature_emergencies": [
        "hypothermia", "frostbite", "cold exposure", "immersion foot",
        "trench foot", "rewarming", "active rewarming", "passive rewarming",
        "heat stroke", "heat exhaustion", "hyperthermia", "heat cramps",
        "sunstroke", "dehydration", "heat index",
        "core temperature", "body temperature", "shivering",
        "stopped shivering", "not shivering", "rigid muscles",
        "altered mental status from cold", "after-drop",
        "hypothermia wrap", "warm", "rewarm", "cold skin",
        "frostbitten", "frozen", "numb fingers", "numb toes",
    ],

    # ── 9. Poisoning / Toxicology ─────────────────────────────────────────
    "poisoning_tox": [
        "poisoning", "overdose", "intoxication", "toxic ingestion",
        "activated charcoal", "antidote", "naloxone", "Narcan",
        "opioid overdose", "carbon monoxide poisoning", "CO poisoning",
        "cyanide", "organophosphate", "snake bite", "snakebite", "envenomation",
        "snake bit", "bit by snake", "bitten by snake", "bitten",  # past tense variants
        "insect sting", "bee sting", "wasp sting", "spider bite", "jellyfish sting",
        "tick bite", "tick removal", "scorpion sting",
        "venom", "antivenin", "antivenom",
        "corrosive ingestion", "acid", "alkali",
        "poison ivy", "poison oak", "plant rash", "contact dermatitis",
    ],

    # ── 10-A. Anaphylaxis / Allergy (expanded from respiratory) ──────────
    "anaphylaxis_allergy": [
        "anaphylaxis", "anaphylactic", "anaphylactic shock", "anaphylactic reaction",
        "epinephrine", "epipen", "auto-injector", "adrenaline", "IM adrenaline",
        "severe allergic reaction", "allergic reaction", "allergy", "allergic",
        "urticaria", "hives", "swelling throat", "throat closing",
        "lip swelling", "tongue swelling", "angioedema",
        "bee sting allergy", "food allergy", "peanut allergy",
        "antihistamine", "diphenhydramine", "benadryl",
        # ── Colloquial user language ──────────────────────────────────────
        "throat swelling",                              # word-order fix: "throat swelling" ≠ "swelling throat"
        "face swelling", "face swollen",               # angioedema descriptors
        "cant breathe", "can't breathe",               # anaphylaxis presentation
        "throat closing", "closing throat",            # airway compromise
    ],

    # ── 10-B. Diabetic / Metabolic Emergencies ────────────────────────────
    "metabolic_endocrine": [
        "diabetic", "diabetes", "diabetic emergency",
        "insulin", "blood sugar", "glucose", "dextrose",
        "hypoglycemia", "low blood sugar", "hyperglycemia", "high blood sugar",
        "diabetic ketoacidosis", "DKA",
        "sweating confused", "shakiness", "trembling",
        "orange juice", "sugar", "glucagon",
    ],

    # ── 10-C. Altitude / Wilderness Emergencies ────────────────────────────
    "altitude_wilderness": [
        "altitude", "altitude sickness", "acute mountain sickness", "AMS",
        "HAPE", "high altitude pulmonary edema",
        "HACE", "high altitude cerebral edema",
        "acclimatization", "acclimatize", "descend", "descent",
        "high elevation", "mountain sickness", "thin air",
        "headache nausea altitude", "altitude headache",
        "avalanche", "buried in snow", "snow burial",
        "drowning", "submersion", "near-drowning", "water rescue",
    ],

    # ── 10-D. Lightning / Electrical Injuries ─────────────────────────────
    "lightning_electrical": [
        "lightning", "lightning strike", "lightning storm", "thunder",
        "struck by lightning", "lightning hit", "lightning victim",
        "electric shock", "electrocution", "electrocuted",
        "live wire", "power line", "electrical injury", "electrical burn",
        "touched wire", "shocked by electricity",
        "circuit breaker", "fuse box", "voltage", "high voltage",
        "entry wound exit wound",   # electrical burn pattern
        "reverse triage",           # lightning-specific triage rule
    ],

    # ── 11. Obstetric / Pediatric ─────────────────────────────────────────
    "obstetric_pediatric": [
        "childbirth", "delivery", "emergency delivery", "precipitous labor",
        "cord prolapse", "placenta", "postpartum hemorrhage",
        "eclampsia", "pre-eclampsia", "neonatal resuscitation",
        "pediatric CPR", "infant CPR", "child CPR",
        "febrile seizure", "croup", "epiglottitis",
        "SIDS", "sudden infant death",
    ],

    # ── 12. Triage / Assessment ───────────────────────────────────────────
    "triage_assessment": [
        "triage", "START triage", "SALT triage", "mass casualty",
        "MCI", "immediate", "delayed", "minimal", "expectant",
        "primary survey", "secondary survey", "ABCDE", "SAMPLE history",
        "vital signs", "blood pressure", "heart rate", "respiratory rate",
        "capillary refill", "skin color", "skin temperature",
        "mental status", "AVPU", "Glasgow Coma Scale", "GCS score",
        "mechanism of injury", "MOI",
        "consciousness", "level of consciousness", "responsiveness",
        "pale skin", "cold clammy skin", "sweating", "pallor",
        "perfusion", "peripheral perfusion", "poor perfusion",
        "hypovolaemic", "hypovolemic", "blood volume",
    ],

    # ── 13-A. Wound Infection / Antibiotics ──────────────────────────────
    "wound_infection": [
        "antibiotic", "antibiotics", "infection", "infected", "infect",
        "wound infection", "infected wound", "sepsis", "cellulitis",
        "pus", "purulent", "abscess", "necrosis", "gangrene",
        "tetanus", "debridement",
    ],

    # ── 13. WHO BEC / IFRC Protocol Terms ────────────────────────────────
    "who_bec_protocol": [
        "basic emergency care", "BEC", "WHO BEC",
        "WHO first aid", "IFRC", "Red Cross",
        "essential emergency care", "emergency health system",
        "point-of-care", "resource-limited", "low-resource setting",
        "standard precautions", "universal precautions",
        "personal protective equipment", "gloves", "mask",  # removed "PPE" — false-positive in "stopped", "happening"
        "scene safety", "danger assessment", "call for help",
        "do no further harm", "informed consent",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Colloquial (detection-only) terms
#
# These terms are used to MAP user queries to categories in Stage 1 (detection
# and keyword_map lookup), but are EXCLUDED from expand_query so they do NOT
# inflate the query_terms denominator in the Stage 2 lexical score calculation.
# Medical document chunks use formal clinical language, not colloquial phrases,
# so including these terms in query_terms would always produce 0 hits and dilute
# lexical_score for every chunk.
# ─────────────────────────────────────────────────────────────────────────────
_COLLOQUIAL_TERMS: frozenset[str] = frozenset({
    # hemorrhage_control colloquial
    "blood", "wont stop", "won't stop", "cant stop", "can't stop",
    "gushing", "soaking through", "soaked through", "soaked",
    "cut", "cuts", "deep cut", "gash",
    # airway_management colloquial
    "cant breathe", "can't breathe", "cant cough", "can't cough",
    "turning blue", "going blue", "lips blue",
    "something stuck", "stuck in throat",
    # circulation_cardiac colloquial
    "dying", "he is dying", "she is dying",
    "not moving", "wont move", "won't move",
    "not waking", "wont wake", "won't wake up",
    "heart stopped", "stopped his heart",
    # anaphylaxis_allergy colloquial
    "throat swelling", "face swelling", "face swollen",
    "closing throat",
    # fractures_ortho colloquial
    "spinal", "spine", "neck injury",
    "cant feel", "cannot feel", "no feeling", "no sensation",
    "paralysis", "paralyzed", "paralysed",
    "numb legs", "numb feet", "numbness below",
    "dont move", "do not move", "keep still",
    # lightning_electrical colloquial
    "lightning hit", "struck by lightning",
    # fractures_ortho mechanism triggers (not in medical doc text)
    "fell from height", "fell from", "fall from height", "fell off",
    "hit head", "head trauma",
    # poisoning_tox colloquial
    "snake bit", "bit by snake", "bitten by snake", "bitten",
    # respiratory / penetrating chest trauma colloquial
    "pull the knife", "knife in chest",
    "object in chest",
    # fractures_ortho / impaled object colloquial
    "object impaled", "object stuck in leg", "object stuck in arm",
    "should i pull", "pull it out", "pull out the object",
    "remove the object", "take it out",
})

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

    Colloquial detection terms (_COLLOQUIAL_TERMS) are excluded from the result
    because they don't appear in medical document chunk texts and would dilute
    the Stage-2 lexical score (query_terms denominator) without adding hits.
    """
    detected  = detect_keywords(query)
    seen_cats: set[str] = set()
    expanded:  list[str] = []

    for _term, category in detected:
        if category not in seen_cats:
            seen_cats.add(category)
            expanded.extend(get_category_terms(category))

    # Deduplicate while preserving order; skip colloquial detection-only terms
    seen: set[str] = set()
    result: list[str] = []
    for t in expanded:
        tl = t.lower()
        if tl not in seen and tl not in _COLLOQUIAL_TERMS:
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
