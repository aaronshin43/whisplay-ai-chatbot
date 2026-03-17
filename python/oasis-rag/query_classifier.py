"""
query_classifier.py — O.A.S.I.S. Query Classification Layer

Classifies a user query into:
  - emergency_type : primary emergency category (str)
  - body_parts     : set of affected body parts (set[str])
  - severity       : "critical" | "urgent" | "moderate" | "unknown"
  - confidence     : 0.0 – 1.0

Used by retriever.py to apply body-part filtering (penalising chunks about
different anatomy) so that "broken finger" never surfaces chest/rib steps.

Usage:
    from query_classifier import classify_query, QueryClassification
    qc = classify_query("I broke my finger, there is bone sticking out")
    # qc.emergency_type → "fractures_ortho"
    # qc.body_parts     → {"finger", "hand"}
    # qc.severity       → "urgent"
"""

from __future__ import annotations

from dataclasses import dataclass, field

from medical_keywords import detect_keywords


# ── Body-part taxonomy ────────────────────────────────────────────────────────
# Maps body-part keywords to canonical part names.
# Canonical names are also used in chunk metadata for filtering.

_BODY_PART_MAP: dict[str, str] = {
    # Head / Brain
    "head": "head", "skull": "head", "brain": "head", "scalp": "head",
    "forehead": "head", "face": "head", "jaw": "head", "cheek": "head",
    "temple": "head",
    # Neck / Spine
    "neck": "neck", "spine": "spine", "spinal": "spine", "cervical": "neck",
    "vertebra": "spine", "vertebrae": "spine",
    # Eye
    "eye": "eye", "eyes": "eye", "cornea": "eye", "pupil": "eye",
    # Nose / Mouth / Throat
    "nose": "nose", "mouth": "mouth", "throat": "throat", "tongue": "throat",
    # Chest / Torso
    "chest": "chest", "rib": "chest", "ribs": "chest", "sternum": "chest",
    "lung": "chest", "lungs": "chest", "thorax": "chest", "torso": "torso",
    "abdomen": "abdomen", "stomach": "abdomen", "belly": "abdomen",
    "pelvis": "pelvis",
    # Back
    "back": "back", "lower back": "back",
    # Shoulder / Arm / Hand
    "shoulder": "shoulder", "clavicle": "shoulder",
    "arm": "arm", "arms": "arm", "elbow": "arm", "forearm": "arm",
    "wrist": "wrist", "hand": "hand", "hands": "hand",
    "finger": "finger", "fingers": "finger", "thumb": "finger",
    # Hip / Leg / Foot
    "hip": "hip", "thigh": "leg", "femur": "leg",
    "knee": "leg", "shin": "leg", "leg": "leg", "legs": "leg",
    "calf": "leg", "tibia": "leg", "fibula": "leg",
    "ankle": "ankle", "foot": "foot", "feet": "foot", "toe": "foot",
    "toes": "foot",
    # Skin / Surface
    "skin": "skin", "wound": "skin", "cut": "skin", "laceration": "skin",
    "burn": "skin",
}

# Upper extremity group — if query is about hand/finger/arm, penalise chest/leg chunks
_UPPER_EXTREMITY = {"shoulder", "arm", "wrist", "hand", "finger"}
_LOWER_EXTREMITY = {"hip", "leg", "ankle", "foot"}
_AXIAL = {"head", "neck", "spine", "chest", "abdomen", "pelvis", "back"}


# ── Emergency-type signals ────────────────────────────────────────────────────
# Maps category key → signal phrases (lower-case)

_EMERGENCY_SIGNALS: dict[str, list[str]] = {
    "cardiac_arrest": [
        "cardiac arrest", "no pulse", "not breathing", "chest compression",
        "cpr", "collapsed no pulse", "no pulse no breath",
    ],
    "choking": [
        "choking", "chok", "can't cough", "cant cough", "turning blue",
        "unable to cough", "foreign body airway",
    ],
    "seizure": [
        "seizure", "convuls", "shaking all over", "twitching", "fits",
        "epilep", "jerking",
    ],
    "stroke": [
        "stroke", "face drooping", "arm weakness", "slurred speech",
        "facial droop", "sudden numbness one side",
    ],
    "hemorrhage_control": [
        "bleeding", "blood", "wound", "laceration", "cut", "hemorrhage",
    ],
    "burns": [
        "burn", "burnt", "scalded", "boiling water", "hot water",
        "on fire", "caught fire",
    ],
    "fractures_ortho": [
        "broken", "fracture", "bone sticking out", "snap", "snapped",
        "dislocated", "sprain",
    ],
    "spinal_injury": [
        "spine", "spinal", "neck injury", "paralyz", "numb legs", "numb feet",
        "cant move", "can't move", "cannot feel",
    ],
    "anaphylaxis_allergy": [
        "allergic", "anaphylaxis", "epipen", "throat swelling", "hives",
        "swelling throat",
    ],
    "temperature_emergencies": [
        "hypothermia", "frostbite", "heat stroke", "heatstroke",
        "heat exhaustion", "frozen", "overheated",
    ],
    "poisoning_tox": [
        "poison", "swallowed", "ingested", "toxic", "overdose",
        "snake", "venom",
    ],
    "electric_shock": [
        "electric shock", "electrocuted", "live wire", "power line",
        "shocked by electricity",
    ],
    "drowning": [
        "drowning", "pulled from water", "found in water", "submersion",
    ],
    "respiratory": [
        "asthma", "inhaler", "wheez", "breathing difficulty",
        "shortness of breath",
    ],
}

# Severity signals
_CRITICAL_SIGNALS = [
    "cardiac arrest", "no pulse", "not breathing", "stopped breathing",
    "collapsed", "unconscious", "unresponsive", "seizure", "stroke",
    "electrocuted", "drowning",
]
_URGENT_SIGNALS = [
    "bleeding", "fracture", "broken", "bone sticking out", "choking",
    "anaphylaxis", "severe", "emergency", "help", "shock",
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class QueryClassification:
    emergency_type: str              # primary emergency category or "unknown"
    body_parts:     set[str]         = field(default_factory=set)
    severity:       str              = "unknown"   # "critical"|"urgent"|"moderate"|"unknown"
    confidence:     float            = 0.0

    @property
    def is_upper_extremity_only(self) -> bool:
        return bool(self.body_parts) and self.body_parts.issubset(_UPPER_EXTREMITY)

    @property
    def is_lower_extremity_only(self) -> bool:
        return bool(self.body_parts) and self.body_parts.issubset(_LOWER_EXTREMITY)

    @property
    def is_axial_only(self) -> bool:
        return bool(self.body_parts) and self.body_parts.issubset(_AXIAL)


# ── Classifier ────────────────────────────────────────────────────────────────

def classify_query(query: str) -> QueryClassification:
    """
    Classify *query* and return a QueryClassification.

    Parameters
    ----------
    query : raw user utterance

    Returns
    -------
    QueryClassification
    """
    q = query.lower()

    # ── 1. Detect body parts ──────────────────────────────────────────────
    body_parts: set[str] = set()
    for kw, canonical in _BODY_PART_MAP.items():
        if kw in q:
            body_parts.add(canonical)

    # ── 2. Detect emergency type ──────────────────────────────────────────
    type_scores: dict[str, int] = {}
    for etype, signals in _EMERGENCY_SIGNALS.items():
        hits = sum(1 for s in signals if s in q)
        if hits:
            type_scores[etype] = hits

    # Also use medical_keywords taxonomy
    for kw, category in detect_keywords(query):
        type_scores[category] = type_scores.get(category, 0) + 1

    if type_scores:
        best_type = max(type_scores, key=lambda k: type_scores[k])
        total_hits = sum(type_scores.values())
        best_hits  = type_scores[best_type]
        confidence = min(1.0, best_hits / max(total_hits, 1) + 0.2 * min(best_hits, 3))
    else:
        best_type  = "unknown"
        confidence = 0.0

    # ── 3. Detect severity ────────────────────────────────────────────────
    if any(s in q for s in _CRITICAL_SIGNALS):
        severity = "critical"
    elif any(s in q for s in _URGENT_SIGNALS):
        severity = "urgent"
    elif type_scores:
        severity = "moderate"
    else:
        severity = "unknown"

    return QueryClassification(
        emergency_type = best_type,
        body_parts     = body_parts,
        severity       = severity,
        confidence     = round(confidence, 3),
    )
