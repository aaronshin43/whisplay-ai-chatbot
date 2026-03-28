"""
categories.py — 32 medical category definitions + out_of_domain cluster.

Each entry: (id, human_name, description, kb_source, priority_level)
priority_level: "critical" | "urgent" | "standard"
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    description: str
    kb_source: str
    priority: Literal["critical", "urgent", "standard", "ood"]


# ---------------------------------------------------------------------------
# All 32 medical categories + out_of_domain
# ---------------------------------------------------------------------------

CATEGORIES: list[Category] = [
    # Life-threatening emergencies (12)
    Category("cpr",              "CPR / Cardiac Arrest",              "Cardiac arrest, CPR protocol",                          "cpr.md",                    "critical"),
    Category("choking",          "Choking / Airway Obstruction",      "Airway obstruction, Heimlich maneuver",                  "airway.md",                 "critical"),
    Category("bleeding",         "Severe Bleeding / Hemorrhage",      "Severe bleeding, hemorrhage control, tourniquet",        "wounds_and_bleeding.md",    "critical"),
    Category("anaphylaxis",      "Anaphylaxis / Severe Allergy",      "Severe allergic reaction, epinephrine, anaphylactic shock","bites_and_stings.md",     "urgent"),
    Category("stroke",           "Stroke",                            "Stroke, FAST assessment, facial droop",                 "stroke.md",                 "urgent"),
    Category("heart_attack",     "Heart Attack",                      "Heart attack / ACS, conscious chest pain",              "chest_pain_cardiac.md",     "urgent"),
    Category("drowning",         "Drowning / Near-Drowning",          "Submersion injury, water rescue, near-drowning",        "submersion.md",             "urgent"),
    Category("electric_shock",   "Electric Shock / Electrocution",    "Electrocution, live wire contact, lightning strike",    "electric_shock.md",         "urgent"),
    Category("poisoning",        "Poisoning / Toxic Ingestion",       "Poisoning, toxic substance ingestion",                  "poisoning_overdose.md",     "urgent"),
    Category("opioid_overdose",  "Opioid Overdose",                   "Opioid overdose, naloxone (Narcan) protocol",           "poisoning_overdose.md",     "standard"),
    Category("chest_wound",      "Chest Wound / Pneumothorax",        "Penetrating chest wound, sucking chest wound, pneumothorax","breathing.md",          "standard"),
    Category("spinal_injury",    "Spinal Injury",                     "Spinal / neck trauma, do not move patient",             "trauma.md",                 "standard"),

    # Trauma and injury (8)
    Category("fracture",         "Fracture / Broken Bone",            "Broken bones, splinting, immobilization",               "bone_and_joint.md",         "standard"),
    Category("sprain_dislocation","Sprain / Dislocation",             "Sprains, strains, dislocations, RICE method",           "bone_and_joint.md",         "standard"),
    Category("head_injury",      "Head Injury / Concussion",          "Concussion, head trauma, TBI",                          "trauma.md",                 "standard"),
    Category("impaled_object",   "Impaled Object",                    "Penetrating foreign body, do not remove",               "trauma.md",                 "standard"),
    Category("abdominal_injury", "Abdominal Injury",                  "Abdominal trauma, evisceration, blunt injury",          "special_situations.md",     "standard"),
    Category("burns",            "Burns",                             "Thermal and chemical burns, degrees of burn",           "burns.md",                  "standard"),
    Category("eye_injury",       "Eye Injury",                        "Chemical eye splash, foreign body in eye, impaled eye", "special_situations.md",     "standard"),
    Category("dental_injury",    "Dental Injury",                     "Knocked-out tooth, mouth injury",                       "special_situations.md",     "standard"),

    # Environmental emergencies (5)
    Category("hypothermia",      "Hypothermia",                       "Cold exposure, hypothermia, core temperature drop",     "cold_emergencies.md",       "standard"),
    Category("frostbite",        "Frostbite",                         "Frostbite, cold injury to extremities",                 "cold_emergencies.md",       "standard"),
    Category("heat_stroke",      "Heat Stroke / Heat Exhaustion",     "Heat stroke, heat exhaustion, hyperthermia",            "heat_emergencies.md",       "standard"),
    Category("lightning",        "Lightning Strike",                  "Lightning injury, lightning safety",                    "lightning.md",              "standard"),
    Category("altitude_sickness","Altitude Sickness",                 "AMS, HACE, HAPE, high altitude illness",               "altitude.md",               "standard"),

    # Medical emergencies (5)
    Category("seizure",          "Seizure / Epilepsy",                "Seizure, epilepsy, convulsions, tonic-clonic",          "seizure_epilepsy.md",       "standard"),
    Category("diabetic",         "Diabetic Emergency",                "Diabetic emergency, hypoglycemia, hyperglycemia",       "diabetic_emergency.md",     "standard"),
    Category("asthma",           "Asthma Attack",                     "Asthma attack, bronchospasm, no inhaler",              "breathing.md",              "standard"),
    Category("shock",            "Shock",                             "Hypovolemic / circulatory shock",                       "shock.md",                  "standard"),
    Category("fainting",         "Fainting / Syncope",                "Syncope, vasovagal episode, loss of consciousness",    "mental_status.md",          "standard"),

    # Other (2)
    Category("bites_and_stings", "Bites and Stings",                  "Snake, spider, scorpion, bee, tick, animal bites",     "bites_and_stings.md",       "standard"),
    Category("pediatric_emergency","Pediatric Emergency",             "Infant and child CPR, febrile seizure, child choking",  "pediatric_emergency.md",    "standard"),

    # OOD cluster — must always exist
    Category("out_of_domain",    "Out of Domain",                     "Non-medical queries, casual chat, unrelated topics",    "",                          "ood"),
]

# ---------------------------------------------------------------------------
# Convenience lookups
# ---------------------------------------------------------------------------

CATEGORY_IDS: list[str] = [c.id for c in CATEGORIES]
CATEGORY_BY_ID: dict[str, Category] = {c.id: c for c in CATEGORIES}

# Ordered index for centroid array alignment (must match build_centroids.py output order)
CATEGORY_INDEX: dict[str, int] = {c.id: i for i, c in enumerate(CATEGORIES)}
