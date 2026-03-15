"""Split redcross_wilderness.md into topic-specific knowledge files."""
import os, sys, re

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = "D:/GitHub/whisplay-ai-chatbot-1/data/knowledge/redcross_wilderness.md"
OUT = "D:/GitHub/whisplay-ai-chatbot-1/data/knowledge"

META_BASE = (
    "**Source:** American Red Cross — Wilderness and Remote First Aid Emergency Reference Guide\n"
    "**Standard:** American Red Cross / SOLO Wilderness Medicine\n"
    "**Applies to:** All ages; field settings without rapid EMS access\n"
)

with open(SRC, encoding="utf-8") as f:
    lines = f.readlines()

total = len(lines)
print(f"Source: {total} lines")

# ── Section line ranges (1-indexed, inclusive) ─────────────────────────────
# Determined by section heading analysis of the 3879-line file.
SECTIONS = {
    "redcross_altitude": {
        "ranges": [(938, 1057)],
        "title": "# Red Cross Wilderness First Aid — Altitude Illnesses",
        "tags": "[DOMAIN_TAGS: altitude, altitude_sickness, acute_mountain_sickness, AMS, HAPE, HACE, high_altitude, acclimatization, descent, headache, nausea, ataxia, mountain, wilderness]",
    },
    "redcross_bone_joint": {
        "ranges": [(1058, 1553)],
        "title": "# Red Cross Wilderness First Aid — Bone and Joint Injuries",
        "tags": "[DOMAIN_TAGS: fracture, broken_bone, dislocation, sprain, strain, splint, immobilization, sling, bone_joint, open_fracture, compound_fracture, crepitus, neurovascular, wilderness, orthopedic]",
    },
    "redcross_burns": {
        "ranges": [(1443, 1553)],
        "title": "# Red Cross Wilderness First Aid — Burns",
        "tags": "[DOMAIN_TAGS: burn, thermal_burn, chemical_burn, scalding, boiling_water, blister, cool_water, second_degree_burn, third_degree_burn, burn_dressing, wilderness, first_aid_burns]",
    },
    "redcross_lightning": {
        "ranges": [(2252, 2321)],
        "title": "# Red Cross Wilderness First Aid — Lightning Injuries",
        "tags": "[DOMAIN_TAGS: lightning, lightning_strike, lightning_storm, thunder, 30_30_rule, cardiac_arrest, CPR, lightning_injuries, wilderness, shelter, storm_safety]",
    },
    "redcross_cold_emergencies": {
        "ranges": [(2026, 2251), (3230, 3351)],
        "title": "# Red Cross Wilderness First Aid — Cold-Related Emergencies",
        "tags": "[DOMAIN_TAGS: hypothermia, frostbite, cold_exposure, shivering, stopped_shivering, hypothermia_wrap, rewarming, immersion_foot, trench_foot, cold_injury, wilderness, core_temperature]",
    },
    "redcross_heat_emergencies": {
        "ranges": [(2026, 2150)],
        "title": "# Red Cross Wilderness First Aid — Heat-Related Emergencies",
        "tags": "[DOMAIN_TAGS: heat_stroke, heat_exhaustion, heat_cramps, hyperthermia, hyponatremia, dehydration, hot_skin, sweating, cooling, wilderness, environmental_heat]",
    },
    "redcross_submersion": {
        "ranges": [(2422, 2610)],
        "title": "# Red Cross Wilderness First Aid — Submersion Incidents (Drowning)",
        "tags": "[DOMAIN_TAGS: drowning, submersion, near_drowning, water_rescue, reach_throw_row_go, CPR, rescue_breathing, swimmer_distress, wilderness, aquatic_emergency]",
    },
    "redcross_wounds": {
        "ranges": [(2611, 2966)],
        "title": "# Red Cross Wilderness First Aid — Wounds and Wound Infection",
        "tags": "[DOMAIN_TAGS: wound, wound_care, wound_infection, wound_cleaning, irrigation, impaled_object, wound_closure, blister, wound_dressing, infection_signs, wilderness, laceration]",
    },
    "redcross_bites_stings": {
        "ranges": [(803, 937), (2967, 3083), (3672, 3851)],
        "title": "# Red Cross Wilderness First Aid — Bites, Stings, and Allergic Reactions",
        "tags": "[DOMAIN_TAGS: snakebite, venomous_snake, bee_sting, wasp_sting, tick_bite, tick_removal, spider_bite, scorpion_sting, jellyfish_sting, marine_life, anaphylaxis, allergic_reaction, epinephrine, wilderness, envenomation, antivenin]",
    },
    "redcross_special": {
        "ranges": [(553, 727), (3084, 3229), (3352, 3671), (3852, 3879)],
        "title": "# Red Cross Wilderness First Aid — Special Situations",
        "tags": "[DOMAIN_TAGS: diabetic_emergency, hypoglycemia, insulin, blood_sugar, asthma_attack, seizure, stroke, abdominal_injury, eye_injury, poisonous_plants, confined_space, childbirth, CPR, check_call_care, SAMPLE_history, evacuation, wilderness]",
    },
}


def get_lines(ranges):
    """Extract lines from 1-indexed inclusive ranges, deduplicating."""
    seen = set()
    out = []
    for start, end in ranges:
        for i in range(start - 1, min(end, total)):
            if i not in seen:
                seen.add(i)
                out.append(lines[i])
    return out


def build_file(key, cfg):
    body_lines = get_lines(cfg["ranges"])
    body = "".join(body_lines).strip()
    content = (
        cfg["title"] + "\n\n"
        + META_BASE + "\n"
        + cfg["tags"] + "\n\n"
        + "---\n\n"
        + body + "\n"
    )
    path = os.path.join(OUT, key + ".md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    lc = content.count("\n")
    kb = len(content.encode("utf-8")) / 1024
    print(f"  {key}.md  ({lc} lines, {kb:.1f} KB)")
    return path


print("\nCreating split files:")
for key, cfg in SECTIONS.items():
    build_file(key, cfg)

# Remove the original monolithic file
os.remove(SRC)
print(f"\nRemoved: {SRC}")
print("Done.")
