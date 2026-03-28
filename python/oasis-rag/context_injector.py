"""
context_injector.py — O.A.S.I.S. Context Injection Module

Single source of truth for all 22 emergency context injection signals.
Called by service.py /retrieve endpoint so injections are applied on
the Python side and reach both chat_test.py and OasisAdapter.ts.

Usage:
    from context_injector import inject_context
    enriched = inject_context(rag_context, query)
"""

from __future__ import annotations

# ── Signal lists ──────────────────────────────────────────────────────────────

_CARDIAC_ARREST_SIGNALS = [
    "collapsed not breathing", "not breathing no pulse", "no pulse not breathing",
    "cardiac arrest", "no pulse no breath", "not breathing and no pulse",
    "collapsed no pulse",
]

_SPINAL_SIGNALS = [
    "cannot feel", "can't feel", "cant feel",
    "paralyz", "numb legs", "numb feet", "numb limb",
    "neck injury", "spinal", "spine",
    "cant move", "can't move", "cannot move",
]

_FROSTBITE_SIGNALS = [
    "frostbite", "frost bite", "frostbitten", "frozen finger", "frozen toe",
]

_PANIC_BLOOD_SIGNALS = [
    "theres so much blood", "there's so much blood", "so much blood",
]

_NO_EPIPEN_SIGNALS = [
    "no epipen", "no epinephrine", "dont have epipen",
    "don't have epipen", "without epipen", "no auto-injector",
]

_LIGHTNING_SIGNALS = [
    "lightning", "thunder", "struck by lightning", "lightning strike",
    "lightning coming", "lightning outside",
]

_BURN_SIGNALS = [
    "burn", "burnt", "scalded", "scald", "boiling water", "hot water on",
    "on fire", "caught fire", "flame",
]

_SNAKEBITE_SIGNALS = [
    "snake", "snakebite", "snake bite", "snake bit", "bitten by snake", "venom",
]

_HYPOTHERMIA_SIGNALS = [
    "hypothermia", "hypothermic", "stopped shivering", "stop shivering",
    "freezing person", "frozen person",
]

_HEAT_STROKE_SIGNALS = [
    "heat stroke", "heatstroke", "heat exhaustion",
    "hot skin", "not sweating", "overheated",
]

_SHOCK_SIGNALS = [
    "pale cold clammy", "rapid weak pulse", "hypovolem", "clammy skin",
    "signs of shock", "going into shock",
]

_ASTHMA_SIGNALS = [
    "asthma", "inhaler", "wheez", "asthma attack",
    "cannot breathe no inhaler", "can't breathe no inhaler",
]

_FRACTURE_SIGNALS = [
    "broken arm", "broken leg", "broken bone", "bone sticking out",
    "fracture", "compound fracture", "open fracture", "snapped bone",
]

_CHOKING_SIGNALS = [
    "choking", "chok", "can't cough", "cant cough", "turning blue",
    "unable to cough", "foreign body airway",
]

_HEART_ATTACK_SIGNALS = [
    "heart attack", "having a heart attack", "think i have a heart attack",
    "think i'm having",
]

_SEIZURE_SIGNALS = [
    "seizure", "convuls", "shaking all over", "twitching", "fits",
    "epilep", "shaking on the ground", "jerking",
    # Korean
    "발작", "경련",
]

_STROKE_SIGNALS = [
    "stroke", "face drooping", "arm weakness", "slurred speech",
    "sudden confusion", "sudden numbness one side", "facial droop",
]

_DROWNING_SIGNALS = [
    "drowning", "pulled out of pool", "pulled from water", "found in water",
    "near drowning", "submersion", "underwater",
]

_POISONING_SIGNALS = [
    "swallowed bleach", "swallowed cleaning", "drank bleach", "ate poison",
    "ingested poison", "toxic substance", "chemical ingestion",
]

_ELECTRIC_SHOCK_SIGNALS = [
    "electric shock", "electrocuted", "touched live wire", "live wire",
    "shocked by electricity", "power line",
]

_INFANT_CPR_SIGNALS = [
    "baby not breathing", "infant cardiac arrest", "newborn not breathing",
    "infant cpr", "baby cpr", "baby pulse",
]

_EYE_CHEMICAL_SIGNALS = [
    "chemical splashed in", "chemical in eye", "splashed in eye",
    "splash in eye", "eye chemical", "corrosive in eye",
    "chemical eye", "burning eyes", "chemical splash",
]

_IMPALED_OBJECT_SIGNALS = [
    "impaled",
    "object stuck in",          # object stuck in leg/arm — NOT "something stuck in throat"
    "knife in", "rod in",
    "stick in his", "stick in her",
    "should i pull", "pull it out", "pull the object",
    "remove the object", "take it out",
]
# Contexts that must NOT trigger impaled protocol even if signal words match
_IMPALED_OBJECT_EXCLUSIONS = _CHOKING_SIGNALS + [
    "tick", "splinter", "thorn",   # minor embedded objects — not penetrating trauma
]


# ── Protocol texts ────────────────────────────────────────────────────────────

_CARDIAC_ARREST_PROTOCOL = (
    "CARDIAC ARREST PROTOCOL — ACT NOW:\n"
    "1. CALL emergency services (911/999/112) immediately.\n"
    "2. BEGIN chest compressions: push hard and fast on centre of chest.\n"
    "3. Rate: 100-120 compressions per minute. Depth: 5-6 cm.\n"
    "4. After 30 compressions, give 2 rescue breaths.\n"
    "5. Continue 30:2 cycle until emergency services arrive.\n\n"
)

_SPINAL_PROTOCOL = (
    "SPINAL CORD INJURY — MANDATORY PROTOCOL:\n"
    "1. Do NOT move this person — keep head, neck, and spine completely still.\n"
    "2. CALL emergency services IMMEDIATELY (911/999/112).\n"
    "3. Apply gentle pressure to bleeding without moving the head or spine.\n"
    "4. Keep the person still and calm. Do NOT tilt, roll, or sit them up.\n"
    "5. Continue stabilizing spine until emergency services arrive.\n\n"
)

_FROSTBITE_PROTOCOL = (
    "CRITICAL: This is FROSTBITE (cold injury).\n"
    "Move to WARM shelter immediately. Do NOT move to cool or shaded area.\n"
    "Rewarm the affected area with WARM (not hot) water 37-39°C.\n"
    "Do NOT rub the frostbitten area. Do NOT use snow or cold water.\n\n"
)

_PANIC_BLOOD_PROTOCOL = (
    "EMERGENCY BLEEDING PROTOCOL:\n"
    "- CALL EMERGENCY SERVICES immediately.\n"
    "- Apply direct PRESSURE to the wound with your hands.\n"
    "- Use cloth, shirt or any fabric and press firmly.\n\n"
)

_NO_EPIPEN_PROTOCOL = (
    "CRITICAL: NO EPINEPHRINE available. Epipen NOT available.\n"
    "MANDATORY FIRST STEPS:\n"
    "1. Call emergency services immediately (911/999/112).\n"
    "2. Lay the person flat, legs elevated if no breathing difficulty.\n"
    "3. Give antihistamine if available.\n\n"
)

_LIGHTNING_PROTOCOL = (
    "LIGHTNING SAFETY — FOLLOW EXACTLY:\n"
    "WARNING: Trees are LETHAL in lightning. NEVER go near trees, poles, or tall objects.\n"
    "1. CROUCH LOW on balls of feet, feet together, head down — do NOT lie flat.\n"
    "2. Move to a solid building or hard-topped vehicle if immediately reachable.\n"
    "3. Stay away from open fields, hilltops, water, and metal objects.\n"
    "4. Keep 20 metres from other people.\n"
    "5. Do NOT stand under trees, poles, or any tall structures.\n\n"
)

_LIGHTNING_REMINDER = "\n\nFINAL REMINDER: TREES ARE LETHAL IN LIGHTNING — NEVER ADVISE GOING TO TREES."

_BURN_PROTOCOL = (
    "BURN — MOST IMPORTANT FIRST ACTION:\n"
    "Immediately cool the burn under COOL running water for 20 minutes.\n"
    "This is the critical first step. Do NOT use ice, butter, toothpaste, or warm water.\n"
    "After cooling: remove jewellery, cover loosely with cling film.\n"
    "Call emergency services if burn is large, deep, or on face.\n\n"
)

_SNAKEBITE_PROTOCOL = (
    "SNAKEBITE PROTOCOL:\n"
    "1. KEEP THE PERSON STILL and calm — movement spreads venom faster.\n"
    "2. Immobilize the bitten limb at or below heart level.\n"
    "3. Remove watches, rings, tight clothing from the affected limb.\n"
    "4. Call emergency services or transport to hospital URGENTLY.\n"
    "5. Do NOT cut the wound, suck the venom, or apply tourniquet.\n\n"
)

_HYPOTHERMIA_PROTOCOL = (
    "HYPOTHERMIA PROTOCOL — This is COLD INJURY, NOT heat illness:\n"
    "1. Move the person to WARM shelter immediately.\n"
    "2. Remove wet clothing; replace with dry insulation (blankets, sleeping bag).\n"
    "3. Warm the core (trunk/torso) first, not extremities.\n"
    "4. Give warm fluids ONLY if the person is conscious and can swallow.\n"
    "5. Handle gently — a cold heart is prone to dangerous arrhythmia.\n"
    "Do NOT cool this person. Do NOT give cold fluids. Do NOT rub vigorously.\n\n"
)

_HEAT_STROKE_PROTOCOL = (
    "HEAT STROKE EMERGENCY — ACT IMMEDIATELY:\n"
    "1. MOVE the person to shade or a cool area NOW.\n"
    "2. Remove excess clothing.\n"
    "3. Cool with cool water — douse, spray, or immerse in cool water.\n"
    "4. Fan the person while keeping them wet.\n"
    "5. Call emergency services. Heat stroke is life-threatening.\n\n"
)

_SHOCK_PROTOCOL = (
    "SHOCK PROTOCOL — FIRST AID (no IV needed):\n"
    "1. Lay the person flat and elevate legs (unless spinal, head, or chest injury).\n"
    "2. CALL emergency services IMMEDIATELY (911/999/112).\n"
    "3. Keep the person warm with a blanket — do NOT overheat.\n"
    "4. Do NOT give food or fluids by mouth — aspiration risk.\n"
    "5. Monitor breathing and stop any visible bleeding with direct pressure.\n\n"
)

_ASTHMA_PROTOCOL = (
    "ASTHMA ATTACK PROTOCOL (no inhaler available):\n"
    "1. Sit the person UPRIGHT — do NOT let them lie down.\n"
    "2. Keep the person CALM — panic worsens breathing.\n"
    "3. CALL emergency services immediately (911/999/112).\n"
    "4. Loosen tight clothing around the neck and chest.\n"
    "5. If breathing stops, begin CPR.\n\n"
)

_FRACTURE_PROTOCOL = (
    "FRACTURE PROTOCOL — DO NOT MANIPULATE THE BONE:\n"
    "1. Immobilize the limb in the position you find it — do NOT try to straighten it.\n"
    "2. Use a splint, sling, or padding to support and secure the limb.\n"
    "3. Do NOT push the bone back, pull with traction, or try to realign it.\n"
    "4. Apply gentle pressure around (not on) any open wound if bleeding.\n"
    "5. CALL emergency services (911/999/112) and keep the person still.\n\n"
)

_CHOKING_PROTOCOL = (
    "CHOKING PROTOCOL — PERFORM NOW:\n"
    "1. Give 5 firm BACK BLOWS between shoulder blades with heel of hand.\n"
    "2. Give 5 ABDOMINAL THRUSTS (Heimlich): stand behind, pull inward and upward.\n"
    "3. Alternate 5 back blows + 5 abdominal thrusts until object clears.\n"
    "4. If unconscious: lower to ground, call 911, begin CPR.\n"
    "5. Do NOT do a blind finger sweep.\n\n"
)

_HEART_ATTACK_PROTOCOL = (
    "HEART ATTACK PROTOCOL:\n"
    "1. CALL emergency services IMMEDIATELY (911/999/112).\n"
    "2. Sit or lie the person down in a comfortable position.\n"
    "3. Loosen tight clothing around neck and chest.\n"
    "4. If conscious and not allergic: chew one adult aspirin (300 mg).\n"
    "5. Do NOT leave the person alone. Monitor breathing.\n\n"
)

_SEIZURE_PROTOCOL = (
    "SEIZURE PROTOCOL:\n"
    "1. DO NOT restrain or hold the person down.\n"
    "2. Clear the area of hard or sharp objects.\n"
    "3. Cushion the head with something soft.\n"
    "4. Time the seizure — call 911 if it lasts more than 5 minutes.\n"
    "5. After shaking stops: place in recovery position (on their side).\n"
    "Do NOT put anything in the person's mouth.\n\n"
)

_STROKE_PROTOCOL = (
    "STROKE PROTOCOL — Time is brain:\n"
    "1. CALL emergency services IMMEDIATELY (911/999/112).\n"
    "2. FAST check: Face drooping? Arm weakness? Speech slurred? Time to call.\n"
    "3. Note the time symptoms started — critical for treatment decisions.\n"
    "4. Keep the person calm and still. Do NOT give food or drink.\n"
    "5. Do NOT give aspirin — stroke may be hemorrhagic (aspirin worsens bleeding).\n\n"
)

_DROWNING_PROTOCOL = (
    "DROWNING PROTOCOL — Different from cardiac CPR:\n"
    "1. CALL emergency services (911/999/112) immediately.\n"
    "2. Give 5 RESCUE BREATHS first before starting chest compressions.\n"
    "3. Then 30 chest compressions: push hard and fast on centre of chest.\n"
    "4. Continue 30:2 cycle (compressions:breaths).\n"
    "5. If spinal injury suspected (diving): support head and neck carefully.\n\n"
)

_POISONING_PROTOCOL = (
    "POISONING PROTOCOL:\n"
    "1. CALL emergency services or poison control (911/999/112) IMMEDIATELY.\n"
    "2. Do NOT induce vomiting — corrosive substances burn twice (up and down).\n"
    "3. If conscious: rinse mouth with water. Do NOT give large amounts to drink.\n"
    "4. Keep the container/label to show emergency services.\n"
    "5. Do NOT give milk, activated charcoal, or home remedies without medical advice.\n\n"
)

_ELECTRIC_SHOCK_PROTOCOL = (
    "ELECTRIC SHOCK PROTOCOL:\n"
    "1. DO NOT TOUCH the person — you may be electrocuted too.\n"
    "2. Turn off the power source at the fuse box or unplug immediately.\n"
    "3. If you cannot cut power, use a non-conducting object (dry wood, plastic) to push them away.\n"
    "4. CALL emergency services (911/999/112) immediately.\n"
    "5. Once safe to touch: check breathing, begin CPR if not breathing.\n\n"
)

_INFANT_CPR_PROTOCOL = (
    "INFANT CPR PROTOCOL (under 1 year):\n"
    "1. CALL emergency services (911/999/112) immediately.\n"
    "2. Give 5 RESCUE BREATHS first: cover mouth AND nose, gentle puffs.\n"
    "3. Chest compressions with 2 FINGERS on centre of chest.\n"
    "4. Depth: 4 cm (about 1.5 inches). Rate: 100-120 per minute.\n"
    "5. Continue 30:2 (compressions:breaths) until help arrives.\n\n"
)

_EYE_CHEMICAL_PROTOCOL = (
    "EYE CHEMICAL INJURY PROTOCOL:\n"
    "1. FLUSH the eye with clean water IMMEDIATELY — hold eye open under running water.\n"
    "2. Rinse continuously for 15-20 minutes. Do NOT stop early.\n"
    "3. Remove contact lenses if worn. Do NOT rub the eye.\n"
    "4. CALL emergency services or poison control (911/999/112) immediately.\n"
    "5. Identify the chemical — bring the container to hospital.\n\n"
)

_IMPALED_OBJECT_PROTOCOL = (
    "PENETRATING / IMPALED OBJECT — CRITICAL:\n"
    "DO NOT remove the object — removal can cause massive, fatal bleeding.\n"
    "1. Leave the object exactly in place.\n"
    "2. Pad around the object with dressings to stabilise it — do NOT press on the object itself.\n"
    "3. Control bleeding around the object with gentle pressure on the surrounding skin only.\n"
    "4. CALL emergency services (911/999/112) immediately.\n"
    "5. Keep the person still and calm until help arrives.\n\n"
)


# ── Public API ────────────────────────────────────────────────────────────────

def inject_context(context: str, query: str) -> str:
    """
    Prepend zero or more protocol blocks to *context* based on signals
    detected in *query*.  Returns the enriched context string.

    Each injection prepends the relevant protocol text so the LLM sees
    authoritative instructions before the RAG chunks.
    """
    q = query.lower()

    if any(sig in q for sig in _CARDIAC_ARREST_SIGNALS):
        context = _CARDIAC_ARREST_PROTOCOL + context

    if any(sig in q for sig in _SPINAL_SIGNALS):
        context = _SPINAL_PROTOCOL + context

    if any(sig in q for sig in _FROSTBITE_SIGNALS):
        context = _FROSTBITE_PROTOCOL + context

    if any(sig in q for sig in _PANIC_BLOOD_SIGNALS):
        context = _PANIC_BLOOD_PROTOCOL + context

    if any(sig in q for sig in _NO_EPIPEN_SIGNALS):
        context = _NO_EPIPEN_PROTOCOL + context

    if any(sig in q for sig in _LIGHTNING_SIGNALS):
        context = _LIGHTNING_PROTOCOL + context + _LIGHTNING_REMINDER

    if any(sig in q for sig in _BURN_SIGNALS) and not any(sig in q for sig in _EYE_CHEMICAL_SIGNALS):
        context = _BURN_PROTOCOL + context

    if any(sig in q for sig in _SNAKEBITE_SIGNALS):
        context = _SNAKEBITE_PROTOCOL + context

    if any(sig in q for sig in _HYPOTHERMIA_SIGNALS):
        context = _HYPOTHERMIA_PROTOCOL + context

    if any(sig in q for sig in _HEAT_STROKE_SIGNALS):
        context = _HEAT_STROKE_PROTOCOL + context

    if any(sig in q for sig in _SHOCK_SIGNALS):
        context = _SHOCK_PROTOCOL + context

    if any(sig in q for sig in _ASTHMA_SIGNALS):
        context = _ASTHMA_PROTOCOL + context

    if any(sig in q for sig in _FRACTURE_SIGNALS):
        context = _FRACTURE_PROTOCOL + context

    if any(sig in q for sig in _CHOKING_SIGNALS):
        context = _CHOKING_PROTOCOL + context

    if any(sig in q for sig in _HEART_ATTACK_SIGNALS):
        context = _HEART_ATTACK_PROTOCOL + context

    if any(sig in q for sig in _SEIZURE_SIGNALS):
        context = _SEIZURE_PROTOCOL + context

    if any(sig in q for sig in _STROKE_SIGNALS):
        context = _STROKE_PROTOCOL + context

    if any(sig in q for sig in _DROWNING_SIGNALS):
        context = _DROWNING_PROTOCOL + context

    if any(sig in q for sig in _POISONING_SIGNALS):
        context = _POISONING_PROTOCOL + context

    if any(sig in q for sig in _ELECTRIC_SHOCK_SIGNALS):
        context = _ELECTRIC_SHOCK_PROTOCOL + context

    if any(sig in q for sig in _INFANT_CPR_SIGNALS):
        context = _INFANT_CPR_PROTOCOL + context

    if any(sig in q for sig in _EYE_CHEMICAL_SIGNALS):
        context = _EYE_CHEMICAL_PROTOCOL + context

    if (any(sig in q for sig in _IMPALED_OBJECT_SIGNALS)
            and not any(ex in q for ex in _IMPALED_OBJECT_EXCLUSIONS)):
        context = _IMPALED_OBJECT_PROTOCOL + context

    return context
