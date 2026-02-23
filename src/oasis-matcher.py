"""
O.A.S.I.S. Protocol Matcher
- Embeds user input and finds the closest protocol using FAISS
- Uses all-MiniLM-L6-v2 (80MB, ~200MB RAM)
- Returns matched protocol text or triage questions
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# 1. PROTOCOL DATABASE
#    Each protocol has:
#    - text: injected into system prompt when matched
#    - scenarios: representative user phrases for indexing
# ─────────────────────────────────────────────

PROTOCOLS = [
    {
        "id": "BLEEDING",
        "text": "Apply firm continuous pressure with cloth or hand. Do not remove cloth even if soaked through — add more on top. Keep pressure constant for at least 10 minutes without checking. Elevate the limb above heart level if possible.",
        "scenarios": [
            "there is so much blood coming out",
            "blood wont stop",
            "i wrapped it but its soaking through",
            "he has a deep cut and bleeding badly",
            "wound is bleeding a lot",
            "blood everywhere i dont know what to do",
            "she cut her hand it wont stop bleeding",
        ],
    },
    {
        "id": "TOURNIQUET",
        "text": "Apply tourniquet 2-3 inches above the wound, not on a joint. Tighten until bleeding stops completely. Write the time of application on the skin. Do not remove or loosen once applied.",
        "scenarios": [
            "limb is spurting blood cant stop it",
            "artery bleeding pressure isnt working",
            "leg is gushing blood nothing is working",
            "massive bleeding from arm wont stop with pressure",
            "how do i use a tourniquet",
        ],
    },
    {
        "id": "SNAKEBITE",
        "text": "Do not suck out venom. Do not cut or squeeze the bite. Immobilize the limb and keep it below heart level. Remove rings or tight clothing near the bite. Keep the person still and calm.",
        "scenarios": [
            "snake bit him on the ankle",
            "bitten by a snake what do i do",
            "do we suck out the venom",
            "snake bite twenty minutes ago",
            "he was bitten by a snake in the woods",
            "snake got her leg should i cut it",
        ],
    },
    {
        "id": "ANAPHYLAXIS",
        "text": "Use epinephrine (EpiPen) immediately — inject into outer thigh, hold for 10 seconds. If no EpiPen, lay the person flat with legs elevated unless breathing is difficult, then sit them up. Do not give anything by mouth. A second dose may be needed in 5-15 minutes.",
        "scenarios": [
            "she cant breathe after eating peanuts",
            "his face is swelling after bee sting",
            "throat is closing after eating",
            "allergic reaction she cant breathe",
            "how do i use an epipen",
            "he ate something and now his throat is swelling",
            "severe allergic reaction face is swelling",
        ],
    },
    {
        "id": "CHOKING_ADULT",
        "text": "If the person can cough, let them cough. If they cannot breathe or cough: give 5 firm back blows between shoulder blades, then 5 abdominal thrusts (Heimlich). Alternate until object is expelled or person loses consciousness.",
        "scenarios": [
            "she was eating and now shes grabbing her throat",
            "something stuck in his throat cant breathe",
            "he swallowed something and is choking",
            "turning blue cant get air",
            "food stuck in throat",
            "i did the heimlich but it didnt come out",
            "adult choking on food",
        ],
    },
    {
        "id": "CHOKING_INFANT",
        "text": "Never do abdominal thrusts on infants under 1 year. Give 5 back blows face-down on your forearm, then 5 chest thrusts face-up. Do not do blind finger sweeps in the mouth.",
        "scenarios": [
            "baby is choking she is only 8 months",
            "infant choking what do i do",
            "my baby swallowed something cant breathe",
            "newborn choking",
            "baby turning blue swallowed something",
            "6 month old choking",
        ],
    },
    {
        "id": "CPR",
        "text": "Place heel of hand on center of chest. Push down 2 inches at 100-120 times per minute. After 30 compressions, give 2 rescue breaths. Continue until breathing resumes or help arrives.",
        "scenarios": [
            "he collapsed and is not breathing",
            "i dont know how to do cpr walk me through it",
            "she has no pulse",
            "he is unconscious and not responding",
            "how do i do chest compressions",
            "heart stopped what do i do",
            "not breathing after falling",
        ],
    },
    {
        "id": "HYPOTHERMIA_SEVERE",
        "text": "If shivering has STOPPED, this is severe hypothermia — life-threatening. Do not rub or massage the skin. Do not apply direct heat to limbs. Warm the core only: chest, neck, armpits, groin. Move gently — sudden movement can trigger cardiac arrest.",
        "scenarios": [
            "he stopped shivering after being out in the cold",
            "we were lost overnight and he is not shivering anymore",
            "shivering stopped is that bad",
            "found someone in the snow not moving",
            "severe hypothermia what do i do",
            "he was shaking and now he stopped",
        ],
    },
    {
        "id": "HYPOTHERMIA_PREVENTION",
        "text": "Remove wet clothing immediately — wet fabric loses heat 25x faster than dry. Insulate from the ground first before worrying about wind. Eat and drink if possible — body needs fuel to generate heat.",
        "scenarios": [
            "he fell in the river and is soaking wet and cold",
            "she is wet and shivering",
            "soaked from rain and getting cold",
            "wet clothes in cold weather what do i do",
            "how to treat someone who is cold and wet",
        ],
    },
    {
        "id": "SPINAL_INJURY",
        "text": "Do not move the person unless there is immediate life threat. Stabilize the head and neck in the position found. If movement is absolutely necessary, keep head, neck, and spine aligned as one unit.",
        "scenarios": [
            "she fell off the roof and cant feel her legs",
            "he cant move his neck should i move him",
            "fell from height and has neck pain",
            "possible spinal injury what do i do",
            "she cant feel her legs after the fall",
            "he was in a car crash and neck hurts",
        ],
    },
    {
        "id": "SEIZURE",
        "text": "Do not put anything in the mouth. Do not restrain the person. Clear the area of hard objects. Turn them on their side after convulsions stop. Time the seizure — if over 5 minutes, treat as emergency.",
        "scenarios": [
            "he is having a seizure",
            "she is convulsing what do i do",
            "epileptic fit happening right now",
            "shaking uncontrollably on the ground foaming",
            "seizure how long is too long",
            "do i put something in his mouth during seizure",
        ],
    },
    {
        "id": "DIABETIC_EMERGENCY",
        "text": "If conscious and able to swallow: give sugar (juice, candy, glucose gel) immediately. If unconscious: do not give anything by mouth. If unsure: giving sugar to a high blood sugar patient is less dangerous than withholding it from a low blood sugar patient.",
        "scenarios": [
            "diabetic acting really weird and sweating",
            "he is diabetic and confused",
            "she has diabetes and passed out",
            "diabetic emergency low blood sugar",
            "insulin reaction what do i do",
            "diabetic person unconscious",
        ],
    },
    {
        "id": "DROWNING",
        "text": "Start CPR immediately — do not waste time trying to drain water from the lungs. Assume possible spinal injury if the person dove in or fell from height. Keep them warm after rescue — drowning victims lose heat rapidly.",
        "scenarios": [
            "we pulled him out of the water not breathing",
            "she was underwater how long does it matter",
            "drowning victim not responding",
            "fell in lake and not breathing",
            "pulled from pool unconscious",
            "near drowning what do i do",
        ],
    },
    {
        "id": "BURN",
        "text": "Do not apply butter, oil, toothpaste, or ice. Cool with running water for 20 minutes. Do not pop blisters. Cover loosely with clean non-stick material.",
        "scenarios": [
            "spilled boiling water on my arm",
            "she touched the stove and burned her hand",
            "fire burn on his leg",
            "do i put butter on a burn",
            "chemical burn on skin",
            "he touched the engine and hand looks bad",
            "burn blister should i pop it",
        ],
    },
    {
        "id": "HEAD_INJURY",
        "text": "Do not let the person fall asleep for at least 2 hours after a significant head impact. Watch for: unequal pupils, repeated vomiting, worsening headache, confusion, one-sided weakness. Any of these signs require emergency care immediately.",
        "scenarios": [
            "she hit her head and went to sleep is that okay",
            "he fell and hit his head hard",
            "knocked out briefly after hitting head",
            "concussion what do i watch for",
            "head injury can she sleep",
            "hit head on concrete now confused",
        ],
    },
    {
        "id": "CHEST_PAIN",
        "text": "Give one adult aspirin (325mg) to chew — not swallow whole — if conscious and not allergic. Keep the person still and calm. Loosen tight clothing. Do not give food or water.",
        "scenarios": [
            "he is clutching his chest in pain",
            "she has pain in her left arm and chest",
            "i think he is having a heart attack",
            "chest pressure and sweating",
            "heart attack symptoms what do i do",
            "sudden chest pain what should i give him",
        ],
    },
    {
        "id": "FROSTBITE",
        "text": "Do not rub or massage frostbitten tissue. Do not apply direct heat. Do not rewarm if there is risk of refreezing. Protect from further cold and seek warmth gradually.",
        "scenarios": [
            "fingers are white and hard from cold",
            "cant feel my toes they look white",
            "frostbite on hands what do i do",
            "frozen fingers how do i warm them",
            "skin is white and waxy from cold",
        ],
    },
    {
        "id": "HEAT_STROKE",
        "text": "Heat stroke: skin is HOT and DRY, person is confused — life-threatening, cool immediately: ice to neck, armpits, groin, wet clothing, fanning. Heat exhaustion: skin is cool and sweaty — move to shade, give water, rest.",
        "scenarios": [
            "his skin is hot and dry and not making sense",
            "she has been in the sun all day now confused",
            "heat stroke symptoms",
            "overheating not sweating confused",
            "heat exhaustion vs heat stroke",
            "person collapsed in the heat",
        ],
    },
    {
        "id": "EYE_CHEMICAL",
        "text": "Flush immediately with large amounts of clean water for at least 15-20 minutes. Do not rub the eye. Remove contact lenses if possible. Flush from inner corner outward.",
        "scenarios": [
            "chemical got in her eyes",
            "bleach splashed in my eye",
            "acid in the eye what do i do",
            "something burned his eye",
            "eye flushing after chemical exposure",
        ],
    },
    {
        "id": "FRACTURE",
        "text": "Do not attempt to straighten the limb. Immobilize in the position found using splint and padding. Use sticks, rolled clothing, or any rigid material. Secure above and below the injury. Check circulation below the injury.",
        "scenarios": [
            "his leg is bent the wrong way",
            "i think her arm is broken",
            "bone sticking out of skin",
            "fracture what do i do no hospital",
            "how do i make a splint",
            "broken leg in the wilderness",
        ],
    },
    {
        "id": "EMBEDDED_OBJECT",
        "text": "Do not remove an embedded object from a wound — it is acting as a plug. Stabilize it in place with padding around it. Removing it can cause rapid uncontrolled blood loss.",
        "scenarios": [
            "knife is still in his stomach",
            "glass embedded in the wound should i remove it",
            "object stuck in wound do i pull it out",
            "something is sticking out of the wound",
        ],
    },
    {
        "id": "POISONING",
        "text": "Do not induce vomiting for corrosive substances (bleach, acid, drain cleaner) or petroleum products (gasoline, kerosene). If unknown substance: do not induce vomiting. Keep airway clear. If unconscious, place on side.",
        "scenarios": [
            "my kid drank something under the sink",
            "he swallowed bleach",
            "she drank gasoline",
            "child ate unknown chemicals",
            "swallowed cleaning product",
            "drank something poisonous do i make them vomit",
        ],
    },
    {
        "id": "PLANT_INGESTION",
        "text": "Do not induce vomiting unless instructed. Do not give milk or water unless instructed. Try to identify or describe the plant. If lips or throat are swelling, treat as anaphylaxis.",
        "scenarios": [
            "we ate some berries on the trail and now sick",
            "child ate unknown berries",
            "ate mushrooms from the forest feel sick",
            "toxic plant eaten what do i do",
            "poisonous plant ingestion",
        ],
    },
    {
        "id": "DEHYDRATION_WATER",
        "text": "Do not drink untreated water from streams, lakes, or puddles. Boil for at least 1 minute (3 minutes at high altitude). If no fire: use iodine tablets, bleach 2 drops per liter wait 30 minutes, or a filter. Do not drink urine.",
        "scenarios": [
            "is this stream water safe to drink",
            "we have no clean water how do we purify",
            "can i drink river water in emergency",
            "no water supply how to find safe water",
            "how to purify water in the wild",
            "dehydrated no clean water",
        ],
    },
    {
        "id": "LOST_WILDERNESS",
        "text": "Stay in place unless you have confirmed direction and distance to safety. Moving increases exposure and makes rescue harder. Signal with fire, reflective material, or whistle — three signals is the universal distress sign.",
        "scenarios": [
            "we are lost in the woods",
            "dont know where we are no signal",
            "stranded in wilderness",
            "lost hiking no map no phone",
            "how do i signal for rescue",
            "should i stay or keep walking when lost",
        ],
    },
    {
        "id": "LIGHTNING",
        "text": "Do not shelter under isolated trees, in open fields, near water, or in caves with water. Crouch low on balls of feet with feet together — do not lie flat. Spread out at least 50 feet if in a group. A vehicle with metal roof is safe.",
        "scenarios": [
            "lightning storm coming where do i hide",
            "struck by lightning what do i do",
            "thunderstorm in open field",
            "lightning safety no shelter",
            "where to go during lightning storm",
        ],
    },
    {
        "id": "BEAR_ATTACK",
        "text": "Black bear: fight back aggressively, target nose and eyes. Brown or grizzly bear: play dead face-down, protect back of neck, spread legs. Do not run from any bear — it triggers chase instinct.",
        "scenarios": [
            "bear is attacking us",
            "grizzly bear charging",
            "black bear encounter what do i do",
            "bear attack how to survive",
            "bear coming toward us",
        ],
    },
    {
        "id": "WILDFIRE",
        "text": "Move to already-burned areas when possible. If no escape: lie face-down in a ditch, cover with soil or non-synthetic clothing. Cover mouth and nose. Breathe through cloth close to the ground.",
        "scenarios": [
            "wildfire is surrounding us",
            "fire spreading cant escape",
            "trapped by wildfire",
            "smoke everywhere fire coming",
            "how to survive being trapped in a wildfire",
        ],
    },
    {
        "id": "AVALANCHE",
        "text": "Before snow settles: punch arm toward surface. As snow settles: create air pocket in front of face immediately. Spit to find which way is down, dig the opposite direction. Stay calm — panic burns oxygen.",
        "scenarios": [
            "caught in an avalanche",
            "buried in snow after avalanche",
            "snowslide buried my friend",
            "avalanche survival what do i do",
            "trapped under snow",
        ],
    },
    {
        "id": "NOSEBLEED",
        "text": "Do not tilt the head back — blood flows into the throat. Lean slightly forward. Pinch the soft part of the nose (not the bridge) for 10-15 minutes without releasing.",
        "scenarios": [
            "nose is bleeding wont stop",
            "nosebleed how do i stop it",
            "blood coming from nose",
            "bad nosebleed",
        ],
    },
]

# Triage questions when no protocol matches
TRIAGE_PROTOCOL = """Ask ONE short question to clarify the situation.
Examples: Is there any bleeding? Can they breathe? Are they conscious? What did they swallow? Where is the pain?
Ask only the most critical question first."""

# ─────────────────────────────────────────────
# 2. BUILD FAISS INDEX
# ─────────────────────────────────────────────

def build_index(protocols, model):
    """
    Embeds all scenario sentences and builds a FAISS index.
    Stores a mapping from sentence index → protocol index.
    """
    sentences = []
    sentence_to_protocol = []

    for proto_idx, proto in enumerate(protocols):
        for scenario in proto["scenarios"]:
            sentences.append(scenario)
            sentence_to_protocol.append(proto_idx)

    print(f"Embedding {len(sentences)} scenario sentences...")
    embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity (normalized)
    index.add(embeddings)

    print(f"FAISS index built. {index.ntotal} vectors, dimension {dimension}.")
    return index, sentence_to_protocol


# ─────────────────────────────────────────────
# 3. MATCH FUNCTION
# ─────────────────────────────────────────────

THRESHOLD = 0.45  # Tune this: higher = stricter matching

def match_protocol(user_input: str, model, index, sentence_to_protocol, protocols, threshold=THRESHOLD):
    """
    Returns (protocol_text, protocol_id, score) if matched,
    or (TRIAGE_PROTOCOL, "TRIAGE", score) if no match.
    """
    embedding = model.encode([user_input], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(embedding, k=1)

    score = float(scores[0][0])
    matched_sentence_idx = int(indices[0][0])
    matched_proto_idx = sentence_to_protocol[matched_sentence_idx]
    matched_proto = protocols[matched_proto_idx]

    print(f"[MATCHER] Input: '{user_input}'")
    print(f"[MATCHER] Best match: {matched_proto['id']} (score: {score:.3f})")

    if score >= threshold:
        return matched_proto["text"], matched_proto["id"], score
    else:
        return TRIAGE_PROTOCOL, "TRIAGE", score


# ─────────────────────────────────────────────
# 4. SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are O.A.S.I.S., an emergency first-aid assistant.
Respond with short, direct commands only.
No markdown, no bullet points, no numbering, no symbols.
Speak in simple sentences. One instruction at a time.
Do not explain. Do not reassure. Just tell them what to do.

ACTIVE PROTOCOL:
{protocol}"""

def build_system_prompt(protocol_text: str) -> str:
    return BASE_SYSTEM_PROMPT.format(protocol=protocol_text)


# ─────────────────────────────────────────────
# 5. MAIN — INTERACTIVE TEST LOOP
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index, sentence_to_protocol = build_index(PROTOCOLS, model)

    print("\nO.A.S.I.S. Protocol Matcher ready.")
    print(f"Threshold: {THRESHOLD}")
    print("Type a user input to test matching. Ctrl+C to quit.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
            if not user_input:
                continue

            protocol_text, protocol_id, score = match_protocol(
                user_input, model, index, sentence_to_protocol, PROTOCOLS
            )

            print(f"\n--- MATCHED: {protocol_id} (score: {score:.3f}) ---")
            print(f"System prompt injection:\n{protocol_text}\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break