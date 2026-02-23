
import { pipeline, env } from "@xenova/transformers";

// Configure local cache if needed, but default is fine
env.allowLocalModels = false; 
env.useBrowserCache = false;

// 1. PROTOCOL DATABASE (Identical to Python version)
interface Protocol {
    id: string;
    text: string;
    scenarios: string[];
}

const PROTOCOLS: Protocol[] = [
    {
        "id": "BLEEDING",
        "text": "Apply firm continuous pressure with cloth or hand. Keep pressure constant for at least 10 minutes without checking. If cloth soaks through, add more on top — do not remove it. Elevate the limb above heart level if possible.",
        "scenarios": [
            "there is so much blood coming out",
            "blood wont stop",
            "i wrapped it but its soaking through",
            "he has a deep cut and bleeding badly",
            "wound is bleeding a lot",
            "blood everywhere i dont know what to do",
            "she cut her hand it wont stop bleeding",
            "red stuff coming out of the cut",
            "deep wound bleeding heavily",
        ],
    },
    {
        "id": "TOURNIQUET",
        "text": "Apply tourniquet 2-3 inches above the wound, not on a joint. Tighten until bleeding stops completely. Write the time of application on the skin. Once applied, do not remove or loosen it.",
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
        "text": "Immobilize the bitten limb and keep it below heart level. Remove rings or tight clothing near the bite. Keep the person still and calm. Do NOT suck out venom. Do NOT cut or squeeze the bite.",
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
        "text": "Inject epinephrine (EpiPen) into outer thigh immediately, hold for 10 seconds. If no EpiPen, lay the person flat with legs elevated. If breathing is difficult, sit them up instead. Do not give anything by mouth. A second dose may be needed in 5-15 minutes.",
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
        "text": "Give 5 firm back blows between the shoulder blades. Then give 5 abdominal thrusts (Heimlich maneuver). Alternate back blows and abdominal thrusts until the object comes out. If the person can still cough forcefully, let them keep coughing. Do not stick your fingers in their mouth.",
        "scenarios": [
            "she was eating and now shes grabbing her throat",
            "something stuck in his throat cant breathe",
            "he swallowed something and is choking",
            "turning blue cant get air",
            "food stuck in throat",
            "i did the heimlich but it didnt come out",
            "adult choking on food",
            "turning blue cant breathe",
            "face is going blue",
            "lips turning purple not breathing",
            "changing color cant get air",
        ],
    },
    {
        "id": "CHOKING_INFANT",
        "text": "Place the baby face-down on your forearm. Give 5 back blows between the shoulder blades, then flip face-up and give 5 chest thrusts. Do not do abdominal thrusts on infants. Do not do blind finger sweeps in the mouth.",
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
        "text": "Start CPR now. Place heel of hand on center of chest. Push hard and fast — 2 inches deep, 100-120 pushes per minute. After 30 pushes, give 2 rescue breaths. Keep going until the person breathes or help arrives.",
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
        "text": "This is severe hypothermia — life-threatening. Warm the core gently: place warm covers on chest, neck, armpits, and groin. Move the person very gently — sudden movement can trigger cardiac arrest. Do not rub or massage the skin. Do not apply direct heat to arms or legs.",
        "scenarios": [
            "he stopped shivering after being out in the cold",
            "we were lost overnight and he is not shivering anymore",
            "shivering stopped is that bad",
            "found someone in the snow not moving",
            "severe hypothermia what do i do",
            "he was shaking and now he stopped",
            "not shivering anymore is that bad",
            "was shaking now stopped and quiet",
            "stopped trembling in the cold",
        ],
    },
    {
        "id": "HYPOTHERMIA_PREVENTION",
        "text": "Remove wet clothing immediately — wet fabric loses heat 25 times faster than dry. Insulate from the ground first before worrying about wind. Eat and drink if possible — the body needs fuel to generate heat.",
        "scenarios": [
            "he fell in the river and is soaking wet and cold",
            "she is wet and shivering",
            "soaked from rain and getting cold",
            "wet clothes in cold weather what do i do",
            "how to treat someone who is cold and wet",
            "he is in cold water and shivering badly",
            "shes shivering uncontrollably after getting wet",
            "soaking wet and shaking from the cold",
        ],
    },
    {
        "id": "SPINAL_INJURY",
        "text": "Keep the person completely still. Stabilize the head and neck in the position found. Do not move them unless there is an immediate life threat. If you must move them, keep head, neck, and spine aligned as one unit.",
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
        "text": "Clear hard objects away from the person. Let the seizure happen — do not restrain them and do not put anything in their mouth. After convulsions stop, turn them on their side. Time the seizure — if over 5 minutes, this is an emergency.",
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
        "text": "If the person is conscious and can swallow, give sugar immediately: juice, candy, or glucose gel. If unconscious, do not give anything by mouth — place them on their side. When in doubt, give sugar — it is safer than withholding it.",
        "scenarios": [
            "diabetic acting really weird and sweating",
            "he is diabetic and confused",
            "she has diabetes and passed out",
            "diabetic emergency low blood sugar",
            "insulin reaction what do i do",
            "diabetic person unconscious",
            "low blood sugar emergency what do i do",
            "he is having a blood sugar crash",
            "hypoglycemia symptoms hes shaking and sweating",
        ],
    },
    {
        "id": "DROWNING",
        "text": "Start CPR immediately. Do not waste time trying to drain water from the lungs. If the person dove in or fell from height, assume possible spinal injury. Keep them warm after rescue — drowning victims lose heat rapidly.",
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
        "text": "Cool the burn with running water for 20 minutes. Cover loosely with clean non-stick material. Do not pop blisters. Do not apply butter, oil, toothpaste, or ice.",
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
        "text": "Keep the person awake for at least 2 hours after the head impact. Watch for these danger signs: unequal pupils, repeated vomiting, worsening headache, confusion, or one-sided weakness. If any of these appear, this is an emergency.",
        "scenarios": [
            "she hit her head and went to sleep is that okay",
            "he fell and hit his head hard",
            "knocked out briefly after hitting head",
            "concussion what do i watch for",
            "head injury can she sleep",
            "hit head on concrete now confused",
            "bumped head hard should i worry",
            "fell and hit head now dizzy",
        ],
    },
    {
        "id": "CHEST_PAIN",
        "text": "Give one adult aspirin (325mg) to chew, not swallow whole, if the person is conscious and not allergic. Keep them still and calm. Loosen tight clothing. Do not give food or water.",
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
        "text": "Protect the frostbitten area from further cold. Seek warmth gradually. Do not rub or massage the frozen tissue. Do not apply direct heat. Do not rewarm if there is any risk of refreezing.",
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
        "text": "Cool the person immediately — apply ice to neck, armpits, and groin. Use wet clothing and fanning. This is life-threatening if the skin is hot and dry and the person is confused. If the skin is cool and sweaty, move to shade, give water, and rest.",
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
        "text": "Flush the eye immediately with large amounts of clean water for at least 15-20 minutes. Flush from inner corner outward. Remove contact lenses if possible. Do not rub the eye.",
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
        "text": "Immobilize the limb in the position found — use sticks, rolled clothing, or any rigid material as a splint. Secure above and below the injury. Check circulation below the injury. Do not attempt to straighten the limb.",
        "scenarios": [
            "his leg is bent the wrong way",
            "i think her arm is broken",
            "bone sticking out of skin",
            "fracture what do i do no hospital",
            "how do i make a splint",
            "broken leg in the wilderness",
            "he fell and hurt his arm",
            "she fell and her arm looks wrong",
            "friend fell arm might be broken",
        ],
    },
    {
        "id": "EMBEDDED_OBJECT",
        "text": "Leave the object in place — it is acting as a plug and slowing the bleeding. Stabilize it with padding around it. Do not pull it out — removing it can cause rapid uncontrolled blood loss.",
        "scenarios": [
            "knife is still in his stomach",
            "glass embedded in the wound should i remove it",
            "object stuck in wound do i pull it out",
            "something is sticking out of the wound",
        ],
    },
    {
        "id": "POISONING",
        "text": "Keep the airway clear. If unconscious, place the person on their side. Do not induce vomiting — especially for bleach, acid, drain cleaner, gasoline, or any unknown substance. Try to identify what was swallowed.",
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
        "text": "NEVER eat unknown or unidentified plants, mushrooms, or berries. If already eaten, do not induce vomiting. Try to identify or photograph the plant. If lips or throat are swelling, treat as a severe allergic reaction.",
        "scenarios": [
            "we ate some berries on the trail and now sick",
            "child ate unknown berries",
            "ate mushrooms from the forest feel sick",
            "toxic plant eaten what do i do",
            "poisonous plant ingestion",
            "can i eat this mushroom",
            "is this berry safe to eat",
            "found berries are they safe",
            "mushroom looks edible should i eat it",
        ],
    },
    {
        "id": "DEHYDRATION_WATER",
        "text": "Boil water for at least 1 minute (3 minutes at high altitude) before drinking. If you cannot boil, use iodine tablets, 2 drops of bleach per liter and wait 30 minutes, or a filter. Do not drink untreated water from streams, lakes, or puddles. Do not drink urine.",
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
        "text": "Stay where you are. Moving increases exposure and makes rescue harder. Signal for help: use fire, reflective material, or a whistle. Three signals is the universal distress sign. Only move if you have confirmed direction and distance to safety.",
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
        "text": "Crouch low on the balls of your feet with feet together. If in a group, spread out at least 50 feet apart. A vehicle with a metal roof is safe shelter. Do not shelter under isolated trees. Do not lie flat on the ground.",
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
        "text": "Black bear: fight back aggressively — target the nose and eyes. Brown or grizzly bear: play dead face-down, protect the back of your neck, spread your legs wide. Do not run from any bear — running triggers their chase instinct.",
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
        "text": "Move to an area that has already burned — it is the safest ground. If you cannot escape, lie face-down in a ditch or depression. Cover yourself with soil or non-synthetic clothing. Cover your mouth and nose. Breathe through cloth close to the ground.",
        "scenarios": [
            "wildfire is surrounding us",
            "fire spreading cant escape",
            "trapped by wildfire",
            "smoke everywhere fire coming",
            "how to survive being trapped in a wildfire",
            "forest fire coming this way what do we do",
            "fire fire everything is on fire",
            "the trees are burning around us help",
        ],
    },
    {
        "id": "AVALANCHE",
        "text": "Punch one arm toward the surface before the snow settles. Create an air pocket in front of your face immediately. Spit to find which way is down, then dig the opposite direction. Stay calm — panic burns oxygen faster.",
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
        "text": "Lean slightly forward. Pinch the soft part of the nose (not the bridge) firmly for 10-15 minutes without releasing. Do not tilt the head back — blood will flow into the throat.",
        "scenarios": [
            "nose is bleeding wont stop",
            "nosebleed how do i stop it",
            "blood coming from nose",
            "bad nosebleed",
        ],
    },
]

const TRIAGE_PROTOCOL = "Ask ONE short question to clarify the situation.\nExamples: Is there any bleeding? Can they breathe? Are they conscious? What did they swallow? Where is the pain?\nAsk only the most critical question first.";

// 2. EMBEDDING PIPELINE
let extractor: any = null;
let indexMatrix: number[][] = [];
let indexMap: { protocolIndex: number, text: string }[] = [];

// Initialize Model & Build Index
export async function initializeMatcher() {
    if (extractor) return; // Already initialized

    console.log("[OASIS-NODE] Loading model: all-MiniLM-L6-v2...");
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    
    console.log("[OASIS-NODE] Building index...");
    const scenarios: string[] = [];
    PROTOCOLS.forEach((p, pIdx) => {
        p.scenarios.forEach(s => {
            scenarios.push(s);
            indexMap.push({ protocolIndex: pIdx, text: s });
        });
    });

    // Compute embeddings for all scenarios
    // Use for loop for sequential processing to avoid memory spike
    for (let i = 0; i < scenarios.length; i++) {
        const output = await extractor(scenarios[i], { pooling: 'mean', normalize: true });
        indexMatrix.push(Array.from(output.data));
    }
    console.log(`[OASIS-NODE] Ready. Index size: ${indexMatrix.length}`);
}

// 3. MATCHING LOGIC
const THRESHOLD = 0.47;

export interface OasisMatchResult {
    match: boolean;
    score: number;
    protocol_id: string;
    text: string;
    triage: boolean;
}

export async function matchProtocolLocal(query: string): Promise<OasisMatchResult> {
    if (!extractor) await initializeMatcher();

    // Embed query
    const output = await extractor(query, { pooling: 'mean', normalize: true });
    const queryVec = Array.from(output.data) as number[];

    // Find best match (brute force cosine similarity)
    let bestScore = -1;
    let bestIdx = -1;

    for (let i = 0; i < indexMatrix.length; i++) {
        const score = cosineSimilarity(queryVec, indexMatrix[i]);
        if (score > bestScore) {
            bestScore = score;
            bestIdx = i;
        }
    }

    const isMatch = bestScore >= THRESHOLD;
    let matchedProto = null;
    let protoId = "TRIAGE";
    let text = TRIAGE_PROTOCOL;

    if (isMatch && bestIdx !== -1) {
        const pIdx = indexMap[bestIdx].protocolIndex;
        matchedProto = PROTOCOLS[pIdx];
        protoId = matchedProto.id;
        text = matchedProto.text;
    }

    console.log(`[OASIS-NODE] Match: ${isMatch} (${bestScore.toFixed(3)}) -> ${protoId}`);

    return {
        match: isMatch,
        score: bestScore,
        protocol_id: protoId,
        text: text,
        triage: !isMatch
    };
}

// Helper for cosine similarity
function cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0;
    let magA = 0;
    let magB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        magA += a[i] * a[i];
        magB += b[i] * b[i];
    }
    return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}
