import dotenv from "dotenv";
import { matchOasisProtocol } from "../core/OasisAdapter";
import { chatWithLLMStream, resetChatHistory } from "../cloud-api/llm";
import { Message } from "../type";

dotenv.config();

interface TestQuery {
  query: string;
  expectedProtocol: string;
}

const TEST_QUERIES: TestQuery[] = [
  // === BLEEDING (3) ===
  { query: "there's so much blood coming out of his leg i dont know what to do", expectedProtocol: "BLEEDING" },
  { query: "she fell and hit her head and blood wont stop", expectedProtocol: "BLEEDING" },
  { query: "i wrapped it with my shirt but its soaking through", expectedProtocol: "BLEEDING" },

  // === TOURNIQUET (3) ===
  { query: "his arm is spurting blood and pressure isnt stopping it", expectedProtocol: "TOURNIQUET" },
  { query: "blood is gushing from his leg nothing works", expectedProtocol: "TOURNIQUET" },
  { query: "artery bleeding wont stop with pressure what else can i do", expectedProtocol: "TOURNIQUET" },

  // === SNAKEBITE (3) ===
  { query: "snake bit him on the ankle like twenty minutes ago do we suck it out", expectedProtocol: "SNAKEBITE" },
  { query: "bitten by a snake in the woods what do i do", expectedProtocol: "SNAKEBITE" },
  { query: "snake got her on the leg should i cut the wound", expectedProtocol: "SNAKEBITE" },

  // === ANAPHYLAXIS (3) ===
  { query: "she says she cant breathe after eating the peanuts", expectedProtocol: "ANAPHYLAXIS" },
  { query: "he has an epipen but i dont know how to use it", expectedProtocol: "ANAPHYLAXIS" },
  { query: "she got stung by something and her face is swelling", expectedProtocol: "ANAPHYLAXIS" },

  // === CHOKING_ADULT (3) ===
  { query: "hes turning blue oh god hes turning blue", expectedProtocol: "CHOKING_ADULT" },
  { query: "something stuck in his throat he cant breathe at all", expectedProtocol: "CHOKING_ADULT" },
  { query: "she was eating and now shes grabbing her throat cant talk", expectedProtocol: "CHOKING_ADULT" },

  // === CHOKING_INFANT (3) ===
  { query: "my kid swallowed something and cant breathe", expectedProtocol: "CHOKING_INFANT" },
  { query: "baby is choking she is 6 months old", expectedProtocol: "CHOKING_INFANT" },
  { query: "my baby put something in her mouth and now shes turning blue", expectedProtocol: "CHOKING_INFANT" },

  // === CPR (3) ===
  { query: "he just collapsed and hes not breathing", expectedProtocol: "CPR" },
  { query: "i dont know how to do cpr can you walk me through it", expectedProtocol: "CPR" },
  { query: "she has no pulse what do i do", expectedProtocol: "CPR" },

  // === HYPOTHERMIA_SEVERE (3) ===
  { query: "we were lost overnight and hes shaking uncontrollably now he stopped shaking", expectedProtocol: "HYPOTHERMIA_SEVERE" },
  { query: "he was shivering but now hes stopped and hes barely conscious", expectedProtocol: "HYPOTHERMIA_SEVERE" },
  { query: "found someone in the snow not moving not shivering", expectedProtocol: "HYPOTHERMIA_SEVERE" },

  // === HYPOTHERMIA_PREVENTION (3) ===
  { query: "he fell in the river and is soaking wet and freezing", expectedProtocol: "HYPOTHERMIA_PREVENTION" },
  { query: "she is wet from the rain and wont stop shivering", expectedProtocol: "HYPOTHERMIA_PREVENTION" },
  { query: "we got soaked and its getting really cold what do we do", expectedProtocol: "HYPOTHERMIA_PREVENTION" },

  // === SPINAL_INJURY (3) ===
  { query: "she fell off the roof and says she cant feel her legs", expectedProtocol: "SPINAL_INJURY" },
  { query: "he cant move his neck after the accident should i move him", expectedProtocol: "SPINAL_INJURY" },
  { query: "fell from a ladder and has severe neck pain", expectedProtocol: "SPINAL_INJURY" },

  // === SEIZURE (3) ===
  { query: "he is having a seizure right now what do i do", expectedProtocol: "SEIZURE" },
  { query: "she is shaking on the ground and foaming at the mouth", expectedProtocol: "SEIZURE" },
  { query: "do i put something in his mouth during a seizure", expectedProtocol: "SEIZURE" },

  // === DIABETIC_EMERGENCY (3) ===
  { query: "hes diabetic and acting really confused and sweating a lot", expectedProtocol: "DIABETIC_EMERGENCY" },
  { query: "she has diabetes and just passed out", expectedProtocol: "DIABETIC_EMERGENCY" },
  { query: "i think hes having a low blood sugar emergency", expectedProtocol: "DIABETIC_EMERGENCY" },

  // === DROWNING (3) ===
  { query: "we pulled him out of the water hes not breathing", expectedProtocol: "DROWNING" },
  { query: "she was underwater for a while now shes not responding", expectedProtocol: "DROWNING" },
  { query: "kid fell in the pool and we just got him out hes unconscious", expectedProtocol: "DROWNING" },

  // === BURN (3) ===
  { query: "he touched the engine and his hand looks really bad", expectedProtocol: "BURN" },
  { query: "i read somewhere you put butter on burns is that right", expectedProtocol: "BURN" },
  { query: "she spilled boiling water on her arm what do i do", expectedProtocol: "BURN" },

  // === HEAD_INJURY (3) ===
  { query: "she hit her head and went to sleep is that okay", expectedProtocol: "HEAD_INJURY" },
  { query: "he fell and hit his head hard on the concrete now hes confused", expectedProtocol: "HEAD_INJURY" },
  { query: "knocked out briefly after hitting head should i worry", expectedProtocol: "HEAD_INJURY" },

  // === CHEST_PAIN (3) ===
  { query: "he is clutching his chest and sweating a lot", expectedProtocol: "CHEST_PAIN" },
  { query: "she has pain in her left arm and chest i think heart attack", expectedProtocol: "CHEST_PAIN" },
  { query: "sudden chest pressure and he looks pale", expectedProtocol: "CHEST_PAIN" },

  // === FROSTBITE (3) ===
  { query: "his fingers are white and hard from the cold", expectedProtocol: "FROSTBITE" },
  { query: "cant feel my toes they look white and waxy", expectedProtocol: "FROSTBITE" },
  { query: "her hands are frozen how do i warm them up safely", expectedProtocol: "FROSTBITE" },

  // === HEAT_STROKE (3) ===
  { query: "his skin is hot and dry and hes not making sense", expectedProtocol: "HEAT_STROKE" },
  { query: "she collapsed in the heat and isnt sweating", expectedProtocol: "HEAT_STROKE" },
  { query: "been in the sun all day now hes confused and his skin is burning", expectedProtocol: "HEAT_STROKE" },

  // === EYE_CHEMICAL (3) ===
  { query: "chemical got in her eyes she was screaming", expectedProtocol: "EYE_CHEMICAL" },
  { query: "bleach splashed in his eye what do i do", expectedProtocol: "EYE_CHEMICAL" },
  { query: "acid got in her eye how long do i flush", expectedProtocol: "EYE_CHEMICAL" },

  // === FRACTURE (3) ===
  { query: "i think his leg is broken it looks bent the wrong way", expectedProtocol: "FRACTURE" },
  { query: "i have nothing to make a splint with what can i use", expectedProtocol: "FRACTURE" },
  { query: "uh my my friend hes uh he fell and i think um his arm is broken", expectedProtocol: "FRACTURE" },

  // === EMBEDDED_OBJECT (3) ===
  { query: "theres a piece of glass stuck in his leg should i pull it out", expectedProtocol: "EMBEDDED_OBJECT" },
  { query: "knife is still in his stomach what do i do", expectedProtocol: "EMBEDDED_OBJECT" },
  { query: "something is sticking out of the wound do i remove it", expectedProtocol: "EMBEDDED_OBJECT" },

  // === POISONING (3) ===
  { query: "my kid drank something under the sink i dont know what it was", expectedProtocol: "POISONING" },
  { query: "he swallowed bleach should i make him vomit", expectedProtocol: "POISONING" },
  { query: "child ate some cleaning product under the counter", expectedProtocol: "POISONING" },

  // === PLANT_INGESTION (3) ===
  { query: "we are lost in the woods and there's nothing to eat. can I eat a mushroom? It looks okay.", expectedProtocol: "PLANT_INGESTION" },
  { query: "she ate some berries on the trail and now feels sick", expectedProtocol: "PLANT_INGESTION" },
  { query: "kid ate unknown berries from the bush what do i do", expectedProtocol: "PLANT_INGESTION" },

  // === DEHYDRATION_WATER (3) ===
  { query: "we have no clean water how do we make it safe to drink", expectedProtocol: "DEHYDRATION_WATER" },
  { query: "is this stream water safe can i drink it", expectedProtocol: "DEHYDRATION_WATER" },
  { query: "totally dehydrated no water supply what can i do", expectedProtocol: "DEHYDRATION_WATER" },

  // === LOST_WILDERNESS (3) ===
  { query: "we are completely lost in the woods no phone signal", expectedProtocol: "LOST_WILDERNESS" },
  { query: "should i stay put or keep walking when lost", expectedProtocol: "LOST_WILDERNESS" },
  { query: "how do i signal for rescue in the wilderness", expectedProtocol: "LOST_WILDERNESS" },

  // === LIGHTNING (3) ===
  { query: "theres a lightning storm and were in an open field", expectedProtocol: "LIGHTNING" },
  { query: "where do i hide from lightning no shelter around", expectedProtocol: "LIGHTNING" },
  { query: "someone just got struck by lightning what do i do", expectedProtocol: "LIGHTNING" },

  // === BEAR_ATTACK (3) ===
  { query: "theres a bear coming toward us what do we do", expectedProtocol: "BEAR_ATTACK" },
  { query: "grizzly bear is charging at us", expectedProtocol: "BEAR_ATTACK" },
  { query: "black bear encounter do i run or fight", expectedProtocol: "BEAR_ATTACK" },

  // === WILDFIRE (3) ===
  { query: "wildfire is surrounding us and we cant escape", expectedProtocol: "WILDFIRE" },
  { query: "smoke everywhere and fire coming closer what do i do", expectedProtocol: "WILDFIRE" },
  { query: "trapped by wildfire how do i survive", expectedProtocol: "WILDFIRE" },

  // === AVALANCHE (3) ===
  { query: "my friend got buried in an avalanche", expectedProtocol: "AVALANCHE" },
  { query: "caught in a snowslide what do i do", expectedProtocol: "AVALANCHE" },
  { query: "buried under snow after avalanche how to survive", expectedProtocol: "AVALANCHE" },

  // === NOSEBLEED (3) ===
  { query: "his nose is bleeding and it wont stop", expectedProtocol: "NOSEBLEED" },
  { query: "bad nosebleed should i tilt my head back", expectedProtocol: "NOSEBLEED" },
  { query: "blood coming from nose for ten minutes now", expectedProtocol: "NOSEBLEED" },

  // === BOUNDARY / AMBIGUOUS CASES (5) ===
  { query: "head wound with lots of blood should i press on it or not", expectedProtocol: "BLEEDING" },
  { query: "he fell off a bike and his arm is bleeding and looks bent", expectedProtocol: "FRACTURE" },
  { query: "she ate peanuts and her throat is closing and shes got a rash", expectedProtocol: "ANAPHYLAXIS" },
  { query: "he was in the cold water and now hes shivering badly", expectedProtocol: "HYPOTHERMIA_PREVENTION" },
  { query: "baby swallowed a small toy and is coughing but still breathing", expectedProtocol: "CHOKING_INFANT" },

  // === PANIC / INCOHERENT EXPRESSIONS (5) ===
  { query: "oh god oh god theres blood everywhere please help", expectedProtocol: "BLEEDING" },
  { query: "he he he just fell down and and hes not moving help", expectedProtocol: "CPR" },
  { query: "fire fire the forest is on fire we cant get out", expectedProtocol: "WILDFIRE" },
  { query: "please help snake snake bit him help", expectedProtocol: "SNAKEBITE" },
  { query: "i dont know what to do shes not breathing shes blue", expectedProtocol: "CHOKING_ADULT" },

  // === NON-MEDICAL / TRIAGE (5) ===
  { query: "whats the weather like today", expectedProtocol: "TRIAGE" },
  { query: "tell me a joke", expectedProtocol: "TRIAGE" },
  { query: "how do i cook pasta", expectedProtocol: "TRIAGE" },
  { query: "my car wont start", expectedProtocol: "TRIAGE" },
  { query: "something happened but im not sure what", expectedProtocol: "TRIAGE" },
];

interface TestResult {
  query: string;
  expectedProtocol: string;
  protocolId: string;
  matchCorrect: boolean;
  score: string;
  mode: "direct" | "llm";
  response: string;
}

async function askWithLLM(
  systemPrompt: string,
  userQuery: string,
): Promise<string> {
  return new Promise(async (resolve) => {
    const messages: Message[] = [
      { role: "system", content: systemPrompt },
      { role: "user", content: userQuery },
    ];
    let response = "";
    await chatWithLLMStream(
      messages,
      (partial) => { response += partial; },
      () => { resolve(response.trim()); },
      () => {},
    );
  });
}

async function runBatchTest() {
  const results: TestResult[] = [];

  console.log(`\n${"=".repeat(80)}`);
  console.log(`  OASIS BATCH TEST (Direct Delivery) — ${TEST_QUERIES.length} queries`);
  console.log(`${"=".repeat(80)}\n`);

  for (let i = 0; i < TEST_QUERIES.length; i++) {
    const { query, expectedProtocol } = TEST_QUERIES[i];
    const num = `[${i + 1}/${TEST_QUERIES.length}]`;

    resetChatHistory();
    console.log(`${num} "${query}"`);
    console.log(`   Expected: ${expectedProtocol}`);

    const oasis = await matchOasisProtocol(query);

    let response: string;
    let mode: "direct" | "llm";

    if (!oasis.isTriage) {
      // Matched protocol → direct delivery (no LLM)
      response = oasis.protocolText;
      mode = "direct";
    } else {
      // TRIAGE → LLM asks clarifying question
      response = await askWithLLM(oasis.systemPrompt, query);
      mode = "llm";
    }

    const protocolId = oasis.protocolId;
    const score = oasis.score.toFixed(3);
    const matchCorrect = protocolId === expectedProtocol;
    const icon = matchCorrect ? "✅" : "❌";

    console.log(`   ${icon} Got: ${protocolId} (${score}) [${mode}]`);
    console.log(`   >> ${response}`);
    console.log("");

    results.push({ query, expectedProtocol, protocolId, matchCorrect, score, mode, response });
  }

  // Summary
  console.log(`\n${"=".repeat(80)}`);
  console.log("  SUMMARY");
  console.log(`${"=".repeat(80)}`);
  console.log("");

  const correctCount = results.filter(r => r.matchCorrect).length;
  const directCount = results.filter(r => r.mode === "direct").length;
  const llmCount = results.filter(r => r.mode === "llm").length;

  console.log(`  Queries:           ${results.length}`);
  console.log(`  Match Accuracy:    ${correctCount}/${results.length} (${Math.round(correctCount / results.length * 100)}%)`);
  console.log(`  Direct Delivery:   ${directCount} (no LLM, 100% protocol accuracy)`);
  console.log(`  LLM (Triage):      ${llmCount}`);
  console.log("");

  // Per-query table
  console.log("   #  | OK | Mode   | Expected             | Got                  | Score | Response (50 chars)");
  console.log("  " + "-".repeat(110));
  results.forEach((r, i) => {
    const num = String(i + 1).padStart(3);
    const icon = r.matchCorrect ? "✅" : "❌";
    const modeStr = r.mode.padEnd(6);
    const expected = r.expectedProtocol.padEnd(20);
    const got = r.protocolId.padEnd(20);
    const resp = r.response.substring(0, 50).replace(/\n/g, " ");
    console.log(`  ${num} | ${icon} | ${modeStr} | ${expected} | ${got} | ${r.score} | ${resp}`);
  });

  // Mismatches
  const mismatches = results.filter(r => !r.matchCorrect);
  if (mismatches.length > 0) {
    console.log(`\n  MISMATCHES (${mismatches.length}):`);
    console.log("  " + "-".repeat(80));
    mismatches.forEach((r, i) => {
      console.log(`  ${i + 1}. "${r.query}"`);
      console.log(`     Expected: ${r.expectedProtocol} | Got: ${r.protocolId} (${r.score})`);
    });
  }

  // Quality warnings
  const shortResponses = results.filter(r => r.mode === "direct" && r.response.length < 30);
  if (shortResponses.length > 0) {
    console.log(`\n  SHORT RESPONSES (<30 chars):`);
    shortResponses.forEach(r => {
      console.log(`  - [${r.protocolId}] "${r.response}"`);
    });
  }

  console.log(`\n${"=".repeat(80)}\n`);
  process.exit(0);
}

setTimeout(runBatchTest, 1000);
