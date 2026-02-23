import dotenv from "dotenv";
import { matchOasisProtocol } from "../core/OasisAdapter";
import { chatWithLLMStream, resetChatHistory } from "../cloud-api/llm";
import { Message } from "../type";

dotenv.config();

const queries = process.argv.slice(2);
if (queries.length === 0) queries.push("My friend was bitten by a snake");

async function askWithLLM(
  systemPrompt: string,
  userQuery: string,
  label: string,
): Promise<void> {
  return new Promise(async (resolve) => {
    const messages: Message[] = [
      { role: "system", content: systemPrompt },
      { role: "user", content: userQuery },
    ];

    console.log(`[${label}] LLM generating response...`);
    let startTime = Date.now();
    let firstTokenTime = 0;

    await chatWithLLMStream(
      messages,
      (partial) => {
        if (!firstTokenTime) {
          firstTokenTime = Date.now();
          console.log(`[Time to First Token]: ${firstTokenTime - startTime}ms\n`);
          process.stdout.write(">> ");
        }
        process.stdout.write(partial);
      },
      () => {
        const endTime = Date.now();
        console.log(`\n[Total Time]: ${endTime - startTime}ms`);
        console.log("---------------------------------------------------");
        resolve();
      },
      () => {},
    );
  });
}

let activeProtocolId = "";

async function runTest() {
  try {
    resetChatHistory();

    for (let i = 0; i < queries.length; i++) {
      const query = queries[i];
      const label = queries.length === 1 ? "Q" : `Q${i + 1}`;

      console.log(`\n[${label}] Query: "${query}"`);
      console.log("---------------------------------------------------");

      const matchStart = Date.now();
      const oasis = await matchOasisProtocol(query);
      console.log(`[OASIS] ${oasis.protocolId} (score: ${oasis.score.toFixed(3)}) in ${Date.now() - matchStart}ms`);

      const isFollowUp = !oasis.isTriage && activeProtocolId === oasis.protocolId;

      if (!oasis.isTriage && !isFollowUp) {
        // First question with matched protocol → direct delivery
        activeProtocolId = oasis.protocolId;
        console.log(`[MODE] Direct Protocol Delivery (no LLM)\n`);
        console.log(`>> ${oasis.protocolText}`);
        console.log("---------------------------------------------------");
      } else if (oasis.isTriage) {
        // No protocol matched → LLM asks clarifying question
        activeProtocolId = "";
        console.log(`[MODE] Triage (LLM)\n`);
        await askWithLLM(oasis.systemPrompt, query, label);
      } else {
        // Follow-up on same protocol → LLM with simple prompt
        console.log(`[MODE] Follow-up on ${activeProtocolId} (LLM)\n`);
        await askWithLLM(oasis.systemPrompt, query, label);
      }
    }

    process.exit(0);
  } catch (error) {
    console.error("Test failed:", error);
    process.exit(1);
  }
}

setTimeout(runTest, 1000);
