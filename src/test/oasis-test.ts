import dotenv from "dotenv";
import { getSystemPromptFromOasis } from "../core/OasisAdapter";
import { chatWithLLMStream, resetChatHistory, warmupSystemPrompt } from "../cloud-api/llm";
import { Message } from "../type";

dotenv.config();

const queries = process.argv.slice(2);
if (queries.length === 0) queries.push("My friend was bitten by a snake");

async function askQuestion(
  systemPrompt: string,
  userQuery: string,
  label: string,
): Promise<void> {
  return new Promise(async (resolve) => {
    const messages: Message[] = [
      { role: "system", content: systemPrompt },
      { role: "user", content: userQuery },
    ];

    console.log(`\n[${label}] Query: "${userQuery}"`);
    console.log("[LLM] Generating Response...");
    let startTime = Date.now();
    let firstTokenTime = 0;

    await chatWithLLMStream(
      messages,
      (partial) => {
        if (!firstTokenTime) {
          firstTokenTime = Date.now();
          const ttfb = firstTokenTime - startTime;
          console.log(`[Time to First Token]: ${ttfb}ms\n`);
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

async function runTest() {
  try {
    const firstQuery = queries[0];

    // 1. Match protocol
    console.log("\n[OASIS] Matching protocol...");
    const matchStart = Date.now();
    const systemPrompt = await getSystemPromptFromOasis(firstQuery);
    console.log(`[OASIS] Match done in ${Date.now() - matchStart}ms`);

    console.log("\n[OASIS] System Prompt:");
    console.log("---------------------------------------------------");
    console.log(systemPrompt);
    console.log("---------------------------------------------------");

    // 2. Wait for KV cache warmup (fired inside getSystemPromptFromOasis)
    console.log("[LLM] Waiting for KV cache warmup...");
    const warmupStart = Date.now();
    await warmupSystemPrompt(systemPrompt);
    console.log(`[LLM] KV cache ready in ${Date.now() - warmupStart}ms`);

    // 3. Ask questions sequentially
    resetChatHistory();

    for (let i = 0; i < queries.length; i++) {
      const label = queries.length === 1 ? "Q" : `Q${i + 1}`;
      await askQuestion(systemPrompt, queries[i], label);
    }

    process.exit(0);
  } catch (error) {
    console.error("Test failed:", error);
    process.exit(1);
  }
}

setTimeout(runTest, 1000);
