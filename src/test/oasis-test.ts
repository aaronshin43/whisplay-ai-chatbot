import dotenv from "dotenv";
import { getSystemPromptFromOasis } from "../core/OasisAdapter";
import { chatWithLLMStream, resetChatHistory } from "../cloud-api/llm";
import { ollamaReadyPromise } from "../cloud-api/local/ollama-llm";
import { Message } from "../type";

dotenv.config();

const query = process.argv[2] || "My friend was bitten by a snake";

console.log(`\n[OASIS TEST] Query: "${query}"`);
console.log("---------------------------------------------------");

async function waitForOasisService(retries = 30, delay = 2000): Promise<boolean> {
  // Pure Node version initializes on first call or explicitly, no need to wait for HTTP service.
  return true;
}

async function runTest() {
  try {
    // Wait for Ollama model to finish loading before starting any timer.
    // Without this, keepAlive model loading (~5-7s cold, ~0s warm) would
    // be included in TTFT measurements, giving misleading results.
    process.stdout.write("[Ollama] Waiting for model to load...");
    await ollamaReadyPromise;
    process.stdout.write(" ready\n");

    // Start timing right when the actual logic begins
    let totalStartTime = Date.now();

    console.log("[OASIS] Waiting for service to start...");
    if (!await waitForOasisService()) {
        console.error("\n[OASIS] Failed to connect to service. Is python script running?");
        process.exit(1);
    }
    
    // 1. Get System Prompt from OASIS Matcher
    console.log("\n[OASIS] Matching protocol...");
    let oasisStartTime = Date.now();
    const systemPrompt = await getSystemPromptFromOasis(query);
    let oasisEndTime = Date.now();
    console.log(`[OASIS] Matching took ${oasisEndTime - oasisStartTime}ms`);
    
    console.log("\n[OASIS] System Prompt Generated:");
    console.log("---------------------------------------------------");
    console.log(systemPrompt);
    console.log("---------------------------------------------------\n");

    // 2. Prepare Messages
    resetChatHistory();
    const messages: Message[] = [
      {
        role: "system",
        content: systemPrompt,
      },
      {
        role: "user",
        content: query,
      },
    ];

    // 3. Call LLM
    console.log("[LLM] Generating Response...");
    let llmStartTime = Date.now();
    let firstTokenTime = 0;
    
    await chatWithLLMStream(
      messages,
      (partial) => {
        if (!firstTokenTime) {
          firstTokenTime = Date.now();
          const ttfb = firstTokenTime - llmStartTime;
          console.log(`[Time to First Token (LLM only)]: ${ttfb}ms\n`);
          process.stdout.write(">> ");
        }
        process.stdout.write(partial);
      },
      () => {
        const endTime = Date.now();
        console.log(`\n\n[LLM Total Time]: ${endTime - llmStartTime}ms`);
        console.log(`[Total Execution Time]: ${endTime - totalStartTime}ms`);
        console.log("---------------------------------------------------");
        process.exit(0);
      },
      (thinking) => {
        // process.stdout.write(`[Thinking]: ${thinking}`); 
      }
    );

  } catch (error) {
    console.error("Test failed:", error);
    process.exit(1);
  }
}

// Gives time for the Python service to spawn if it's being started by the import
runTest();
