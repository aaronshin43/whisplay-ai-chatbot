import dotenv from "dotenv";
import { getSystemPromptFromOasis } from "../core/OasisAdapter";
import { chatWithLLMStream, resetChatHistory, warmupSystemPrompt } from "../cloud-api/llm";
import { Message } from "../type";

dotenv.config();

const TEST_QUERIES = [
  "there's so much blood coming out of his leg i dont know what to do",
  "she fell and hit her head and blood wont stop",
  "i wrapped it with my shirt but its soaking through",
  "he just collapsed and hes not breathing",
  "i dont know how to do cpr can you walk me through it",
  "my kid swallowed something and cant breathe",
  "hes turning blue oh god hes turning blue",
  "he touched the engine and his hand looks really bad",
  "chemical got in her eyes she was screaming",
  "i think his leg is broken it looks bent the wrong way",
  "i have nothing to make a splint with what can i use",
  "we were lost overnight and hes shaking uncontrollably now he stopped shaking",
  "his skin is hot and dry and hes not making sense",
  "my kid drank something under the sink i dont know what it was",
  "she says she cant breathe after eating the peanuts",
  "he has an epipen but i dont know how to use it",
  "we pulled him out of the water hes not breathing",
  "uh my my friend hes uh he fell and i think um his arm",
  "she hit her head and went to sleep is that okay",
  "we are lost in the woods and there's nothing to eat. can I eat a mushroom? It looks okay.",
  "she got stung by something and her face is swelling",
  "i read somewhere you put butter on burns is that right",
  "snake bit him on the ankle like twenty minutes ago do we suck it out",
];

interface TestResult {
  query: string;
  protocolId: string;
  score: string;
  ttft: number;
  totalTime: number;
  response: string;
}

async function askQuestion(
  systemPrompt: string,
  userQuery: string,
): Promise<{ ttft: number; totalTime: number; response: string }> {
  return new Promise(async (resolve) => {
    const messages: Message[] = [
      { role: "system", content: systemPrompt },
      { role: "user", content: userQuery },
    ];

    let startTime = Date.now();
    let firstTokenTime = 0;
    let response = "";

    await chatWithLLMStream(
      messages,
      (partial) => {
        if (!firstTokenTime) {
          firstTokenTime = Date.now();
        }
        response += partial;
      },
      () => {
        const endTime = Date.now();
        resolve({
          ttft: firstTokenTime - startTime,
          totalTime: endTime - startTime,
          response: response.trim(),
        });
      },
      () => {},
    );
  });
}

async function runBatchTest() {
  const results: TestResult[] = [];

  console.log(`\n${"=".repeat(80)}`);
  console.log(`  OASIS BATCH TEST — ${TEST_QUERIES.length} queries`);
  console.log(`${"=".repeat(80)}\n`);

  for (let i = 0; i < TEST_QUERIES.length; i++) {
    const query = TEST_QUERIES[i];
    const num = `[${i + 1}/${TEST_QUERIES.length}]`;

    // Each query is independent — fresh match + fresh conversation
    resetChatHistory();

    console.log(`${num} "${query}"`);

    // Match protocol (imports matchProtocol internally)
    const matchStart = Date.now();
    const { matchProtocolLocal } = await import("../cloud-api/local/oasis-matcher-node");
    const matchResult = await matchProtocolLocal(query);
    const matchTime = Date.now() - matchStart;

    const systemPrompt = `You are O.A.S.I.S., an emergency first-aid assistant.
Respond with short, direct commands only.
No markdown, no bullet points, no numbering, no symbols.
Speak in simple sentences. One instruction at a time.
Do not explain. Do not reassure. Just tell them what to do.

ACTIVE PROTOCOL:
${matchResult.text}`;

    // Warmup KV cache
    await warmupSystemPrompt(systemPrompt);

    // Ask
    const { ttft, totalTime, response } = await askQuestion(systemPrompt, query);

    const protocolId = matchResult.protocol_id;
    const score = matchResult.score.toFixed(3);

    console.log(`   Protocol: ${protocolId} (${score}) | Match: ${matchTime}ms | TTFT: ${ttft}ms | Total: ${totalTime}ms`);
    console.log(`   >> ${response}`);
    console.log("");

    results.push({ query, protocolId, score, ttft, totalTime, response });
  }

  // Summary table
  console.log(`\n${"=".repeat(80)}`);
  console.log("  SUMMARY");
  console.log(`${"=".repeat(80)}`);
  console.log("");

  const ttfts = results.map(r => r.ttft);
  const avgTtft = Math.round(ttfts.reduce((a, b) => a + b, 0) / ttfts.length);
  const minTtft = Math.min(...ttfts);
  const maxTtft = Math.max(...ttfts);

  console.log(`  Queries:  ${results.length}`);
  console.log(`  Avg TTFT: ${avgTtft}ms`);
  console.log(`  Min TTFT: ${minTtft}ms`);
  console.log(`  Max TTFT: ${maxTtft}ms`);
  console.log("");

  // Per-query table
  console.log("  #  | Protocol             | Score | TTFT    | Total   | Response (first 60 chars)");
  console.log("  " + "-".repeat(100));
  results.forEach((r, i) => {
    const num = String(i + 1).padStart(2);
    const proto = r.protocolId.padEnd(20);
    const resp = r.response.substring(0, 60).replace(/\n/g, " ");
    console.log(`  ${num} | ${proto} | ${r.score} | ${String(r.ttft).padStart(5)}ms | ${String(r.totalTime).padStart(5)}ms | ${resp}`);
  });

  console.log(`\n${"=".repeat(80)}\n`);
  process.exit(0);
}

setTimeout(runBatchTest, 1000);
