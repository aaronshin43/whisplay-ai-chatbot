
import dotenv from "dotenv";
import { chatWithRAGStream } from "../core/rag-chat";
import { Message } from "../type";
import ollamaLLM from "../cloud-api/local/ollama-llm";

const { resetChatHistory } = ollamaLLM;

dotenv.config();

const query = process.argv[2] || "How do I perform CPR on an adult?";

console.log(`Testing RAG Chat with query: "${query}"`);
console.log("---------------------------------------------------");

const messages: Message[] = [
  {
    role: "user",
    content: query,
  },
];

// Reset standard LLM history to avoid pollution for this test
resetChatHistory();

let startTime = Date.now();
let firstTokenTime = 0;
let tokenCount = 0;

chatWithRAGStream(
  messages,
  (partial) => {
    if (tokenCount === 0) {
      firstTokenTime = Date.now();
      const ttfb = firstTokenTime - startTime;
      console.log(`\n[Time to First Token]: ${ttfb}ms`);
      process.stdout.write("[O.A.S.I.S.]: ");
    }
    process.stdout.write(partial);
    tokenCount++;
  },
  () => {
    const endTime = Date.now();
    const totalTime = endTime - startTime;
    console.log("\n---------------------------------------------------");
    console.log(`[Total Time]: ${totalTime}ms`);
    process.exit(0);
  },
  (thinking) => {
    process.stdout.write(`[Thinking]: ${thinking}`);
  }
).catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
