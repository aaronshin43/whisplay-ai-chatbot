
import dotenv from "dotenv";
import ollamaLLM from "../cloud-api/local/ollama-llm";
import { Message } from "../type";

const { chatWithLLMStream } = ollamaLLM;

dotenv.config();

const prompt = process.argv[2] || "Hello, who are you?";

console.log(`Testing Ollama with prompt: "${prompt}"`);
console.log("---------------------------------------------------");

const messages: Message[] = [
  {
    role: "user",
    content: prompt,
  },
];

let startTime = Date.now();
let firstTokenTime = 0;
let tokenCount = 0;

chatWithLLMStream(
  messages,
  (partial: string) => {
    if (tokenCount === 0) {
      firstTokenTime = Date.now();
      const ttfb = firstTokenTime - startTime;
      console.log(`\n[Time to First Token]: ${ttfb}ms`);
      process.stdout.write("[Response]: ");
    }
    process.stdout.write(partial);
    tokenCount++;
  },
  () => {
    const endTime = Date.now();
    const totalTime = endTime - startTime;
    console.log("\n---------------------------------------------------");
    console.log(`[Total Time]: ${totalTime}ms`);
    if (tokenCount > 0) {
        // This is a rough estimate since partial callbacks might contain multiple tokens or parts of tokens
        // but it gives a general idea of throughput.
        // For accurate TPS, using the token count from the model response would be better if available.
        console.log(`[Approx. Generation Time]: ${endTime - firstTokenTime}ms`);
    }
    process.exit(0);
  },
  (thinking: string) => {
      // If thinking is enabled and supported
      process.stdout.write(`[Thinking]: ${thinking}`);
  }
).catch((err: any) => {
  console.error("Error:", err);
  process.exit(1);
});
