
import { Message, OllamaMessage } from "../type";
import { getRelevantContext } from "./rag-pipeline";
import ollamaLLM from "../cloud-api/local/ollama-llm";
import dotenv from "dotenv";
import axios from "axios";

const { chatWithLLMStream: originalChatWithLLMStream } = ollamaLLM;

dotenv.config();

const OLLAMA_ENDPOINT = process.env.OLLAMA_ENDPOINT || "http://localhost:11434";
const EMBEDDING_MODEL_NAME = "mxbai-embed-large"; 
const CHAT_MODEL_NAME = process.env.OLLAMA_MODEL || "qwen3:1.7b";
const REWRITE_MODEL_NAME = "qwen2.5:0.5b"; 

// Define a specialized System Prompt for First Aid
const FIRST_AID_SYSTEM_PROMPT = `You are O.A.S.I.S., an emergency first-aid assistant. You are operating offline in a potential crisis situation.
Your goal is to provide immediate, step-by-step first aid instructions based ONLY on the context provided below.

Instructions:
1. Be concise. Use short sentences (bullet points preferred).
2. Prioritize life-saving actions (Airway, Breathing, Circulation).
3. If the context does not contain the answer, state "I do not have information on this specific injury in my database. Stabilize the patient and seek professional help."
4. Do NOT make up medical advice not found in the context.
5. If the user asks general questions not related to emergencies, answer briefly and professionally.

Context:
{context}
`;

/**
 * Rewrites the user query into a concise medical search query using LLM.
 * @param originalQuery The user's spoken input.
 * @returns A concise search query string.
 */
const rewriteQuery = async (originalQuery: string): Promise<string> => {
  try {
    const rewritePrompt = `Convert this spoken text into a concise medical search query. Output ONLY the keywords. Do not explain. Text: "${originalQuery}"`;

    const response = await axios.post(`${OLLAMA_ENDPOINT}/api/chat`, {
      model: REWRITE_MODEL_NAME,
      messages: [{ role: "user", content: rewritePrompt }],
      stream: false,
      options: {
        num_predict: 20, // Slightly increased limit
        temperature: 0,   // Deterministic output
      },
    });

    let rewritten = response.data?.message?.content?.trim() || originalQuery;
    
    // Remove "Search query:" or thinking blocks if any model adds them
    rewritten = rewritten.replace(/<think>[\s\S]*?<\/think>/gi, ""); // Remove think blocks
    rewritten = rewritten.replace(/^search query: /i, "").replace(/^query: /i, "").replace(/"/g, "").trim(); 
    
    return rewritten || originalQuery;
  } catch (error) {
    console.warn(`[QueryRewrite] Failed to rewrite using ${REWRITE_MODEL_NAME}, falling back to original query.`);
    // console.error(error);
    return originalQuery; 
  }
};

/**
 * Enhanced Chat Function with RAG
 * 1. Rewrites user query for better search accuracy.
 * 2. Retrieves relevant context from LanceDB based on the rewritten query.
 * 3. Injects the context into the System Prompt.
 * 4. Calls the original LLM function.
 */
export const chatWithRAGStream = async (
  messages: Message[],
  partialCallback: (partialAnswer: string) => void,
  endCallback: () => void,
  partialThinkingCallback?: (partialThinking: string) => void
) => {
  // 1. Identify the user query (last message)
  const userQuery = messages[messages.length - 1].content;
  
  // 2. Rewrite Query (Experimental)
  console.log(`[RAG] Original Query: "${userQuery}"`);
  const searchKey = await rewriteQuery(userQuery);
  console.log(`[RAG] Rewritten Query: "${searchKey}"`);
  
  
  // 3. Retrieve Context
  let context = "";
  try {
    context = await getRelevantContext(searchKey, 3); // Reduced from 3 to 1 for optimization
    if (context) {
      console.log(`[RAG] Found context length: ${context.length}`);
    } else {
      console.log(`[RAG] No relevant context found.`);
    }
  } catch (err) {
    console.error(`[RAG] Error retrieving context:`, err);
  }

  // 4. Construct System Message with Context
  const systemMessageContent = FIRST_AID_SYSTEM_PROMPT.replace("{context}", context || "No specific context available.");
  console.log("[RAG] final prompt: ", systemMessageContent)
  // 5. Prepare messages for Ollama
  // Replace the default system prompt or add a new one at the beginning
  const ragMessages: Message[] = [
    { role: "system", content: systemMessageContent },
    ...messages // This might contain previous chat history
  ];

  console.log("[RAG] Sending prompt to LLM...");
  await originalChatWithLLMStream(
      ragMessages, 
      partialCallback, 
      endCallback, 
      partialThinkingCallback
  );
};
