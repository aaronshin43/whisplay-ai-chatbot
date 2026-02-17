
import { Message, OllamaMessage } from "../type";
import { getRelevantContext } from "./rag-pipeline";
import ollamaLLM from "../cloud-api/local/ollama-llm";
import dotenv from "dotenv";
import axios from "axios";

const { chatWithLLMStream: originalChatWithLLMStream } = ollamaLLM;

dotenv.config();

const OLLAMA_ENDPOINT = process.env.OLLAMA_ENDPOINT || "http://localhost:11434";
const EMBEDDING_MODEL_NAME = "mxbai-embed-large"; 

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
 * Force unload a specific model from Ollama memory
 */
const unloadModel = async (modelName: string) => {
  try {
    console.log(`[Ollama] Unloading model: ${modelName}...`);
    // Sending an empty request with keep_alive=0 unloads the model immediately
    await axios.post(`${OLLAMA_ENDPOINT}/api/chat`, {
      model: modelName,
      keep_alive: 0
    });
    console.log(`[Ollama] User unloaded model: ${modelName}`);
  } catch (error) {
    // It's okay if it fails (maybe model wasn't loaded), but we log it.
    // console.warn(`[Ollama] Failed to unload model ${modelName}:`, error);
  }
};

/**
 * Enhanced Chat Function with RAG
 * 1. Retrieves relevant context from LanceDB based on the user's latest message.
 * 2. Unloads embedding model to free up RAM/CPU.
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
  
  console.log(`[RAG] Searching context for: "${userQuery}"`);
  
  // 2. Retrieve Context
  let context = "";
  try {
    context = await getRelevantContext(userQuery, 1); // Reduced from 3 to 1 for optimization
    if (context) {
      console.log(`[RAG] Found context length: ${context.length}`);
    } else {
      console.log(`[RAG] No relevant context found.`);
    }
  } catch (err) {
    console.error(`[RAG] Error retrieving context:`, err);
  }

  // 3. OPTIMIZATION: Unload Embedding Model
  // await unloadModel(EMBEDDING_MODEL_NAME);

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
