// NOTE: qdrant-vectordb removed — OASIS uses Python Flask RAG service (oasis-rag-client.ts)
import { summaryTextWithLLM } from "./llm";

const enableRAG = (process.env.ENABLE_RAG || "false").toLowerCase() === "true";

const vectorDB: any = null;
const embedText: (text: string) => Promise<number[]> = null as any;

export { vectorDB, embedText, summaryTextWithLLM, enableRAG };
