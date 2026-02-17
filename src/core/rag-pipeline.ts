
import path from "path";
import fs from "fs";
import { connect } from "@lancedb/lancedb";
import { OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import dotenv from "dotenv";

dotenv.config();

const LANCED_DB_PATH = path.join(process.cwd(), "data", "lancedb");
const TABLE_NAME = "knowledge_store";

// Initialize Ollama Embeddings
const embeddings = new OllamaEmbeddings({
  model: "mxbai-embed-large", // Using the model you tested, or change to a dedicated embedding model like "mxbai-embed-large"
  baseUrl: process.env.OLLAMA_ENDPOINT || "http://localhost:11434",
});

export const indexKnowledgeBase = async () => {
  const knowledgeDir = path.join(process.cwd(), "data", "knowledge");
  
  if (!fs.existsSync(knowledgeDir)) {
    console.error(`Knowledge directory not found: ${knowledgeDir}`);
    return;
  }

  const files = fs.readdirSync(knowledgeDir).filter(f => f.endsWith(".md") || f.endsWith(".txt"));
  
  if (files.length === 0) {
    console.log("No knowledge files found.");
    return;
  }

  console.log(`Found ${files.length} knowledge files.`);

  // Load and chunk documents
  let docs: Document[] = [];
  
  // Strategy: Semantic chunking using Markdown headers if possible, 
  // but RecursiveCharacterTextSplitter is robust for general text/md.
  // We use a large chunk size to capture full context of a procedure.
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
    separators: ["\n## ", "\n### ", "\n#### ", "\n", " ", ""], // Prioritize headers
  });

  for (const file of files) {
    const filePath = path.join(knowledgeDir, file);
    const content = fs.readFileSync(filePath, "utf-8");
    
    // Create a base document
    const rawDocs = [new Document({
      pageContent: content,
      metadata: { source: file }
    })];

    const splitDocs = await splitter.splitDocuments(rawDocs);
    docs.push(...splitDocs);
    console.log(`Processed ${file}: ${splitDocs.length} chunks`);
  }

  // Connect to LanceDB
  const db = await connect(LANCED_DB_PATH);
  
  // Re-create table (Full refresh strategy for simplicity on update)
  // In production, you might want to check existing hashes.
  try {
    await db.openTable(TABLE_NAME);
    await db.dropTable(TABLE_NAME);
    console.log("Dropped existing table for refresh.");
  } catch (e) {
    // Table might not exist, ignore
  }

  // Generate embeddings and store
  console.log("Generating embeddings and indexing...");
  
  // LanceDB interacting with LangChain often requires manual embedding handling or specific wrappers.
  // Here we will use a simplified approach: Embed -> Format for LanceDB -> Add
  
  const data = [];
  for (const doc of docs) {
    const vector = await embeddings.embedQuery(doc.pageContent);
    data.push({
      vector: vector,
      text: doc.pageContent,
      source: doc.metadata.source
    });
  }

  await db.createTable(TABLE_NAME, data);
  console.log(`Successfully indexed ${data.length} chunks into LanceDB.`);
};

export const getRelevantContext = async (query: string, limit: number = 3): Promise<string> => {
  try {
    const db = await connect(LANCED_DB_PATH);
    const table = await db.openTable(TABLE_NAME);

    const queryVector = await embeddings.embedQuery(query);
    const results = await table.vectorSearch(queryVector)
      .limit(limit)
      .toArray();

    if (results.length === 0) return "";

    // Format findings
    return results.map((r: any) => 
      `---\nSource: ${r.source}\nContent: ${r.text}\n---`
    ).join("\n");

  } catch (error) {
    console.error("Error retrieving context:", error);
    return "";
  }
};
