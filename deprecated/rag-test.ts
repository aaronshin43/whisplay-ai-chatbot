
import { indexKnowledgeBase, getRelevantContext } from "../core/rag-pipeline";

const run = async () => {
  const args = process.argv.slice(2);
  const command = args[0];

  if (command === "index") {
    console.log("Starting knowledge base indexing...");
    const startTime = Date.now();
    await indexKnowledgeBase();
    console.log(`Indexing completed in ${(Date.now() - startTime) / 1000}s`);
  } else if (command === "search") {
    const query = args[1];
    if (!query) {
      console.error("Please provide a search query.");
      return;
    }
    console.log(`Searching for: "${query}"...`);
    const startTime = Date.now();
    const result = await getRelevantContext(query);
    console.log(`Search completed in ${(Date.now() - startTime) / 1000}s`);
    console.log("\n[Search Results]:");
    console.log(result || "No relevant context found.");
  } else {
    console.log("Usage:");
    console.log("  npm run rag:index         - Index the knowledge base");
    console.log("  npm run rag:search \"query\" - Search the knowledge base");
  }
};

run().catch(console.error);
