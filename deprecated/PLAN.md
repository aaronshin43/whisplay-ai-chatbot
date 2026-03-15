# RAG Implementation Plan for O.A.S.I.S.

This document outlines the step-by-step plan to implement the Retrieval-Augmented Generation (RAG) pipeline for the O.A.S.I.S. (Offline AI Survival & First-aid Kit) project.

## 1. Goal
To build a highly reliable, offline RAG system that retrieves accurate first-aid instructions from local medical manuals and guides the LLM to provide precise, actionable advice in emergency situations.

## 2. Technology Stack (from PROJECT_CONTEXT.md)
- **Framework:** LangChain (Node.js)
- **Vector Database:** LanceDB (Embedded, file-based, no separate server process needed)
- **Embedding Model:** Ollama (using the same local model or a specialized embedding model like `mxbai-embed-large`)
- **Data Source Format:** Markdown (`.md`) or Text (`.txt`) - *Reasoning below*

## 3. Data Format Strategy
We recommend converting PDF manuals into **Markdown (`.md`)** format before indexing.
*   **Why Markdown?**
    *   **Structure:** First-aid manuals rely heavily on headings (e.g., "# CPR Steps", "## Adult", "## Infant"). Markdown preserves this hierarchy, allowing for "Context-Aware Chunking" (splitting text by headers).
    *   **Cleanliness:** PDF text extraction often results in garbage characters, headers/footers interrupting sentences, and loss of list structures.
    *   **LLM Friendliness:** LLMs understand Markdown syntax very well.
*   **Action:** We will assume the knowledge base consists of clean Markdown files in `knowledge/`.

## 4. Implementation Steps

### Phase 1: Dependency & Setup
1.  **Install Dependencies:**
    *   `npm install @langchain/core @langchain/community @langchain/ollama lancedb @lancedb/lancedb`
2.  **Environment Variables:**
    *   Update `.env` to include RAG configurations (keeping existing keys where possible, adding new ones for LangChain).

### Phase 2: Core Logic Implementation (`src/core/rag-pipeline.ts`)
We will create a new module `src/core/rag-pipeline.ts` to encapsulate the LangChain logic, keeping it separate from the old `Knowledge.ts` initially.

1.  **Loading & Chunking:**
    *   Use `DirectoryLoader` to load `.md` files.
    *   Use `MarkdownHeaderTextSplitter` to split documents by sections (e.g., `#`, `##`, `###`). This ensures a search for "CPR" retrieves the whole CPR section, not just a random sentence.
2.  **Vector Store (LanceDB):**
    *   Initialize LanceDB locally in the `data/lancedb` directory.
    *   Implement `indexKnowledgeBase()` function: Reads files -> Chunks -> Embeds via Ollama -> Stores in LanceDB.
3.  **Retrieval Chain:**
    *   Implement `getRelevantContext(query)`: Searches LanceDB for the top 3-5 most relevant chunks.

### Phase 3: System Prompt Engineering (`src/config/prompts.ts`)
We need a valid, safety-critical system prompt.

*   **Role:** Emergency Medical Responder Assistant.
*   **Tone:** Calm, authoritative, concise, actionable.
*   **Safety Hooks:** "If unsure, advise calling professional help immediately (if available) or stabilizing the patient."
*   **Context usage:** "Answer strictly based on the provided context. Do not hallucinate medical procedures."

**Draft Prompt:**
```text
You are O.A.S.I.S., an emergency first-aid assistant. You are operating offline in a potential crisis situation.
Your goal is to provide immediate, step-by-step first aid instructions based ONLY on the context provided below.

Context:
{context}

Question:
{question}

Instructions:
1. Be concise. Use short sentences (bullet points preferred).
2. Prioritize life-saving actions (Airway, Breathing, Circulation).
3. If the context does not contain the answer, state "I do not have information on this specific injury in my database. Stabilize the patient and seek professional help."
4. Do NOT make up medical advice not found in the context.
```

### Phase 4: Integration with ChatFlow (`src/core/ChatFlow.ts`)
1.  **Refactor `ChatFlow.ts`:**
    *   Import the new RAG pipeline.
    *   Before calling `chatWithLLMStream`, retrieve context.
    *   Inject the retrieved context into the System Message (replacing the generic prompt).
2.  **Performance Check:** Measure the latency added by the retrieval step.

## 5. Migration from Existing Code
*   **`src/core/Knowledge.ts`**: This file currently handles simple file reading and Qdrant interaction. We will deprecate its Qdrant logic but reuse its utility functions for file path management if applicable.
*   **`src/cloud-api/knowledge.ts`**: This will be superseded by the new LangChain/LanceDB implementation. We can keep it as a fallback or remove it later.
*   **`src/utils/knowledge.ts`**: The manual chunking logic here will be replaced by LangChain's smart splitters.

## 6. Verification Plan
1.  **Unit Test (`src/test/rag-test.ts`):**
    *   Index a sample "CPR Guide.md".
    *   Query "How to do CPR on an adult?"
    *   Verify that the retrieved chunk contains the correct section.
2.  **End-to-End Test:**
    *   Run the chatbot (text mode).
    *   Ask a medical question.
    *   Verify the response cites the manual.

