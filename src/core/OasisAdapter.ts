/**
 * OasisAdapter.ts
 *
 * Builds the system prompt for O.A.S.I.S. emergency responses.
 *
 * Priority chain:
 *   1. RAG service (Python Flask, port 5001)  — full Hybrid RAG context
 *   2. Safe fallback — instructs the LLM to direct the user to emergency services
 *      without providing any unverified medical content
 *
 * ChatFlow.ts is NOT modified; it calls getSystemPromptFromOasis(query)
 * and receives a ready-to-use system prompt string.
 */

import * as fs from "fs";
import * as path from "path";
import moment from "moment";
import { ragRetrieve } from "../cloud-api/local/oasis-rag-client";
import { oasisLogDir } from "../utils/dir";

// ── Markdown stripping ──────────────────────────────────────────────────────

/**
 * Strip markdown formatting from RAG context so the 1B LLM doesn't copy it.
 * Preserves the textual content, numbers, and structure.
 */
const stripMarkdown = (text: string): string =>
    text
        .replace(/^#{1,6}\s+/gm, "")          // ### headings → plain text
        .replace(/\*\*(.+?)\*\*/g, "$1")       // **bold** → bold
        .replace(/\*(.+?)\*/g, "$1")           // *italic* → italic
        .replace(/^[ \t]*[-*]\s+/gm, "- ")     // normalise bullets to "- "
        .replace(/\|[^\n]+\|/g, "")            // remove markdown tables
        .replace(/^---+$/gm, "")               // remove horizontal rules
        .replace(/\n{3,}/g, "\n\n");            // collapse extra blank lines

// ── System prompt template ───────────────────────────────────────────────────

/**
 * RAG-backed system prompt.
 * {context} is replaced with the compressed knowledge chunks from the RAG service.
 */
const RAG_SYSTEM_PROMPT_TEMPLATE = `You are OASIS, an emergency first aid assistant.

FORMAT RULES (follow exactly):
1. Identify the specific injury in the TASK. Extract steps ONLY for that condition. IGNORE unrelated conditions in the REFERENCE.
2. Numbered list only: 1. 2. 3. (max 7 steps, one sentence each).
3. Flatten any sub-bullets into the numbered step.
4. Keep exact numbers (depths, rates, ratios) from REFERENCE.
5. If REFERENCE says "Do not...", that MUST be step 1.
6. No markdown, no headers, no trailing text after the last step.
7. Use ONLY the REFERENCE below. Do NOT add any outside information.

EXAMPLE:
Reference: 
Care for Rib Fracture: Do not wrap a band tightly. 
Care for Broken Finger: Tape the broken finger to adjacent uninjured fingers with padding.
Task: Write numbered first aid steps for this emergency: I broke my finger
Response:
1. Tape the broken finger to adjacent uninjured fingers with padding.

REFERENCE:
{context}

TASK: Write numbered first aid steps for this emergency: {query}
RESPONSE:`;



/**
 * Safe fallback prompt used when the RAG service is unavailable.
 * Does NOT include any hardcoded medical content — only directs to emergency services.
 * This avoids the risk of providing unverified first-aid advice.
 */
const SAFE_FALLBACK_PROMPT = `You are OASIS, an offline first-aid assistant.
The medical knowledge base is currently unavailable.
Tell the user clearly and calmly:
1. Call emergency services immediately (local emergency number).
2. Stay on the line with the dispatcher — they will guide you.
3. Do not leave the person alone.
Do not provide any specific medical instructions without the knowledge base.`;

// ── Streaming chunk sanitizer ────────────────────────────────────────────────

/**
 * Strip markdown formatting characters from LLM streaming token chunks.
 * Applied to every partial token before it reaches TTS so asterisks, hashes,
 * and backticks are never spoken aloud by the TTS engine.
 */
export const sanitizeOasisChunk = (chunk: string): string =>
    chunk
        .replace(/\*+/g, "")       // **bold** / *italic* markers
        .replace(/`+/g, "")        // `code` backticks
        .replace(/^#+\s*/gm, "");  // ### headings (safe in chunks — only matches at line start)

// ── Response logger ───────────────────────────────────────────────────────────

/**
 * Write a structured JSONL entry to data/oasis_logs/ for every completed OASIS response.
 * Allows offline analysis of hallucination patterns and format compliance.
 */
export const logOasisResponse = (query: string, response: string): void => {
    try {
        const entry = JSON.stringify({
            ts: moment().toISOString(),
            query,
            response,
            steps: (response.match(/^\d+\./gm) ?? []).length,
            has_markdown: /[*_#`]/.test(response),
        });
        const logFile = path.join(oasisLogDir, `oasis_${moment().format("YYYY-MM-DD")}.jsonl`);
        fs.appendFileSync(logFile, entry + "\n", "utf8");
    } catch {
        // logging failure must never affect the response path
    }
};

// ── Main export ──────────────────────────────────────────────────────────────

/**
 * Build and return the system prompt for a given user query.
 *
 * Attempts RAG retrieval first. If the RAG service is down or returns
 * empty context, returns a safe fallback prompt that directs to emergency
 * services — without any hardcoded medical content.
 *
 * @param query  The user's raw utterance from ASR.
 * @returns      System prompt string (never empty).
 */
export const getSystemPromptFromOasis = async (query: string): Promise<string> => {

    // ── Stage 1: Try RAG service ─────────────────────────────────────────────
    try {
        const context = await ragRetrieve(query);

        if (context && context.trim().length > 0) {
            console.log("[OasisAdapter] Using RAG context");
            // Context injection is applied server-side in context_injector.py
            return RAG_SYSTEM_PROMPT_TEMPLATE
                .replace("{context}", stripMarkdown(context))
                .replace("{query}", query);
        }

        console.warn("[OasisAdapter] RAG returned empty context — using safe fallback");
    } catch (err) {
        // ragRetrieve should never throw, but guard defensively
        console.warn("[OasisAdapter] RAG unexpected error — using safe fallback:", err);
    }

    // ── Stage 2: Safe fallback (RAG unavailable) ─────────────────────────────
    console.warn("[OasisAdapter] RAG unavailable. Directing user to emergency services.");
    return SAFE_FALLBACK_PROMPT;
};
