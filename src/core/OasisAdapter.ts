/**
 * OasisAdapter.ts
 *
 * Retrieves the system prompt from the RAG service for O.A.S.I.S. emergency responses.
 *
 * The prompt template is defined in python/oasis-rag/prompt.py (single source of truth).
 * The RAG service builds the full system prompt and returns it via /retrieve → system_prompt.
 *
 * Priority chain:
 *   1. RAG service (Python Flask, port 5001) — returns ready-to-use system prompt
 *   2. Safe fallback — directs the user to emergency services
 */

import * as fs from "fs";
import * as path from "path";
import moment from "moment";
import { ragRetrieve } from "../cloud-api/local/oasis-rag-client";
import { oasisLogDir } from "../utils/dir";

// ── Safe fallback (RAG unavailable) ─────────────────────────────────────────

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
 * Get the system prompt for a given user query.
 *
 * Calls the RAG service which returns a ready-to-use system prompt
 * (template defined in python/oasis-rag/prompt.py).
 * Falls back to a safe prompt if the service is unavailable.
 *
 * @param query  The user's raw utterance from ASR.
 * @returns      System prompt string (never empty).
 */
export const getSystemPromptFromOasis = async (query: string): Promise<string> => {

    // ── Try RAG service ──────────────────────────────────────────────────────
    try {
        const systemPrompt = await ragRetrieve(query);

        if (systemPrompt && systemPrompt.trim().length > 0) {
            console.log("[OasisAdapter] Using RAG system prompt");
            return systemPrompt;
        }

        console.warn("[OasisAdapter] RAG returned empty prompt — using safe fallback");
    } catch (err) {
        console.warn("[OasisAdapter] RAG unexpected error — using safe fallback:", err);
    }

    // ── Safe fallback (RAG unavailable) ──────────────────────────────────────
    console.warn("[OasisAdapter] RAG unavailable. Directing user to emergency services.");
    return SAFE_FALLBACK_PROMPT;
};
