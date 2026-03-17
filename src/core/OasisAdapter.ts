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

import { ragRetrieve } from "../cloud-api/local/oasis-rag-client";

// ── System prompt template ───────────────────────────────────────────────────

/**
 * RAG-backed system prompt.
 * {context} is replaced with the compressed knowledge chunks from the RAG service.
 */
const RAG_SYSTEM_PROMPT_TEMPLATE = `You are OASIS, a critical first aid AI. A person needs first aid RIGHT NOW. Provide immediate, life-saving instructions based ONLY on the REFERENCE.

RULES YOU MUST STRICTLY FOLLOW:
- Output the required actions as a numbered list starting with "1.".
- Identify the specific injury or illness the user has. Extract steps ONLY for that specific condition. Completely IGNORE instructions for other body parts or conditions in the REFERENCE.
- Provide ONLY the necessary steps. Stop writing when the required steps for the specific injury are complete (maximum 7 steps). Do not fill up steps with unrelated information.
- If the REFERENCE states an absolute restriction (e.g., "Do not give water"), make it step "1.".
- Each step must be ONE short sentence, maximum 12 words.
- Use plain text only. No markdown, no bolding, no headers.
- Use direct command verbs (e.g., "Apply firm pressure", "Call emergency services").
- Start directly with step 1. No introductions, no greetings, no conclusions.

REFERENCE:
{context}

YOUR RESPONSE:`;
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
            return RAG_SYSTEM_PROMPT_TEMPLATE.replace("{context}", context);
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
