/**
 * OasisAdapter.ts
 *
 * Builds the system prompt for O.A.S.I.S. emergency responses.
 *
 * Priority chain:
 *   1. RAG service (Python Flask, port 5001)  — full Hybrid RAG context
 *   2. Local protocol matcher (oasis-matcher-node.ts) — hardcoded 30 protocols
 *   3. Empty string — ChatFlow uses its own default system prompt
 *
 * ChatFlow.ts is NOT modified; it calls getSystemPromptFromOasis(query)
 * and receives a ready-to-use system prompt string.
 */

import { ragRetrieve } from "../cloud-api/local/oasis-rag-client";
import { matchProtocolLocal } from "../cloud-api/local/oasis-matcher-node";

// ── System prompt template ───────────────────────────────────────────────────

/**
 * RAG-backed system prompt.
 * {context} is replaced with the compressed knowledge chunks from the RAG service.
 */
const RAG_SYSTEM_PROMPT_TEMPLATE = `You are OASIS, an offline first-aid assistant.
You respond ONLY based on the REFERENCE below.
Rules:
1. Maximum 5 numbered steps. Plain text only.
2. Each step under 15 words.
3. If supplies unavailable, suggest alternatives.
4. Never diagnose. Never prescribe medication.
5. If unsure: Call emergency services immediately.
6. If panicking: Start with 'Take a deep breath. I will guide you.'
7. Begin directly with step 1. No preambles, disclaimers, or introductions.
8. Answer in your own words. Never copy or reproduce the reference text.

REFERENCE:
{context}`;

/**
 * Fallback system prompt used when RAG has no context but the local
 * protocol matcher found a match.
 * {protocol} is replaced with the matched protocol text.
 */
const FALLBACK_SYSTEM_PROMPT = `You are O.A.S.I.S., an emergency first-aid assistant.
Respond with short, direct commands only.
No markdown, no bullet points, no numbering, no symbols.
Speak in simple sentences. One instruction at a time.
Do not explain. Do not reassure. Just tell them what to do.

ACTIVE PROTOCOL:
{protocol}`;

// ── Spinal injury detection ──────────────────────────────────────────────────

/** Keywords that suggest possible spinal cord injury. */
const SPINAL_SIGNALS: string[] = [
    "cannot feel", "can't feel", "cant feel",
    "paralyz", "numb legs", "numb feet", "numb limb",
    "neck injury", "spinal", "spine",
];

/**
 * Inject a spinal warning into RAG context when the query signals possible
 * spinal cord injury. The small on-device LLM follows in-context evidence
 * more reliably than abstract conditional rules.
 */
function injectSpinalWarning(context: string, query: string): string {
    const q = query.toLowerCase();
    if (SPINAL_SIGNALS.some(s => q.includes(s))) {
        return "CRITICAL: Possible spinal cord injury. Do not move the person.\n\n" + context;
    }
    return context;
}

// ── Main export ──────────────────────────────────────────────────────────────

/**
 * Build and return the system prompt for a given user query.
 *
 * Attempts RAG retrieval first; falls back to local protocol matcher on
 * any RAG failure (service down, timeout, empty context).
 *
 * @param query  The user's raw utterance from ASR.
 * @returns      System prompt string, or empty string if all sources fail.
 */
export const getSystemPromptFromOasis = async (query: string): Promise<string> => {

    // ── Stage 1: Try RAG service ─────────────────────────────────────────────
    try {
        const context = await ragRetrieve(query);

        if (context && context.trim().length > 0) {
            console.log("[OasisAdapter] Using RAG context");
            const enriched = injectSpinalWarning(context, query);
            return RAG_SYSTEM_PROMPT_TEMPLATE.replace("{context}", enriched);
        }

        console.log("[OasisAdapter] RAG returned empty context — falling back");
    } catch (err) {
        // ragRetrieve should never throw, but guard defensively
        console.warn("[OasisAdapter] RAG unexpected error — falling back:", err);
    }

    // ── Stage 2: Local protocol matcher fallback ─────────────────────────────
    try {
        const result = await matchProtocolLocal(query);

        if (result.match) {
            console.log(
                `[OasisAdapter] Fallback: protocol=${result.protocol_id}  score=${result.score.toFixed(3)}`,
            );
            return FALLBACK_SYSTEM_PROMPT.replace("{protocol}", result.text);
        }

        // Triage: no strong protocol match — return triage prompt
        console.log("[OasisAdapter] Fallback: triage (no protocol match)");
        return FALLBACK_SYSTEM_PROMPT.replace("{protocol}", result.text);

    } catch (err) {
        console.error("[OasisAdapter] Protocol matcher failed:", err);
    }

    // ── Stage 3: All sources failed ──────────────────────────────────────────
    console.error("[OasisAdapter] All retrieval sources failed. Returning empty prompt.");
    return "";
};
