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
const RAG_SYSTEM_PROMPT_TEMPLATE = `You are OASIS. A person needs first aid RIGHT NOW.

RULES YOU MUST FOLLOW:
- Your response is ONLY numbered steps 1 through 5.
- Do NOT write anything before "1."
- Do NOT write anything after step 5.
- Each step is ONE sentence, maximum 12 words.
- Do NOT use asterisks, bold, markdown, or headers.
- Do NOT ask questions. Give commands only.
- Do NOT say "Okay" or "Let's" or any introduction.

REFERENCE:
{context}

YOUR RESPONSE MUST START WITH "1." AND END AFTER STEP 5. NOTHING ELSE.`;

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

// ── Seizure / Convulsion detection ───────────────────────────────────────────

/** Keywords that indicate active or recent seizure/convulsion. */
const SEIZURE_SIGNALS: string[] = [
    "seizure", "convuls", "shaking", "fitting", "fit ",
    "jerking", "twitching", "epilep",
    "발작", "경련",
];

/**
 * Inject a seizure safety notice into RAG context when the query signals
 * active convulsions. Prioritises bystander safety actions over ABCDE
 * assessment — the LLM must not instruct pulse/breathing checks mid-seizure.
 */
function injectSeizureWarning(context: string, query: string): string {
    const q = query.toLowerCase();
    if (SEIZURE_SIGNALS.some(s => q.includes(s))) {
        return (
            "SEIZURE PRIORITY OVERRIDE: If the patient is having active convulsions, " +
            "prioritize safety (clearing area, head protection) over ABCDE assessment " +
            "until the shaking stops. Do NOT restrain. Do NOT put anything in the mouth. " +
            "Do NOT attempt pulse or breathing checks during convulsions.\n\n" +
            context
        );
    }
    return context;
}

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
            let enriched = injectSpinalWarning(context, query);
            enriched = injectSeizureWarning(enriched, query);
            return RAG_SYSTEM_PROMPT_TEMPLATE.replace("{context}", enriched);
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
