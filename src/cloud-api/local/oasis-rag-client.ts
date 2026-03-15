/**
 * oasis-rag-client.ts
 *
 * HTTP client for the O.A.S.I.S. Python RAG Flask service (port 5001).
 *
 * The RAG service runs as a separate Python process:
 *   python python/oasis-rag/service.py
 *
 * Design contract:
 *  - All methods must resolve (never reject) so callers can use
 *    simple null/empty checks instead of try/catch.
 *  - On any network or timeout error: return null / empty string.
 *  - Timeout default: OASIS_RAG_TIMEOUT_MS env (fallback 5000 ms).
 *    5 seconds is enough for a warm retrieval (≈100–400 ms) while
 *    still failing fast in an emergency if the service is down.
 */

import axios, { AxiosError } from "axios";

// ── Config ──────────────────────────────────────────────────────────────────

const BASE_URL: string =
    process.env.OASIS_RAG_SERVICE_URL ?? "http://localhost:5001";

const TIMEOUT_MS: number =
    parseInt(process.env.OASIS_RAG_TIMEOUT_MS ?? "5000", 10);

// ── Types ────────────────────────────────────────────────────────────────────

export interface RagChunk {
    source:          string;
    section:         string;
    hybrid_score:    number;
    cosine_score:    number;
    lexical_score:   number;
    compressed_text: string;
}

export interface RagRetrieveResponse {
    context:           string;
    chunks:            RagChunk[];
    stage1_candidates: number;
    stage2_passing:    number;
    latency_ms:        number;
}

export interface RagHealthResponse {
    status:      "ok" | "degraded";
    index_ready: boolean;
    chunk_count: number;
    model:       string;
    error?:      string;
}

// ── Client ───────────────────────────────────────────────────────────────────

/**
 * Retrieve compressed medical context for a user query.
 *
 * @param query   The user's raw utterance / question.
 * @param topK    Optional override for number of chunks to return (default: 3).
 * @returns       Compressed context string, or empty string on failure.
 */
export async function ragRetrieve(
    query: string,
    topK?: number,
): Promise<string> {
    try {
        const body: Record<string, unknown> = { query };
        if (topK !== undefined) body.top_k = topK;

        const res = await axios.post<RagRetrieveResponse>(
            `${BASE_URL}/retrieve`,
            body,
            { timeout: TIMEOUT_MS },
        );

        const context = res.data?.context ?? "";
        const latency = res.data?.latency_ms ?? 0;

        console.log(
            `[RAG] Retrieved ${res.data?.chunks?.length ?? 0} chunks in ${latency.toFixed(1)} ms`,
        );

        return context;
    } catch (err) {
        _logError("ragRetrieve", err);
        return "";   // caller treats empty string as "no context available"
    }
}

/**
 * Retrieve full response object (for logging / debugging).
 * Returns null on failure.
 */
export async function ragRetrieveFull(
    query: string,
    topK?: number,
): Promise<RagRetrieveResponse | null> {
    try {
        const body: Record<string, unknown> = { query };
        if (topK !== undefined) body.top_k = topK;

        const res = await axios.post<RagRetrieveResponse>(
            `${BASE_URL}/retrieve`,
            body,
            { timeout: TIMEOUT_MS },
        );
        return res.data ?? null;
    } catch (err) {
        _logError("ragRetrieveFull", err);
        return null;
    }
}

/**
 * Check whether the RAG service is live and the index is ready.
 * Returns null on network failure.
 */
export async function ragHealth(): Promise<RagHealthResponse | null> {
    try {
        const res = await axios.get<RagHealthResponse>(
            `${BASE_URL}/health`,
            { timeout: TIMEOUT_MS },
        );
        return res.data ?? null;
    } catch (err) {
        _logError("ragHealth", err);
        return null;
    }
}

/**
 * Quick boolean check: is the RAG service up and index ready?
 */
export async function isRagReady(): Promise<boolean> {
    const h = await ragHealth();
    return h?.status === "ok" && h?.index_ready === true;
}

// ── Internal ─────────────────────────────────────────────────────────────────

function _logError(fn: string, err: unknown): void {
    if (axios.isAxiosError(err)) {
        const axErr = err as AxiosError;
        if (axErr.code === "ECONNREFUSED") {
            // Service is simply not running — expected during dev / fallback path
            console.warn(`[RAG] ${fn}: service not reachable (${BASE_URL})`);
        } else if (axErr.code === "ECONNABORTED") {
            console.warn(`[RAG] ${fn}: timeout after ${TIMEOUT_MS} ms`);
        } else {
            console.warn(`[RAG] ${fn}: ${axErr.message}`);
        }
    } else {
        console.warn(`[RAG] ${fn}: unexpected error`, err);
    }
}
