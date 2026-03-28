/**
 * oasis-classify-client.ts
 *
 * HTTP client for the O.A.S.I.S. Python classify service (port 5002).
 *
 * The classify service runs as a separate Python process:
 *   python python/oasis-classify/service.py
 *
 * Design contract:
 *  - All methods must resolve (never reject) so callers can use
 *    simple null/empty checks instead of try/catch.
 *  - On any network or timeout error: return a safe fallback DispatchResult.
 *  - Timeout default: OASIS_CLASSIFY_TIMEOUT_MS env (fallback 3000 ms).
 *    3 seconds matches the Pi5 latency target for the classify pipeline.
 */

import axios, { AxiosError } from "axios";

// ── Config ──────────────────────────────────────────────────────────────────

const BASE_URL: string =
    process.env.OASIS_CLASSIFY_SERVICE_URL ?? "http://localhost:5002";

const TIMEOUT_MS: number =
    parseInt(process.env.OASIS_CLASSIFY_TIMEOUT_MS ?? "3000", 10);

// ── Types ────────────────────────────────────────────────────────────────────

export type DispatchMode = "direct_response" | "llm_prompt" | "triage_prompt" | "ood_response";

export interface Top3Entry {
    category: string;
    score:    number;
}

export interface DispatchResult {
    mode:                DispatchMode;
    response_text:       string | null;
    system_prompt:       string | null;
    category:            string | null;   // always set for triage_prompt
    top3:                Top3Entry[];
    score:               number | null;
    threshold_path:      string;
    latency_ms:          number;
    hint_changed_result: boolean;
}

export interface ClassifyHealthResponse {
    status: "ok" | "degraded";
    model:  string;
    error?: string;
}

// ── Client-side threshold_path constants ─────────────────────────────────────

/** Synthesized by the client when the service cannot be reached. */
export const THRESHOLD_PATH_NETWORK_ERROR  = "network_error";

/** Synthesized by the client when the service returns a non-2xx status. */
export const THRESHOLD_PATH_SERVICE_ERROR  = "service_error";

/** Synthesized by the client when the response does not match the expected schema. */
export const THRESHOLD_PATH_INVALID_SCHEMA = "invalid_schema";

// ── Fallback response ────────────────────────────────────────────────────────

const FALLBACK_RESPONSE_TEXT =
    "I am a first-aid assistant. I could not process your request right now. " +
    "If this is an emergency, please call your local emergency services immediately.";

function _makeFallback(thresholdPath: string): DispatchResult {
    return {
        mode:                "ood_response",
        response_text:       FALLBACK_RESPONSE_TEXT,
        system_prompt:       null,
        category:            null,
        top3:                [],
        score:               null,
        threshold_path:      thresholdPath,
        latency_ms:          0,
        hint_changed_result: false,
    };
}

// ── Client ───────────────────────────────────────────────────────────────────

/**
 * Send a query to the classify service and receive a DispatchResult.
 *
 * Always resolves. On any failure a safe fallback DispatchResult is returned
 * with mode="ood_response" and the appropriate client-side threshold_path.
 *
 * @param query           The user's raw utterance from ASR.
 * @param prevTriageHint  Category string from the previous triage response, or null.
 * @returns               A DispatchResult — never throws.
 */
export async function dispatch(
    query: string,
    prevTriageHint: string | null,
): Promise<DispatchResult> {
    try {
        const body: Record<string, unknown> = { query };
        if (prevTriageHint !== null) body.prev_triage_hint = prevTriageHint;

        const res = await axios.post<DispatchResult>(
            `${BASE_URL}/dispatch`,
            body,
            { timeout: TIMEOUT_MS },
        );

        const data = res.data;

        // Minimal schema check — mode must be one of the four known values
        if (!data || !["direct_response", "llm_prompt", "triage_prompt", "ood_response"].includes(data.mode)) {
            console.warn("[Classify] dispatch: unexpected response schema", data);
            return _makeFallback(THRESHOLD_PATH_INVALID_SCHEMA);
        }

        console.log(
            `[Classify] dispatch mode=${data.mode} category=${data.category ?? "null"} ` +
            `score=${data.score?.toFixed(2) ?? "null"} path=${data.threshold_path} ` +
            `latency=${data.latency_ms.toFixed(1)}ms hint_changed=${data.hint_changed_result}`,
        );

        return data;
    } catch (err) {
        const thresholdPath = _classifyError("dispatch", err);
        return _makeFallback(thresholdPath);
    }
}

/**
 * Check whether the classify service is live.
 * Returns null on network failure.
 */
export async function classifyHealth(): Promise<ClassifyHealthResponse | null> {
    try {
        const res = await axios.get<ClassifyHealthResponse>(
            `${BASE_URL}/health`,
            { timeout: TIMEOUT_MS },
        );
        return res.data ?? null;
    } catch (err) {
        _classifyError("classifyHealth", err);
        return null;
    }
}

/**
 * Quick boolean check: is the classify service up and ready?
 */
export async function isClassifyReady(): Promise<boolean> {
    const h = await classifyHealth();
    return h?.status === "ok";
}

// ── Internal ─────────────────────────────────────────────────────────────────

/**
 * Log an error and return the appropriate client-side threshold_path constant.
 */
function _classifyError(fn: string, err: unknown): string {
    if (axios.isAxiosError(err)) {
        const axErr = err as AxiosError;
        if (axErr.code === "ECONNREFUSED") {
            console.warn(`[Classify] ${fn}: service not reachable (${BASE_URL})`);
            return THRESHOLD_PATH_NETWORK_ERROR;
        } else if (axErr.code === "ECONNABORTED") {
            console.warn(`[Classify] ${fn}: timeout after ${TIMEOUT_MS} ms`);
            return THRESHOLD_PATH_NETWORK_ERROR;
        } else if (axErr.response) {
            console.warn(`[Classify] ${fn}: HTTP ${axErr.response.status} — ${axErr.message}`);
            return THRESHOLD_PATH_SERVICE_ERROR;
        } else {
            console.warn(`[Classify] ${fn}: ${axErr.message}`);
            return THRESHOLD_PATH_NETWORK_ERROR;
        }
    } else {
        console.warn(`[Classify] ${fn}: unexpected error`, err);
        return THRESHOLD_PATH_SERVICE_ERROR;
    }
}
