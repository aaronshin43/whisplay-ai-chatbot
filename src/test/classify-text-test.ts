/**
 * classify-text-test.ts
 *
 * Text-in / text-out end-to-end test for the oasis-classify pipeline.
 * Bypasses all hardware (STT, TTS, button, display).
 *
 * Usage:
 *   Single query:
 *     npm run test:classify -- "someone is not breathing"
 *
 *   Interactive (REPL loop, triage hint carry-forward across turns):
 *     npm run test:classify
 */

import * as readline from "readline";
import dotenv from "dotenv";
import { dispatchQuery } from "../core/OasisAdapter";
import { chatWithLLMStream, resetChatHistory } from "../cloud-api/llm";
import { Message } from "../type";

dotenv.config();

const TRIAGE_HINT_TTL_MS = 60_000;

let triageHint: { category: string; expiresAt: number } | null = null;

function getActiveTriageHint(): string | null {
    if (!triageHint) return null;
    if (Date.now() > triageHint.expiresAt) {
        triageHint = null;
        return null;
    }
    return triageHint.category;
}

async function runQuery(query: string): Promise<void> {
    const hint = getActiveTriageHint();
    console.log(`\n${"─".repeat(60)}`);
    console.log(`[Query]      ${query}`);
    if (hint) console.log(`[TriageHint] ${hint} (TTL: ${Math.round((triageHint!.expiresAt - Date.now()) / 1000)}s remaining)`);

    const t0 = Date.now();
    const dispatch = await dispatchQuery(query, hint);
    const dispatchMs = Date.now() - t0;

    console.log(`[Dispatch]   mode=${dispatch.mode}  path=${dispatch.raw.threshold_path}  score=${dispatch.raw.score?.toFixed(3) ?? "—"}  latency=${dispatch.raw.latency_ms.toFixed(1)}ms  (client: ${dispatchMs}ms)`);
    if (dispatch.raw.top3.length > 0) {
        const top3str = dispatch.raw.top3.map(e => `${e.category}:${e.score.toFixed(3)}`).join("  ");
        console.log(`[Top3]       ${top3str}`);
    }
    if (dispatch.raw.hint_changed_result) {
        console.log(`[HintEffect] triage hint changed the top-1 result`);
    }

    // ── Handle mode ──────────────────────────────────────────────────────────

    if (dispatch.mode === "direct_response" || dispatch.mode === "ood_response") {
        triageHint = null;
        console.log(`\n[Response]`);
        console.log(dispatch.responseText);
        return;
    }

    // llm_prompt or triage_prompt → call LLM
    if (dispatch.mode === "llm_prompt") {
        triageHint = null;
    } else {
        // triage_prompt: store hint
        triageHint = { category: dispatch.triageHint, expiresAt: Date.now() + TRIAGE_HINT_TTL_MS };
        console.log(`[HintSet]    storing triage hint: ${dispatch.triageHint}`);
    }

    const messages: Message[] = [
        { role: "system", content: dispatch.systemPrompt },
        { role: "user",   content: query },
    ];

    resetChatHistory();

    console.log(`\n[LLM Response]`);
    let firstToken = 0;
    const llmStart = Date.now();

    await new Promise<void>((resolve, reject) => {
        chatWithLLMStream(
            messages,
            (partial) => {
                if (!firstToken) {
                    firstToken = Date.now();
                    process.stdout.write(`[TTFT: ${firstToken - llmStart}ms]\n`);
                }
                process.stdout.write(partial);
            },
            () => {
                const total = Date.now() - llmStart;
                console.log(`\n[LLM total: ${total}ms]`);
                resolve();
            },
            () => { /* ignore thinking tokens */ },
        );
    });
}

async function interactive(): Promise<void> {
    const rl = readline.createInterface({
        input:  process.stdin,
        output: process.stdout,
        prompt: "\n> ",
    });

    console.log("oasis-classify text test — interactive mode");
    console.log("Type a query and press Enter. Empty line to quit.");
    console.log("Triage hint is carried forward across turns (60s TTL).");

    rl.prompt();

    rl.on("line", async (line) => {
        const query = line.trim();
        if (!query) {
            rl.close();
            return;
        }
        rl.pause();
        try {
            await runQuery(query);
        } catch (err) {
            console.error("[Error]", err);
        }
        rl.resume();
        rl.prompt();
    });

    rl.on("close", () => {
        console.log("\nBye.");
        process.exit(0);
    });
}

// ── Entry point ──────────────────────────────────────────────────────────────

const singleQuery = process.argv[2];

if (singleQuery) {
    runQuery(singleQuery)
        .then(() => process.exit(0))
        .catch((err) => { console.error(err); process.exit(1); });
} else {
    interactive();
}
