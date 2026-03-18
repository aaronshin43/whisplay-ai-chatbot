# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Critical Rules

- **All responses and code comments must be written in English.**
- **Never generate medical advice directly** — only reference RAG-validated manuals from `data/knowledge/`.
- **`context_injector.py` is the single source of truth** for emergency signal injection — never duplicate signal logic elsewhere.
- **Pi5 memory budget: total pipeline ≤ 4.5 GB** — do not introduce models or dependencies that exceed this.
- **Do not modify `data/rag_index/`** — it is a build artifact; regenerate via `bash index_knowledge.sh`.
- **Validation must stay 109/109 PASS** — run `python validation/run_all.py` after any RAG change.

---

## Architecture

**Stack:** Whisper STT → Python RAG Flask (:5001, FAISS + gte-small) → gemma3:1b (Ollama) → Piper TTS, running on Node.js + TypeScript chatbot layer. Hardware: Raspberry Pi 5 + Whisplay HAT.

**Key entry points:**

| File | Role |
|------|------|
| `src/core/ChatFlow.ts` | Main loop: button → STT → RAG → LLM → TTS |
| `src/core/OasisAdapter.ts` | RAG result → LLM system prompt (3-tier fallback) |
| `src/cloud-api/server.ts` | ASR/LLM/TTS provider router (reads `.env`) |
| `src/cloud-api/local/oasis-rag-client.ts` | RAG HTTP client (primary path) |
| `src/cloud-api/local/oasis-matcher-node.ts` | Embedded fallback matcher (no server needed) |
| `python/oasis-rag/service.py` | Flask RAG server |
| `python/oasis-rag/retriever.py` | 3-Stage Hybrid Retriever |
| `python/oasis-rag/context_injector.py` | 22-signal context injection (single source) |

**RAG pipeline:** Query → QueryClassifier → Stage 1 lexical filter (top-50) → Stage 2 hybrid rerank (cosine 0.6 + BM25 0.4, top-4) → Stage 3 compression → context_injector → OasisAdapter → LLM

> For pipeline internals, Context Injection signal list, Flask API reference, and knowledge base document format → `docs/architecture.md`

---

## Build/Test

```bash
# Services (start in order)
ollama serve
cd python/oasis-rag && python app.py   # RAG Flask (:5001)
npm start

# Build
bash build.sh                          # TypeScript: rm -rf dist && tsc
bash index_knowledge.sh                # Rebuild FAISS index after KB changes

# Test
cd python/oasis-rag && python validation/run_all.py   # 109 tests, no Flask needed
yarn test:oasis                                        # TypeScript integration
```

> For individual test suite commands, test IDs, and SAFE-check authoring patterns → `docs/testing.md`

---

## Domain Context

O.A.S.I.S. (Offline AI Survival & first-aid kIt System) — Whisplay chatbot fork implementing 3-Stage Hybrid RAG (Pocket RAG paper) for fully offline emergency first-aid on Raspberry Pi 5.

**Non-obvious terms:**
- **Context Injection** — protocol directives prepended to RAG context so the LLM reads them before retrieved chunks (`context_injector.py`)
- **3-tier fallback** — RAG Flask → embedded matcher (`oasis-matcher-node.ts`) → empty prompt

**Targets:** RAG latency PC < 200ms · Pi5 < 2000ms · memory ≤ 4.5 GB · validation 109/109

---

## Reference Docs

Load only the relevant doc — do not read all by default.

| File | Read when... |
|------|-------------|
| `docs/architecture.md` | Modifying RAG stages, Context Injection signals, Flask API, KB document format, or LLM prompt |
| `docs/testing.md` | Writing/debugging tests; looking up test IDs; authoring SAFE checks |
| `docs/roadmap.md` | Investigating known bugs (BUG-001 lightning), planning Phase 2–6, choosing LLM upgrade |
| `docs/decisions.md` | Before changing embedding model, vector DB, chunking strategy, or LLM |

---

## Coding Conventions

- **All code comments and commit messages must be in English.**
- TypeScript: `camelCase` for variables/functions, `PascalCase` for classes and types.
- Python: `snake_case`; follow existing module structure in `python/oasis-rag/`.
- Commit format: `type: short description` (e.g., `feat:`, `fix:`, `refactor:`, `docs:`).
- New ASR/LLM/TTS providers go in `src/cloud-api/local/` or a named subfolder — never in `server.ts` directly.
