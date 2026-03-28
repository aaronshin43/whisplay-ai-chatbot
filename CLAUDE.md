# CLAUDE.md

This file provides top-level guidance for Claude Code when working in this repository.
For subsystem-specific rules, navigate to the relevant subfolder — each has its own CLAUDE.md.

---

## Critical Rules

- **All code, comments, and commit messages must be written in English.**
- **Never generate medical advice directly** — only reference validated manuals from `data/knowledge/`.
- **Pi5 memory budget: total pipeline ≤ 4.5 GB** — do not introduce models or dependencies that exceed this.
- **Do not modify `data/rag_index/`** — it is a build artifact; regenerate via `bash index_knowledge.sh`.

---

## Project Overview

**O.A.S.I.S.** (Offline AI Survival & first-aid kIt System) — fully offline emergency first-aid assistant running on Raspberry Pi 5 + Whisplay HAT.

**Stack:** Whisper STT → Python backend (:5001 / :5002) → gemma3:1b (Ollama) → Piper TTS, orchestrated by a Node.js + TypeScript chatbot layer.

**Two independent Python backends — read the relevant CLAUDE.md before working in either:**

| Folder | Purpose | Port | CLAUDE.md |
|--------|---------|------|-----------|
| `python/oasis-rag/` | 3-Stage Hybrid RAG pipeline (FAISS + gte-small) | :5001 | [`python/oasis-rag/CLAUDE.md`](python/oasis-rag/CLAUDE.md) |
| `python/oasis-classify/` | Medical intent classifier + pre-generated manual dispatch | :5002 | [`python/oasis-classify/CLAUDE.md`](python/oasis-classify/CLAUDE.md) |

---

## TypeScript Layer Entry Points

| File | Role |
|------|------|
| `src/core/ChatFlow.ts` | Main loop: button → STT → backend → LLM → TTS |
| `src/core/OasisAdapter.ts` | Backend result → LLM system prompt (fallback chain) |
| `src/cloud-api/server.ts` | ASR/LLM/TTS provider router (reads `.env`) |
| `src/cloud-api/local/oasis-rag-client.ts` | RAG HTTP client |
| `src/cloud-api/local/oasis-matcher-node.ts` | Embedded fallback matcher (no server required) |

New ASR/LLM/TTS providers go in `src/cloud-api/local/` or a named subfolder — never in `server.ts` directly.

---

## Build

```bash
bash build.sh           # TypeScript: rm -rf dist && tsc
bash index_knowledge.sh # Rebuild FAISS index after knowledge base changes
```

---

## Coding Conventions

- TypeScript: `camelCase` for variables/functions, `PascalCase` for classes and types.
- Python: `snake_case`; follow existing module structure in the relevant subfolder.
- Commit format: `type: short description` (e.g., `feat:`, `fix:`, `refactor:`, `docs:`).

---

## Reference Docs

| File | Read when... |
|------|-------------|
| `docs/architecture.md` | Modifying RAG stages, context injection signals, Flask API, or KB document format |
| `docs/testing.md` | Writing or debugging tests; looking up test IDs |
| `docs/roadmap.md` | Investigating known bugs, planning phases, or choosing an LLM upgrade |
| `docs/decisions.md` | Before changing embedding model, vector DB, chunking strategy, or LLM |
| `docs/entrypoints.md` | Looking up what any file in `python/oasis-rag/` does |
