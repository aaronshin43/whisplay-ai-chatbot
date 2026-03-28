# CLAUDE.md — python/oasis-rag/

3-Stage Hybrid RAG service for O.A.S.I.S. Based on the Pocket RAG paper, optimized for Raspberry Pi 5.

---

## Critical Rules (RAG-specific)

- **`context_injector.py` is the single source of truth** for emergency signal injection — never duplicate signal logic in any other file.
- **Validation must stay 117/117 PASS** — run `python tests/run_all.py` after any change to retrieval logic, context injection, or the knowledge base.
- **Do not modify `data/rag_index/`** — rebuild via `bash index_knowledge.sh` or `POST /index`.
- **TOP_K and all thresholds live in `config.py`** — never hardcode them in logic files.

---

## Pipeline Overview

```
Query
  |
  v  Stage 1 — Lexical Pre-filter     (medical_keywords.py inverted index, top-50 candidates)
  |
  v  Stage 2 — Hybrid Semantic Rerank  (hybrid = 0.6×cosine + 0.4×lexical, gte-small 384-dim)
  |            Body-part mismatch penalty applied (query_classifier.py)
  |            SCORE_THRESHOLD = 0.50 · TOP_K = 1 · MAX_PER_SOURCE = 1
  |
  v  Confidence Gate                   (app.py: best_score < 0.35 → LOW_CONFIDENCE_PROMPT)
  |
  v  Stage 3 — Context Compression     (compressor.py, COMPRESS_ENABLED = False by default)
  |
  v  Context Injection                 (context_injector.py, 22 signals prepended)
  |
  v  Prompt Build                      (prompt.py → system prompt → OasisAdapter.ts → LLM)
```

---

## File Reference

### Core pipeline

| File | Role |
|------|------|
| `app.py` | Flask entry point (:5001). `GET /health`, `POST /retrieve`, `POST /index`. |
| `retriever.py` | `Retriever` class — orchestrates Stage 1 → 2 → 3. Returns `RetrievalResult`. |
| `indexer.py` | Chunks → embeds → FAISS `IndexFlatIP` → keyword inverted index. Contains `IndexStore` and `load_index()`. |
| `config.py` | **All hyperparameters.** Paths, ALPHA, TOP_K, thresholds, compression settings, Flask port. Override via `OASIS_*` env vars. |
| `context_injector.py` | 22-signal injection (single source of truth). `inject_context(context, query)`. |
| `prompt.py` | `SYSTEM_PROMPT_TEMPLATE`, `SAFE_FALLBACK_PROMPT`, `LOW_CONFIDENCE_PROMPT`. `build_system_prompt()`. |

### Stage modules

| File | Stage | Role |
|------|-------|------|
| `medical_keywords.py` | Stage 1 | ~200-term taxonomy. `detect_keywords()`, `expand_query()`. Also used by indexer to build inverted index. |
| `query_classifier.py` | Stage 2 | `classify_query()` → `QueryClassification(emergency_type, body_parts, severity, confidence)`. Used for body-part mismatch penalty. |
| `document_chunker.py` | Pre-index | `SectionAwareChunker` — H3-boundary splits with min-size merge. |
| `compressor.py` | Stage 3 | `compress_chunk(chunk_text, query, section)`. Safety sentences always preserved. |

### Tools and tests

| File | Role |
|------|------|
| `tools/chat_test.py` | Interactive CLI — connects to Flask (:5001) and Ollama (:11434). |
| `tools/test_retriever.py` | 5 retriever functional tests with Stage 1/2/3 stats. |
| `tests/run_all.py` | **Primary gate: 117/117 PASS.** No Flask or Ollama required. |
| `validation/test_llm_response.py` | 35-test LLM+RAG integration. Requires Flask + Ollama running. |
| `validation/compare_models.py` | Side-by-side LLM model comparison with latency estimates. |

---

## Key Config Values (current)

| Parameter | Value | Effect |
|-----------|-------|--------|
| `ALPHA` | 0.6 | Semantic weight in `hybrid = ALPHA×cosine + (1-ALPHA)×lexical` |
| `SCORE_THRESHOLD` | 0.50 | Minimum hybrid score to pass Stage 2 |
| `CONFIDENCE_THRESHOLD` | 0.35 | Below this → `LOW_CONFIDENCE_PROMPT` (in `app.py`) |
| `TOP_K` | 1 | Final chunks returned to LLM |
| `COMPRESS_ENABLED` | False | Stage 3 compression disabled by default |

---

## Build / Run / Test

```bash
# Start service
cd python/oasis-rag && python app.py      # Flask RAG server (:5001)

# Rebuild knowledge index (after adding/editing data/knowledge/*.md)
bash index_knowledge.sh

# Run all tests (no Flask needed)
cd python/oasis-rag && python tests/run_all.py

# LLM integration tests (requires Flask + Ollama)
cd python/oasis-rag && python validation/test_llm_response.py

# Manual end-to-end test
cd python/oasis-rag && python tools/chat_test.py
```

---

## Prompt Templates (3-state)

| Condition | Template |
|-----------|----------|
| Chunks exist AND `best_score >= 0.35` | `SYSTEM_PROMPT_TEMPLATE` — full RAG context |
| Chunks exist BUT `best_score < 0.35` | `LOW_CONFIDENCE_PROMPT` — "no specific info found" |
| Zero chunks (all below `SCORE_THRESHOLD`) | `SAFE_FALLBACK_PROMPT` — "knowledge base unavailable" |

---

## Knowledge Base Format

All documents under `data/knowledge/` must use this header format for Stage 1 to work:

```markdown
# Document Title

**Source:** ...
**Standard:** ...
**Category:** ...

[DOMAIN_TAGS: tag1, tag2, ...]

## Section

### Subsection  ← chunk boundary (SectionAwareChunker splits here)

Content...
```

Missing or incorrect `[DOMAIN_TAGS: ...]` causes Stage 1 to miss the document on keyword queries.

**Rebuild required after:** adding/editing `.md` files, changing `document_chunker.py`, adding keywords to `medical_keywords.py`.

---

## Context Injection Rules

- Injections are **prepended** — LLM reads the directive before retrieved chunks.
- `context_injector.py` is the only file that may contain signal detection logic.
- When adding a new signal: add signal list + protocol text + `inject_context()` call all in `context_injector.py`; then add a corresponding `CI-` unit test in `tests/unit/test_context_injector.py`.
- Do NOT add numbered format to injected protocols — the 1b model echoes numbers in output.

---

## Known Bug

**BUG-001 (LLM-008):** Lightning query retrieves altitude document (rank 1) instead of lightning document. Root cause: `redcross_altitude.md` shares vocabulary ("open area", "shelter", "move downhill"). Context injection prepend/append both attempted and failed — 1b model follows rank-1 RAG regardless.

Resolution paths: re-chunk altitude doc to isolate overlapping language; or strengthen `DOMAIN_TAGS` in lightning doc. See `docs/roadmap.md` for full details.

---

## Reference Docs

| File | Read when... |
|------|-------------|
| `docs/architecture.md` | Full pipeline internals, all 22 injection signals, Flask API schema, KB format |
| `docs/testing.md` | Test IDs, SAFE-check patterns, AND vs OR keyword assertions |
| `docs/entrypoints.md` | Dependency map, tools reference |
| `docs/decisions.md` | Before changing embedding model, vector DB, or chunking strategy |
| `docs/roadmap.md` | Known bugs, Phase 2–6 plans, LLM upgrade candidates |
