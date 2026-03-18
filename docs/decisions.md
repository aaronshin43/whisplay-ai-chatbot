# Architectural Decisions

Key decisions made during O.A.S.I.S. development. Each entry records what was decided, why, and what alternatives were rejected.

---

## DEC-001: Embedding model — gte-small

**Decision:** Use `thenlper/gte-small` (384-dim, ~67MB) via sentence-transformers directly.

**Rejected alternatives:**
- `all-MiniLM-L6-v2` — already installed in repo, but 4.5% lower accuracy on benchmark
- `mxbai-embed-large` — already in Ollama, but 670MB RAM; exceeds Pi5 memory budget and blocks concurrent LLM

**Why gte-small:** Best accuracy/size tradeoff for medical retrieval; sentence-transformers native (no Ollama overhead). First run requires internet to download (~67MB); after that fully offline.

---

## DEC-002: Vector DB — FAISS over ChromaDB

**Decision:** FAISS `IndexFlatIP` (in-memory, cosine via L2-normalized inner product).

**Rejected:** ChromaDB — adds 150MB RAM, requires separate server process, and its metadata filter is redundant because Stage 1 lexical filtering already does candidate narrowing.

**Why FAISS:** Matches Pocket RAG paper design; already installed via oasis-matcher.py; Stage 1 handles filtering so FAISS only needs flat search over top-50 candidates.

---

## DEC-003: RAG service in Python Flask, not Node.js

**Decision:** Implement RAG pipeline as a separate Python Flask service on `:5001`.

**Rejected:** Port RAG to TypeScript using `@xenova/transformers`.

**Why Python:** `sentence-transformers` is Python-native and runs 2–3× faster than the JS port. FAISS Python bindings are more stable. Keeps the existing Node.js chatbot unchanged — OasisAdapter just calls HTTP.

---

## DEC-004: 3-tier fallback in OasisAdapter

**Decision:** RAG failure degrades gracefully in three tiers:
1. Python RAG Flask (`/retrieve`) — primary
2. `oasis-matcher-node.ts` — embedded protocol matcher, no server needed
3. Empty prompt — LLM responds with "call emergency services"

**Why:** Pi5 boots services sequentially; RAG service may not be ready when first query arrives. Having an embedded fallback prevents silent failures in safety-critical situations.

---

## DEC-005: SectionAwareChunker (H3-boundary chunking)

**Decision:** Split at H3 (`###`) boundaries, merge sections < 80 tokens with sibling sections under the same H2. Named `SectionAwareChunker` in `document_chunker.py`.

**Rejected:** Sliding window (300-token, 50-token overlap) — created 547 micro-chunks averaging 20 tokens, severely degrading embedding quality and causing unrelated body-part sections (e.g., "Broken Finger" and "Rib Fracture") to share chunks.

**Why SectionAwareChunker:** H3 level aligns with medical procedure boundaries in WHO BEC and Red Cross documents. Minimum-size merge prevents empty embeddings while respecting anatomical separation.

---

## DEC-006: context_injector.py as single source of truth

**Decision:** All 22 emergency signal injection rules live exclusively in `python/oasis-rag/context_injector.py`. The Python `/retrieve` endpoint applies injections before returning context.

**Rejected:** Keep injection logic split across `chat_test.py`, `test_llm_response.py`, and `OasisAdapter.ts`.

**Why:** Prior to Phase 1 fix, `OasisAdapter.ts` (the production path) only had 2 of 22 signals (spinal, seizure). The remaining 20 protocols were missing from real device operation. Single-source ensures test and production paths are identical.

---

## DEC-007: LLM — gemma3:1b (current), upgrade path planned

**Decision:** Current LLM is `gemma3:1b` via Ollama.

**Context:** Model evaluation on 35 LLM+RAG tests (2026-03-15):

| Model | Critical pass | CPR (LLM-002) | Pi5 latency est. | RAM |
|-------|:---:|:---:|:---:|:---:|
| qwen3.5:0.8b | 95.2% | **100%** | ~64s | 1.0 GB |
| gemma3:1b | 85.7% | **0%** | ~62s | 0.8 GB |
| gemma3:4b Q4 | higher | high | ~8–12s | 3.0 GB |
| phi-4-mini Q4 | high | high | ~7–10s | 2.8 GB |

**Note:** gemma3:1b scored 0% on CPR (LLM-002) during model comparison but was adopted after context injection improvements; current validation shows 109/109 PASS on RAG layer. Upgrade to `phi-4-mini Q4` or `gemma3:4b Q4` is planned (Phase 3) pending Pi5 memory profiling.

---

## DEC-008: Hybrid scoring α = 0.6 (semantic) + 0.4 (lexical)

**Decision:** `hybrid_score = 0.6 × cosine_similarity + 0.4 × lexical_overlap`

**Why not pure dense:** Medical emergency queries contain exact clinical terms ("tourniquet", "epinephrine", "30:2") that must match lexically. Pure cosine retrieval caused anatomically wrong documents to rank above correct ones in early tests.

**Why not pure lexical:** Panicked, typo-heavy, or colloquial queries ("shes shaking eyes rolled back") need semantic understanding that BM25 alone misses.

**Tuning note:** α is defined in `python/oasis-rag/config.py` and does not require index rebuild to change.

---

## DEC-010: Two-threshold retrieval confidence gate

**Decision:** Use two separate thresholds with different semantics:
- `SCORE_THRESHOLD = 0.10` — structural noise floor in `retriever.py`; chunks below this are discarded entirely during Stage 2
- `CONFIDENCE_THRESHOLD = 0.35` — service-layer confidence gate in `app.py`; if the best returned chunk's score is below this, `LOW_CONFIDENCE_PROMPT` is delivered to the LLM instead of the full RAG template

**Why two thresholds, not one:** The structural threshold is a retrieval concern (discard noise); the confidence threshold is a prompt-selection concern (decide what to tell the LLM). Merging them into one would conflate retrieval mechanics with service behaviour, and would also break the 109 validation tests which test `Retriever` directly.

**Why the check is in `app.py`, not `retriever.py`:** `retriever.py` is a pure retrieval engine. Confidence evaluation — determining what to tell the LLM — is a service-layer decision. This keeps the retriever testable in isolation and the 109 tests unchanged.

**Why `LOW_CONFIDENCE_PROMPT` instead of empty context:** Returning empty context silently to `build_system_prompt` would produce `SAFE_FALLBACK_PROMPT` ("KB unavailable"), which is misleading when the KB is healthy but has no relevant answer. A distinct prompt is more honest and more useful to the user.

**Threshold value rationale:** 0.35 sits between the noise floor (0.10) and well-matched medical chunks (typically 0.60+). EDG-004/005 (off-topic queries) score below 0.50 in validation, confirming off-topic queries will trigger the gate. Tune via `config.py` — no index rebuild required.

---

## DEC-011: Test suite restructured into tests/ with unit/ tier

**Decision:** Consolidated all test files under `python/oasis-rag/tests/`, with a dedicated `tests/unit/` subdirectory for model-free unit tests. The old `validation/` directory retains only LLM-gate tests and utilities; `tools/` retains only manual development helpers.

**Previous state:** Tests were scattered across three locations — `validation/` (integration), `tools/` (overlap with validation), and no unit tests existed. Four files tested the same "keyword in context" condition (test_retrieval_accuracy, test_coverage, test_source_quality, test_context_quality), and two files measured latency (test_latency + benchmark.py).

**Rejected alternative:** Keep dual `validation/` + `tools/` structure — rejected because redundant files caused maintenance drift (fixing a test failure required updating multiple files) and the absence of unit tests made it hard to isolate which pipeline stage was responsible for a failure.

**Why this structure:** Clear tier separation — unit tests (no model, fast), integration tests (model required, no Flask), LLM gate (model + Flask). Deleting 5 redundant files (test_coverage, test_source_quality, test_context_quality, benchmark, run_all_tests) reduced total test count from ~180 cases across files to 117 canonical cases with no duplication. `test_retrieval_accuracy.py` was the strictest (checks both keywords AND source document) so it was kept as the authority; the others were subsets.

---

## DEC-012: Unit tests added for context_injector, compressor, medical_keywords

**Decision:** Added three unit test files (CI × 25, COMP × 10, MKW × 8) that test pipeline components in isolation — no embedding model or FAISS index required.

**Why these three modules:** They contain non-trivial logic that was previously only verified indirectly through integration tests. Failures in integration tests (e.g., a wrong keyword in context) were ambiguous: was it a retrieval failure, a compression failure, or a signal injection failure? Unit tests provide the missing signal.

**Key design choices:**
- `test_context_injector.py`: one test per signal (CI-001..022) so a regression pinpoints exactly which signal broke, plus three special-case tests (mutual exclusion, append-at-end, empty query)
- `test_compressor.py`: explicitly tests safety-critical invariants — "Do not" / "Never" sentences must survive compression regardless of relevance score
- `test_medical_keywords.py`: MKW-007 uses "programming syntax errors in javascript code" (not a weather/temperature query) because terms like "warm" exist in the temperature_emergencies taxonomy and would produce a false positive

---

## DEC-013: Per-stage latency tests absorbed into test_latency.py

**Decision:** Merged `tools/benchmark.py` per-stage timing into `tests/test_latency.py` as LAT-S1/S2/S3 test cases. benchmark.py was deleted.

**Stage targets:** S1 (lexical filter) < 5ms, S2 (semantic rerank) < 500ms, S3 (compression) < 50ms.

**Why merged:** benchmark.py and test_latency.py both measured retrieval latency with different methodologies. Having them separate meant a latency regression might be caught in one but not tracked in the CI gate. Merging makes per-stage timing part of the 117-test suite so it is always run and always enforced.

**Implementation:** `test_latency.run(retriever, store)` accepts an optional `store` parameter. If provided, `_bench_stages(store)` runs isolated Stage 1/2/3 timing (5 queries × 5 warmup runs). If omitted, only E2E tests run — backward compatible with any caller that passes only `retriever`.

---

## DEC-014: validation/run_all.py kept as backward-compat stub

**Decision:** After moving the test suite to `tests/run_all.py`, `validation/run_all.py` was replaced with a two-line stub using `runpy.run_path` that delegates to the new location.

**Why not delete it:** The old path (`python validation/run_all.py`) is in shell history, CI scripts, and developer muscle memory. A stub costs nothing and prevents silent failures for anyone using the old path.

**Why runpy over a shell alias or symlink:** Works cross-platform (Windows + Linux/Pi5) without filesystem-level tricks, and preserves `__main__` semantics so `sys.exit()` in the real runner propagates correctly.

---

## DEC-009: text_with_prefix embedding format

**Decision:** Index chunks using `text_with_prefix` (heading breadcrumb + content) rather than raw content only.

**Why:** Increases vector distance between anatomically different sections (e.g., "Broken Finger" under hand injuries vs "Rib Fracture" under chest injuries), reducing cross-anatomy false matches in Stage 2 retrieval.

**Implementation:** `indexer.py` passes `m["text_with_prefix"]` to the encoder instead of `m["text"]`.
