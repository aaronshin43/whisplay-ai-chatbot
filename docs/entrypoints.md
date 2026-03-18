# Python Module Reference — `python/oasis-rag/`

Quick reference for every Python file in the RAG service. Load only the files relevant to your task.

---

## Core Pipeline Modules

| File | Role | Read when... |
|------|------|-------------|
| `app.py` | Flask HTTP service (port 5001). Exposes `GET /health`, `POST /retrieve`, `POST /index`. Entry point when running `python app.py`. | Modifying API endpoints, request/response schema, or service startup logic |
| `retriever.py` | **3-Stage Hybrid Retriever** — orchestrates Stage 1 → Stage 2 → Stage 3. Main class: `Retriever`. Returns `RetrievalResult` with `context`, `chunks`, `latency_ms`. | Changing retrieval logic, score thresholds, source diversity, or adding a new retrieval stage |
| `indexer.py` | **Document Indexing Pipeline** — chunks → embeds (gte-small) → FAISS IndexFlatIP → keyword inverted index. Writes artifacts to `data/rag_index/`. Also contains `IndexStore` and `load_index()` used by `app.py`. | Changing the embedding model, index format, or artifact layout |
| `config.py` | **Central configuration** — all hyperparameters in one place. Paths, ALPHA, TOP_K, SCORE_THRESHOLD, CONFIDENCE_THRESHOLD, compression settings, Flask host/port. Override via env vars (`OASIS_*`). | Tuning any pipeline parameter; checking what a constant value is |
| `context_injector.py` | **22-signal context injection** (single source of truth). `inject_context(context, query)` prepends emergency protocol blocks when signals are detected in the query. | Adding/editing emergency signals or protocol texts |
| `prompt.py` | **LLM system prompt templates** — `SYSTEM_PROMPT_TEMPLATE`, `SAFE_FALLBACK_PROMPT`, `LOW_CONFIDENCE_PROMPT`. `build_system_prompt()` selects the right template based on context availability and confidence. | Editing the LLM instruction format, fallback behavior, or markdown stripping logic |

---

## Stage-Level Modules

| File | Stage | Role |
|------|-------|------|
| `document_chunker.py` | Pre-index | Loads Markdown/txt files from `data/knowledge/` and splits them into overlapping chunks. Two chunkers: `DocumentChunker` (sliding window) and `SectionAwareChunker` (H3-boundary split, used by `Indexer`). |
| `medical_keywords.py` | Stage 1 | Curated medical keyword taxonomy (~200 terms, 14 categories). `detect_keywords(text)` scans for matches; `expand_query(query)` returns related category terms for query expansion. Used by `Indexer` to build the inverted index and by `Retriever` for Stage 1 filtering. |
| `query_classifier.py` | Stage 2 | Classifies query into `emergency_type`, `body_parts`, `severity`, `confidence`. Used by `Retriever._stage2_semantic()` to apply body-part mismatch penalties (e.g. finger fracture query should not retrieve chest/rib chunks). |
| `compressor.py` | Stage 3 | Selective context compression — scores each sentence by keyword hits + position, keeps sentences above threshold, preserves safety-critical sentences (`Do not / Never`). `compress_chunk(chunk_text, query, section)`. |

---

## Tools (`tools/`)

| File | Role |
|------|------|
| `tools/chat_test.py` | Interactive CLI — connects to `localhost:5001` (RAG) and `localhost:11434` (Ollama) for end-to-end manual testing. Run: `python tools/chat_test.py` |
| `tools/benchmark.py` | Latency benchmark — measures per-stage and total pipeline latency over N iterations, flags stages exceeding targets (Stage 2 target: <200 ms PC). Run: `python tools/benchmark.py` |
| `tools/test_retriever.py` | Retriever integration test — runs 5 realistic emergency queries, prints Stage 1/2/3 stats and context preview. Run: `python tools/test_retriever.py` |
| `tools/test_accuracy.py` | Content-accuracy suite — 30 cases (physical first-aid, safety, panic queries). Checks must_contain / must_not_contain keywords in context. |
| `tools/run_all_tests.py` | Full integration runner — Stage 0 (index rebuild) → Stage 1 (retriever) → Stage 2 (accuracy) → Stage 3 (benchmark). Loads index once. |

---

## Validation Suite (`validation/`)

> **Primary gate:** `python validation/run_all.py` — must stay **109/109 PASS**

| File | Test IDs | Role |
|------|----------|------|
| `validation/run_all.py` | — | **Main validation runner**. Discovers and runs all suites below. No Flask required. |
| `validation/_shared.py` | — | Shared helpers: `TestResult`, `get_context_text()`, `context_contains()`, `top_score()`, `top_source()`. Imported by every test file. |
| `validation/test_retrieval_accuracy.py` | BLD-, FRX-, BRN-, CHK-, ANA-, TMP-, POI-, ELC-, DRW-, MIX- | 47-case retrieval accuracy suite across 10 emergency categories. |
| `validation/test_safety.py` | SAF- | Safety guardrails — dangerous content (specific drug names, harmful advice) must NOT appear in context. |
| `validation/test_coverage.py` | COV- | Scenario coverage — every emergency scenario must return cosine_score ≥ 0.75. |
| `validation/test_edge_cases.py` | EDG- | Edge cases — panic input (all-caps), typos, empty queries, very long queries. |
| `validation/test_source_quality.py` | MED- | Medical fact verification — specific procedural facts (CPR depth, tourniquet placement) must be present in context. |
| `validation/test_latency.py` | LAT- | Latency benchmark — 20 repeats per query, target <200 ms (PC/CUDA). |
| `validation/test_llm_response.py` | LLM- | RAG + LLM integration — requires Flask on :5001 and Ollama on :11434. Evaluates response format, content correctness, safety. |
| `validation/compare_models.py` | — | LLM model comparison tool — runs N models × N iterations, produces side-by-side table with latency estimates. |

---

## Dependency Map

```
app.py
  ├── retriever.py        ← Retriever, RetrievalResult
  │     ├── indexer.py    ← IndexStore, load_index
  │     ├── compressor.py ← compress_chunk  (Stage 3)
  │     ├── medical_keywords.py ← detect_keywords, expand_query  (Stage 1)
  │     └── query_classifier.py ← classify_query  (Stage 2)
  ├── indexer.py          ← Indexer, load_index
  │     ├── document_chunker.py ← SectionAwareChunker
  │     └── medical_keywords.py ← detect_keywords
  ├── context_injector.py ← inject_context
  ├── prompt.py           ← build_system_prompt
  └── config.py           ← all hyperparameters
```
