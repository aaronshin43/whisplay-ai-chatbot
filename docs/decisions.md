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

## DEC-009: text_with_prefix embedding format

**Decision:** Index chunks using `text_with_prefix` (heading breadcrumb + content) rather than raw content only.

**Why:** Increases vector distance between anatomically different sections (e.g., "Broken Finger" under hand injuries vs "Rib Fracture" under chest injuries), reducing cross-anatomy false matches in Stage 2 retrieval.

**Implementation:** `indexer.py` passes `m["text_with_prefix"]` to the encoder instead of `m["text"]`.
