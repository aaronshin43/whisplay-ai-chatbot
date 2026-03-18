# O.A.S.I.S. Architecture

Detailed architecture reference for the RAG pipeline, Context Injection system, Flask API, and knowledge base structure.

---

## 1. RAG 3-Stage Pipeline

`retriever.py` implements the core algorithm, based on the Pocket RAG paper, optimized for Pi5's constrained resources.

```
Query
  │
  ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Medical Keyword Lexical Filter      │
│  medical_keywords.py inverted-index lookup    │
│  → up to 50 candidate chunks                 │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Stage 2: Hybrid Semantic Re-ranking          │
│  hybrid_score = 0.6×cosine + 0.4×lexical     │
│  gte-small (384-dim) vector similarity        │
│  → top-4 chunks above threshold (0.10)       │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Stage 3: Context Compression                 │
│  compressor.py — query-relevant sentences     │
│  Safety sentences always preserved           │
│  → 20–40% token reduction                   │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
            RetrievalResult
          (.chunks, .context, .latency_ms)
```

### Stage detail

**Stage 1 — Lexical Pre-filtering**
- `medical_keywords.py` taxonomy → inverted index built at index time
- Query keywords detected → `keyword_map` lookup for candidate chunk IDs
- If no keywords match, falls back to full FAISS search over all chunks
- Purpose: eliminate expensive vector similarity calculations for irrelevant chunks

**Stage 2 — Hybrid Re-ranking**
```python
hybrid_score = 0.6 * cosine_similarity + 0.4 * lexical_overlap
```
- `cosine`: gte-small embedding similarity (384-dim L2-normalized → `IndexFlatIP`)
- `lexical`: query-chunk keyword overlap ratio (query-specific, not global density)
- Body-part mismatch penalty applied to reduce cross-anatomy false positives
- Threshold 0.10, top-4 selected

**Stage 3 — Context Compression**
- Sentence-level scoring: keyword hits × 1.0 + position bonus (0.05/position)
- "Do NOT", "Never", "Avoid" + query keywords always preserved
- Minimum 2 sentences or 40% token preservation guaranteed
- Output: `[Section Header]\ncompressed sentences`

---

## 2. Context Injection

Even with correct RAG retrieval, compact models (0.8b–1b) sometimes ignore context. Context Injection prepends protocol directives before the RAG context so the LLM reads them first.

### How it works

```python
# context_injector.py (single source of truth)
q_lower = query.lower()

if any(sig in q_lower for sig in _CARDIAC_ARREST_SIGNALS):
    context = "CARDIAC ARREST PROTOCOL — ACT NOW:\n1. CALL..." + context

if any(sig in q_lower for sig in _BURN_SIGNALS):
    context = "⚠ BURN — MOST IMPORTANT FIRST ACTION:\n..." + context
```

Injections are **prepended** — the LLM reads the directive before the retrieved context.

### 22 injection signals

| Signal variable | Detected situation | Injected content |
|---|---|---|
| `_CARDIAC_ARREST_SIGNALS` | No pulse, not breathing | CPR 30:2 protocol |
| `_SPINAL_SIGNALS` | Suspected spinal injury | Do not move |
| `_BURN_SIGNALS` | Burns | Cool 20 min immediately |
| `_CHOKING_SIGNALS` | Airway obstruction | Back blows + Heimlich |
| `_SNAKEBITE_SIGNALS` | Snake bite | Immobilize, do not suck |
| `_HYPOTHERMIA_SIGNALS` | Hypothermia | Warm core, handle gently |
| `_HEAT_STROKE_SIGNALS` | Heat stroke | Cool immediately |
| `_LIGHTNING_SIGNALS` | Lightning | Crouch low, avoid trees |
| `_FROSTBITE_SIGNALS` | Frostbite | Warm water rewarming |
| `_PANIC_BLOOD_SIGNALS` | Mass bleeding panic | Direct pressure |
| `_NO_EPIPEN_SIGNALS` | No EpiPen available | Call emergency first |
| `_SEIZURE_SIGNALS` | Seizure/convulsion | Do not restrain, side position |
| `_STROKE_SIGNALS` | Stroke | FAST assessment, no aspirin |
| `_DROWNING_SIGNALS` | Submersion | Rescue breathing first |
| `_POISONING_SIGNALS` | Poisoning | Do not induce vomiting |
| `_ELECTRIC_SHOCK_SIGNALS` | Electric shock | Do not touch victim |
| `_INFANT_CPR_SIGNALS` | Infant cardiac arrest | 2-finger compressions |
| `_EYE_CHEMICAL_SIGNALS` | Chemical eye injury | Running water irrigation |
| `_SHOCK_SIGNALS` | Shock | Elevate legs, call emergency |
| `_FRACTURE_SIGNALS` | Fracture | Immobilize as found, no realignment |
| `_ASTHMA_SIGNALS` | Asthma attack | Sit upright, do not lie down |
| `_HEART_ATTACK_SIGNALS` | Heart attack | Aspirin if conscious, call 911 |

### Design principles

1. **Prepend:** directives appear before RAG context → LLM reads them first
2. **Priority:** when signals overlap, more dangerous takes precedence (e.g., chemical eye > burn)
3. **No numbered format:** numbered injections cause LLM to echo those numbers in output → use unnumbered paragraphs
4. **Single source:** `context_injector.py` only — never add signal logic elsewhere

---

## 3. Index Build Process

```bash
bash index_knowledge.sh
# or directly:
cd python/oasis-rag && python indexer.py
```

```
data/knowledge/*.md
       │
       ▼  SectionAwareChunker (document_chunker.py)
  H3-boundary splits, min-size merge within H2
       │
       ▼  sentence-transformers/gte-small
  text_with_prefix → 384-dim embedding
       │
       ├──▶ data/rag_index/chunks.faiss      (FAISS IndexFlatIP)
       ├──▶ data/rag_index/metadata.json     (chunk text, source, section)
       └──▶ data/rag_index/keyword_map.json  (keyword → chunk ID inverted index)
```

**Rebuild required when:**
- Adding or editing `data/knowledge/*.md` files
- Changing chunking logic in `document_chunker.py`
- Adding keywords to `medical_keywords.py`

**No rebuild needed for:**
- Changing `ALPHA`, `SCORE_THRESHOLD`, `TOP_K` in `config.py`
- Changing context injection signals in `context_injector.py`

---

## 4. Knowledge Base Document Format

All documents under `data/knowledge/` must follow this header format for Stage 1 lexical matching to work correctly:

```markdown
# WHO BEC — CPR Protocol

**Source:** WHO Basic Emergency Care 2018
**Standard:** WHO / ICRC 2018
**Category:** Clinical Skills — Cardiopulmonary Resuscitation

[DOMAIN_TAGS: CPR, cardiac_arrest, chest_compressions, 30_2, AED, ...]

## Section Title

### Subsection (chunk boundary)

Content...
```

- `[DOMAIN_TAGS: ...]` — used for BM25 lexical scoring and keyword map construction
- `###` headers mark chunk boundaries (SectionAwareChunker splits here)
- Without correct DOMAIN_TAGS, Stage 1 will miss the document on keyword queries

---

## 5. Flask API Reference

Base URL: `http://localhost:5001`

### GET /health

```bash
curl http://localhost:5001/health
```

Response:
```json
{
  "status": "ok",
  "index_ready": true,
  "chunk_count": 374,
  "model": "thenlper/gte-small"
}
```

### POST /retrieve

```bash
curl -X POST http://localhost:5001/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "patient bleeding from leg wont stop"}'
```

Request body:
```json
{
  "query": "patient bleeding from leg wont stop",
  "top_k": 4,       // optional (default 4)
  "compress": true  // optional (default true)
}
```

Response:
```json
{
  "context": "[Source 1: who_bec_skills_bleeding.md]\n...",
  "chunks": [
    {
      "source": "who_bec_skills_bleeding.md",
      "section": "Tourniquet Application",
      "hybrid_score": 0.72,
      "cosine_score": 0.81,
      "lexical_score": 0.55,
      "compressed_text": "..."
    }
  ],
  "stage1_candidates": 12,
  "stage2_passing": 4,
  "latency_ms": 25.4
}
```

### POST /index

Rebuild the knowledge index from `data/knowledge/`. Run after adding or editing documents.

```bash
curl -X POST http://localhost:5001/index
```

---

## 6. LLM System Prompt

The system prompt injected into every LLM call:

```
You are OASIS. A person needs first aid RIGHT NOW.

RULES YOU MUST FOLLOW:
- Your response is ONLY numbered steps 1 through 7 maximum.
- Do NOT write anything before "1."
- Each step is ONE sentence, maximum 12 words.
- Do NOT use asterisks, bold, markdown, or headers.
- Do NOT ask questions. Give commands only.

REFERENCE:
{RAG context + context injections}

YOUR RESPONSE MUST START WITH "1."
```

**Design rationale:** 5–7 steps covers all critical protocols. Plain text is required because TTS reads markdown symbols aloud. Strict format constraint compensates for small model (1b) tendency to add unnecessary explanation.

---

## 7. TypeScript Bridge

```
ChatFlow.ts
  └── OasisAdapter.getSystemPromptFromOasis(query)
        ├── [1] ragRetrieve()          oasis-rag-client.ts → POST :5001/retrieve (5s timeout)
        ├── [2] matchProtocolLocal()   oasis-matcher-node.ts (embedded, no server needed)
        └── [3] ""                     empty → LLM falls back to "call emergency services"
```

`oasis-rag-client.ts` exports:
- `isRagReady()` — health check
- `ragRetrieve(query)` — returns context string (empty string on error, never throws)
- `ragRetrieveFull(query)` — returns full response including chunk metadata
