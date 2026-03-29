# CLAUDE.md — python/oasis-classify/

Medical intent classification + pre-generated manual dispatch for O.A.S.I.S.
Standalone system — independent of oasis-rag. Designed to reduce LLM TTFT on Pi5 from ~16s to ~3-6s by minimizing prompt length.

For full design rationale, category definitions, manual format, and implementation phases → read `PLAN.md` in this folder first.

---

## Critical Rules

- **This system is standalone** — it does not call or depend on oasis-rag (:5001).
- **Manuals are the single source of medical content** — never duplicate protocol text outside `data/manuals/`. If a protocol needs updating, update the `.txt` file only.
- **`out_of_domain` cluster must always exist** — removing it causes non-medical queries to enter triage incorrectly.
- **Tier 0 word count guard: keyword matching only for queries ≤ 3 words** — do not apply keyword or fuzzy matching to full sentences.
- **`data/centroids.npy` is a build artifact** — regenerate via `python build_centroids.py`, never edit manually.

---

## Dispatch Pipeline

```
User Query
  |
  v  Tier 0A (word count <= 3): SHORT_QUERIES dict lookup     ~0ms
  |  Tier 0B (word count >  3): SENTENCE_MATCHES dict lookup  ~0ms
  |  No match → continue to Tier 1
  |
  v  Tier 1: gte-small embed → cosine similarity to centroids  ~10-100ms
  |          33 centroids: 32 medical categories + out_of_domain
  |
  +-- score < OOD_FLOOR (0.30)    → pre-baked OOD response
  +-- out_of_domain cluster best  → pre-baked OOD response
  +-- score in [0.30, 0.65)       → Triage (LLM asks clarifying question)
  +-- score >= CLASSIFY_THRESHOLD → Manual lookup → compact prompt → LLM
```

**Prompt size target:** ~200-350 tokens total (vs ~500-800 in oasis-rag).

---

## File Reference

| File | Role |
|------|------|
| `config.py` | All thresholds and paths. `CLASSIFY_THRESHOLD`, `OOD_FLOOR`, `TIER0_MAX_WORDS`, `TRIAGE_HINT_BOOST`. |
| `categories.py` | 32 category definitions + metadata (ID, description, KB source, priority level). |
| `fast_match.py` | Tier 0 — word match (≤ 3 words) and sentence match (> 3 words). ASR-robust normalization + edit distance 1 for short tokens. |
| `classifier.py` | Tier 1 — loads `data/centroids.npy`, embeds query with gte-small, returns `DispatchResult` with `mode`, `category`, `score`, `top3`, `threshold_path`, `latency_ms`. |
| `manual_store.py` | Loads `data/manuals/*.txt` at startup into a `dict[str, str]`. Serves manual by category ID. |
| `triage.py` | Triage prompt template for ambiguous medical queries. |
| `prompt_builder.py` | Assembles compact LLM prompt from manual + query. Handles multi-label: primary manual + short "also check" block. Enforces `MAX_PROMPT_TOKENS (400)` hard ceiling. |
| `service.py` | Flask API (:5002). `POST /dispatch` → returns pre-baked response, compact prompt, or triage prompt. |
| `build_centroids.py` | **Offline build script.** Embeds all prototypes from `data/prototypes.json` with gte-small, computes per-category centroids, saves to `data/centroids.npy`. |

### Data files

| File | Role |
|------|------|
| `data/prototypes.json` | `{ category_id: [prototype_query, ...] }` — 15-30 queries per category. Used by `build_centroids.py`. |
| `data/centroids.npy` | **Build artifact.** `(33, 384)` float32 array — one centroid per category. |
| `data/short_queries.json` | Tier 0A — `{ normalized_query: category_id_or_response }` for word count ≤ 3. |
| `data/sentence_matches.json` | Tier 0B — `{ normalized_sentence: category_id_or_response }` for word count > 3. |
| `data/also_check_summaries.json` | Per-category curated one-liners for multi-label "ALSO CHECK" block. Human-written; never generated at runtime. |
| `data/manuals/*.txt` | 32 manual files, one per category. STEPS + NEVER DO format. 80-140 tokens each. |

### Tests

| File | Role |
|------|------|
| `tests/test_fast_match.py` | Tier 0 — exact match, sentence match, word count guard, ASR normalization, edit distance 1. |
| `tests/test_classifier.py` | Tier 1 — category accuracy, OOD detection, triage band coverage, top3 field correctness. |
| `tests/test_manuals.py` | Format validation — STEPS + NEVER DO both present, token count in 150-250 range. |
| `tests/test_integration.py` | End-to-end: all four dispatch modes, multi-label token ceiling, telemetry fields populated. |
| `tests/test_contract.py` | `/dispatch` response schema stability — ensures TypeScript contract never silently breaks. |
| `tests/test_adversarial.py` | Mixed-topic inputs, ASR noise, long nonsense, foreign language, canary recall (life-critical categories recall >= 0.95). |

---

## Key Config Values

| Parameter | Value | Effect |
|-----------|-------|--------|
| `CLASSIFY_THRESHOLD` | 0.65 | >= this → manual; [OOD_FLOOR, this) → triage |
| `OOD_FLOOR` | 0.30 | < this → OOD response (not triage) |
| `TIER0_MAX_WORDS` | 3 | Tier 0 word/keyword match only for queries ≤ this many words |
| `MULTI_LABEL_RATIO` | 0.80 | Secondary category included if score >= primary × this |
| `MAX_CATEGORIES` | 2 | Maximum categories per query |
| `MAX_PROMPT_TOKENS` | 400 | Hard ceiling — measured with `tiktoken` (cl100k_base), not `transformers.AutoTokenizer` (too heavy for Pi5) |
| `TRIAGE_HINT_BOOST` | 0.05 | Score boost applied when `prev_triage_hint` matches a category |
| `TRIAGE_HINT_MIN_RELEVANCE` | 0.20 | Skip boost if query cosine sim to hint < this; handles topic-shift in Python |
| `TRIAGE_HINT_TTL_SEC` | 60 | Hint expiry in seconds — managed by TypeScript layer |
| `CATEGORY_THRESHOLDS` | `{}` | Per-category threshold overrides; falls back to `CLASSIFY_THRESHOLD` |

---

## /dispatch Response Schema

```python
{
  "mode":               "direct_response" | "llm_prompt" | "triage_prompt" | "ood_response",
  "response_text":      str | None,   # direct_response and ood_response only
  "system_prompt":      str | None,   # llm_prompt and triage_prompt only
  "category":           str | None,   # always set for triage_prompt; None for ood/direct
  "top3":               [{"category": str, "score": float}, ...],  # JSON objects, not tuples
  "score":              float | None,
  "threshold_path":     str,          # server: "tier0_short" | "tier0_sentence" | "classifier_hit"
                                      #         | "triage" | "ood_floor" | "ood_cluster"
                                      # client (synthesized): "network_error" | "service_error"
                                      #         | "invalid_schema"
  "latency_ms":         float,
  "hint_changed_result": bool
}
```

**TypeScript adapter decision table:**

| mode | ChatFlow behavior |
|------|-------------------|
| `direct_response` | Speak `response_text` via TTS. No LLM call. Clear triage hint. |
| `llm_prompt` | Send `system_prompt` to LLM → speak result via TTS. Clear triage hint. |
| `triage_prompt` | Send `system_prompt` to LLM → speak result via TTS. Store `category` as new triage hint with TTL. |
| `ood_response` | Speak `response_text` via TTS. No LLM call. Clear triage hint. |

**Client contract:** The classify client must resolve, never reject. On network error or unexpected response, return `mode="ood_response"` with a generic help message.

---

## Category Priority Levels

When two categories are returned, the more dangerous one is placed first:

```python
PRIORITY_CRITICAL = ["cpr", "choking", "bleeding"]
PRIORITY_URGENT   = ["anaphylaxis", "electric_shock", "poisoning", "drowning"]
# All others: ordered by classifier score
```

---

## Manual Format

Every manual in `data/manuals/` must follow this exact format:

```
Category: [Human-readable name]

STEPS:
1. [First and most critical action].
2. [Second action].
...

NEVER DO:
- Do NOT [common myth or dangerous mistake].
- Do NOT [another dangerous action].
...
```

- Plain text only — no markdown. The 1b LLM copies markdown formatting if it sees it.
- 80-140 tokens per manual.
- STEPS and NEVER DO sections are both required in every manual.
- Content must be distilled from `data/knowledge/` source documents — not invented.

---

## Build / Run / Test

```bash
# Build centroids (required after editing prototypes.json or adding categories)
cd python/oasis-classify && python build_centroids.py

# Start service
cd python/oasis-classify && python service.py    # Flask dispatch server (:5002)

# Run tests
cd python/oasis-classify && python -m pytest tests/

# Smoke test a single query
curl -X POST http://localhost:5002/dispatch \
  -H "Content-Type: application/json" \
  -d '{"query": "my friend is bleeding from the leg"}'
```

**Rebuild centroids required after:** editing `data/prototypes.json`, adding/removing categories in `categories.py`, or changing `EMBEDDING_MODEL` in `config.py`.

---

## Triage Hint — TypeScript Integration Note

This service is stateless. The `prev_triage_hint` field in `/dispatch` requests must be managed by the TypeScript caller:

- `OasisAdapter.ts` or `ChatFlow.ts` holds `triageHint: { category: string, expiresAt: number } | null`
- TTL: **60 seconds** (`TRIAGE_HINT_TTL_SEC`) from the last triage response
- Send `prev_triage_hint: null` if the hint is absent or expired
- Clear hint when the service returns `mode == "llm_prompt"` (triage resolved)
- **Do not inspect query text in TypeScript** — topic-shift detection is handled Python-side via `TRIAGE_HINT_MIN_RELEVANCE`. Pass the hint blindly.

---

## Score Routing Summary

```
score < 0.30                    → OOD response ("I am a first-aid assistant...")
out_of_domain cluster best hit  → OOD response
0.30 <= score < 0.65            → Triage prompt → LLM asks clarifying question
score >= 0.65                   → Manual lookup → compact prompt → LLM
```

---

## Reference

| File | Read when... |
|------|-------------|
| `PLAN.md` | Full design rationale, all 32 categories, manual creation process, implementation phases |
