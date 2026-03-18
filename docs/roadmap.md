# Production Roadmap

Goal: bring RAG + LLM quality to production level on Raspberry Pi 5 for real emergency first-aid use.

**Current baseline (2026-03-17):** 109/109 RAG tests PASS · 374 chunks · 33 documents · avg latency ~25ms (PC)

---

## Known bugs

### [BUG-001] LLM-008 — Lightning query retrieves altitude document 🔴

**Severity:** High (safety-related) | **Status:** Open

**Symptom:**
```
Query: "lightning coming no shelter in open field"
RAG rank 1: redcross_altitude.md  ← WRONG
RAG rank 2: redcross_lightning.md
LLM step 1: "Move downhill to a low rolling hill or tree."  ← dangerous
```

**Root cause:** `redcross_altitude.md` contains "open area", "shelter", "move downhill" → FAISS scores it higher than the lightning document.

**Attempted fixes that failed:**
- Context injection prepend + append simultaneously → 1b model follows rank-1 RAG doc regardless
- Stronger injection text ("NEVER go near trees") → ignored

**Resolution paths:**
1. Re-chunk `redcross_altitude.md` to isolate "open area" language into a separate chunk
2. Strengthen `DOMAIN_TAGS` in `redcross_lightning.md` to match query keywords
3. Add "lightning", "thunder" explicitly to `medical_keywords.py` taxonomy

---

## Phase 1: Foundation (complete ✅)

- ✅ BRN-001 test fixed (false positive on "ice" → changed to "apply ice")
- ✅ `context_injector.py` created — 22 signals unified as single source
- ✅ `service.py` concurrency bug fixed (per-request params instead of shared state mutation)
- ✅ `SectionAwareChunker` — H3-boundary chunking with min-size merge
- ✅ `text_with_prefix` embedding (heading context included)
- ✅ 109/109 PASS + anatomy mismatch resolved

---

## Phase 2: RAG Pipeline Hardening (in progress)

**Goal:** RAG retrieval works correctly without context injection for all scenarios.

### 2.0 Confidence threshold ✅
- `CONFIDENCE_THRESHOLD = 0.35` added to `config.py`
- `app.py`: if `best_score < 0.35`, delivers `LOW_CONFIDENCE_PROMPT` ("no specific info found") instead of the full RAG template
- Distinct from `SAFE_FALLBACK_PROMPT` (infrastructure failure) — see `docs/architecture.md §6`
- Zero test changes required — the 109 tests call `Retriever` directly and are unaffected

### 2.1 Query classifier (planned)
- New file: `python/oasis-rag/query_classifier.py`
- Output: `QueryClassification(emergency_type, body_part, severity, confidence)`
- Multi-injury: prioritize most life-threatening condition first
- Success criterion: 95%+ accuracy on 100-query benchmark

### 2.2 Body-part filtering (planned)
- Apply penalty in Stage 1/2 for chunks whose body-part metadata mismatches query
- Fixes: "broken finger" retrieving chest/rib/leg chunks

### 2.3 Compressor safety hardening (planned)
- Always preserve sentences with "Do NOT", "Never", "Avoid" + query keyword
- Pre-filter prescription drug names before LLM sees context (defense-in-depth)
- Preserve numbered-step structure from source documents

### 2.4 Knowledge base expansion (planned)

Priority additions to absorb remaining context-injection-only protocols:

| Priority | Document | Why needed |
|---|---|---|
| High | `who_bec_choking_airway.md` | Airway obstruction detail is thin |
| High | `who_bec_cardiac_arrest.md` | Cardiac arrest protocol is injection-only |
| High | `who_bec_shock_first_aid.md` | Shock first aid detail missing |
| Medium | `who_bec_infant_child_cpr.md` | Pediatric CPR differences |
| Medium | `redcross_eye_injuries.md` | Chemical splash coverage |
| Medium | `who_bec_drug_overdose.md` | AMS-005 partial cover |

**Phase 2 success criterion:** context injection not required for any of the 200+ planned tests.

---

## Phase 3: LLM Quality & Safety

### 3.1 Model upgrade

Pi5 memory budget for LLM (8GB total):
```
OS + system:      ~1.5 GB
Flask + FAISS:    ~1.2 GB
STT + TTS:        ~0.5 GB
────────────────────────
LLM headroom:     ~4.8 GB
```

Candidate models:

| Model | RAM | Pi5 response | Instruction following | Recommendation |
|---|---|---|---|---|
| gemma3:1b (current) | ~1.5 GB | 3–5s | Low | ❌ production risk |
| **phi-4-mini Q4** | ~2.8 GB | 7–10s | High | ✅ first choice |
| **gemma3:4b Q4** | ~3.0 GB | 8–12s | High | ✅ second choice |
| qwen2.5:3b Q4 | ~2.4 GB | 6–8s | Medium | 🟡 alternative |

**Latency target revision:** PC < 5s, Pi5 < 12s for 3b+ models.

### 3.2 Structured output validation
- Verify response follows numbered-list format (regex: `^\d+\.\s.+`)
- On failure: one retry with stricter prompt
- On second failure: return context injection text directly
- New file: `python/oasis-rag/response_validator.py`

### 3.3 Hallucination detection
- Compute "grounding score" — keyword overlap between response and RAG context
- Block action terms in `medical_keywords.py` that aren't present in context
- New file: `python/oasis-rag/hallucination_detector.py`

---

## Phase 4: Test Suite Expansion

| Category | Current | Target |
|---|:---:|:---:|
| Retrieval accuracy | 47 | 100+ |
| Safety | 8 | 30+ |
| LLM response quality | 35 | 60+ |
| E2E integration | 0 | 30+ |
| **Total** | **109** | **250+** |

Additional test areas:
- Anatomical precision (finger vs arm vs leg)
- Multi-injury priority triage
- Dangerous folk remedies (butter on burns, object in mouth during seizure)
- Out-of-scope rejection (surgery, suturing, prescription)
- Korean/English mixed queries

---

## Phase 5: Pi5 Optimization & Deployment

- Real device memory profiling
- Startup time < 30s (ONNX optimization, mmap FAISS)
- systemd services ready within 60s of boot
- Offline resilience: watchdog + auto-index rebuild on corruption

---

## Phase 6: Safety Validation Framework (ongoing)

- **Safety matrix:** all supported scenarios → source document + test ID mapping (auditable artifact)
- **Red-team tests:** 50 adversarial queries (mixed states, typos, harmful advice elicitation)
- **Out-of-scope detection:** hybrid score < 0.35 (`CONFIDENCE_THRESHOLD`) → `LOW_CONFIDENCE_PROMPT` (implemented in Phase 2.0)
- **Post-deployment monitoring:** all queries/responses logged locally, weekly review

---

## Timeline summary

```
Phase 1  ████████████████████  DONE   Foundation stability
Phase 2  ████░░░░░░░░░░░░░░░░  WK1-4  RAG hardening + KB expansion
Phase 3  ░░░░████████░░░░░░░░  WK5-8  LLM quality + safety
Phase 4  ░░░░░░░░████████░░░░  WK9-12 Test suite 250+
Phase 5  ░░░░░░░░░░░░████░░░░  WK13-15 Pi5 optimization
Phase 6  ░░░░░░░░░░░░░░░████→ WK16+  Safety validation (ongoing)
```

Total estimated: 16–20 weeks from Phase 1 completion.
