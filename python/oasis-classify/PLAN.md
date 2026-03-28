# OASIS-Classify: Medical Intent Classification + Pre-generated Manual Dispatch

## Problem

On Raspberry Pi 5, the Gemma 1b model takes ~16s to first token because the full RAG prompt (system template + retrieved chunks + context injection + user query) is too long. The RAG retrieval itself completes in ~1s. **Prompt length is the bottleneck.**

First-aid is a closed domain — there are ~32 distinct emergency types with standard protocols. Most queries map cleanly to one known scenario. Feeding raw WHO/Red Cross manual text through a retrieve-rank-compress pipeline produces accurate but verbose context that a 1b model processes slowly.

---

## Solution: Classify → Lookup → Compact Prompt

Replace multi-stage RAG retrieval with:
1. **Fast-path check** — exact match (short keyword or known sentence) → pre-written response, skip LLM entirely
2. **Classify** — detect medical intent category from user query
3. **Lookup** — fetch a pre-written compact manual for that category
4. **Prompt** — build a minimal prompt (~200-350 tokens total)

```
User Query
     |
     v
+---------------------------------------+
|  Tier 0A (word count <= 3)            |
|  SHORT_QUERIES exact dict lookup      |
|  ASR-robust normalization applied     |
+------------------+--------------------+
                   | no match
                   v
+---------------------------------------+
|  Tier 0B (word count > 3)             |
|  SENTENCE_MATCHES exact dict lookup   |
+------------------+--------------------+
                   | no match
                   v
+---------------------------------------+
|  Tier 1: Medical Intent Classifier    |
|  gte-small embed -> centroid match    |
|  33 centroids: 32 medical + OOD       |
+------------------+--------------------+
                   |
      +------------+------------+
      |            |            |
  score < 0.30  OOD cluster  0.30 <= score < 0.65
  (hard floor)    hit              (triage band)
      |            |                    |
      +-----+------+                    v
            |                    Triage Protocol
            v                    LLM clarifying Q
     OOD Response
     (pre-baked)         score >= 0.65
                              |
                              v
                       Manual Lookup
                       -> Compact Prompt
                       -> Gemma 1b
```

### Comparison

| Metric | Current (oasis-rag) | Proposed (oasis-classify) |
|--------|---------------------|---------------------------|
| Approach | Query -> FAISS -> re-rank -> compress -> LLM | Match/Classify -> manual lookup -> LLM (or skip) |
| Prompt tokens | ~500-800 | ~200-350 (0 for exact match) |
| Context source | Dynamic retrieval from 374 chunks | Static pre-written manual per category |
| Est. TTFT (Pi5) | ~16s | ~3-6s (0ms for exact match) |
| Coverage | Any query (open-ended) | 32 categories + triage + OOD |
| Low confidence | Generic "no info found" prompt | Triage: targeted clarifying questions |

**oasis-classify is a standalone system.** It does not depend on or fall back to oasis-rag.

---

## Component Design

### 0. Tier 0 — Pre-classifier Fast Path

Before the classifier runs, check if the query can be resolved instantly with no LLM call at all.

**Two branches based on word count:**

| Query length | Match method | Fallthrough if no match |
|---|---|---|
| ≤ 3 words | Word / keyword exact match | Classifier |
| > 3 words | Full sentence exact match against stored list | Classifier |

The type of matching changes with word count — this prevents keyword collisions on full sentences (e.g., "I watched a video on CPR but my friend is choking" must not match the CPR keyword).

#### ASR-robust normalization (applied before all Tier 0 lookups)

This is a voice-first system. Whisper STT introduces predictable noise patterns that must be normalized before matching:

```python
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)        # strip punctuation
    text = re.sub(r"\s+", " ", text)            # collapse whitespace

    # Number word normalization
    text = text.replace("nine one one", "911")
    text = text.replace("nine eleven", "911")

    # Common ASR homophone map
    HOMOPHONES = {
        "bleed in": "bleeding", "bled in": "bleeding",
        "seizing": "seizure",   "sieze": "seizure",
        "strock": "stroke",     "stroak": "stroke",
        "heart tack": "heart attack",
    }
    for wrong, right in HOMOPHONES.items():
        text = text.replace(wrong, right)

    # Repeated-letter collapse (ASR stress artifacts: "heeelp" -> "help")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    return text
```

**Normalization is applied once at pipeline entry and propagated through all tiers.** The normalized query string is used for Tier 0 lookups *and* passed to gte-small for embedding in Tier 1. ASR noise (homophones, repeated letters) degrades embedding quality; cleaning before encoding produces more accurate centroid similarity scores.

**Edit distance 1 tolerance for short queries only (≤ 6 chars after normalization):**
For short queries, also check edit distance 1 against all keys in `SHORT_QUERIES`. This catches single-character ASR substitutions like "hrlp" → "help" or "cpr" misheard as "spr". Not applied to long queries or sentence matching — too many false positives.

```python
def tier0_lookup(query: str) -> str | None:
    norm = normalize(query)
    words = norm.split()

    if len(words) <= TIER0_MAX_WORDS:
        # Exact match
        if norm in SHORT_QUERIES:
            return SHORT_QUERIES[norm]
        # Edit distance 1 for very short tokens
        if len(norm) <= 6:
            for key in SHORT_QUERIES:
                if editdistance(norm, key) == 1:
                    return SHORT_QUERIES[key]
    else:
        # Sentence exact match only
        if norm in SENTENCE_MATCHES:
            return SENTENCE_MATCHES[norm]

    return None
```

#### Branch A: Short queries (≤ 3 words)

```python
SHORT_QUERIES = {
    "help":       GENERIC_HELP_RESPONSE,
    "sos":        GENERIC_HELP_RESPONSE,
    "emergency":  GENERIC_HELP_RESPONSE,
    "911":        CALL_911_RESPONSE,
    "cpr":        "cpr",           # routes to manual
    "tourniquet": "bleeding",
    "burns":      "burns",
    "choking":    "choking",
    "seizure":    "seizure",
    "bleeding":   "bleeding",
    "stroke":     "stroke",
    "drowning":   "drowning",
    # ... ~50-80 entries
}
```

#### Branch B: Long queries (> 3 words)

```python
SENTENCE_MATCHES = {
    "what do i do if someone is not breathing":       "cpr",
    "someone collapsed and is not breathing":         "cpr",
    "how do i stop the bleeding":                     "bleeding",
    "there is so much blood i dont know what to do":  "bleeding",
    "my friend is choking and cant breathe":          "choking",
    "what do i do for a bad burn":                    "burns",
    "someone is seriously hurt what do i do":         GENERIC_HELP_RESPONSE,
    # ... ~50-100 entries
}
```

If no match in either branch → classifier.

---

### 1. Medical Intent Classifier

#### Phase 1: Centroid-based (no training required)

Reuses the gte-small model already loaded in memory.

1. Define 32 medical categories + 1 `out_of_domain` cluster, each with 15-30 prototype queries
2. At build time: embed all prototypes with gte-small, compute centroid embedding per category
3. At runtime: embed query → cosine similarity to all 33 centroids → route by score

```python
query_vec = gte_small.encode(query)               # (384,)
scores = cosine_similarity(query_vec, centroids)  # (33,)
best_idx = argmax(scores)
best_score = scores[best_idx]
best_category = categories[best_idx]

if best_score < OOD_FLOOR:
    return DispatchResult(mode="ood_response", ...)
elif best_category == "out_of_domain":
    return DispatchResult(mode="ood_response", ...)
elif best_score >= CLASSIFY_THRESHOLD:
    return DispatchResult(mode="llm_prompt", category=best_category, ...)
else:   # [OOD_FLOOR, CLASSIFY_THRESHOLD)
    return DispatchResult(mode="triage_prompt", ...)
```

**OOD cluster prototypes:**
- "what time is it", "how do I boil eggs", "what's the weather", "tell me a joke",
  "who won the game", "play some music", "set a timer", "what is 2 plus 2"

**Two-layer OOD defense:** OOD cluster handles casual chat gracefully; hard floor catches gibberish and foreign language that lands nowhere near any cluster.

#### Phase 2: Trained linear head

If centroid accuracy is insufficient:
1. Keep gte-small frozen (preserves compatibility)
2. Train `LogisticRegression` or `nn.Linear(384, N_CATEGORIES)` on gte-small embeddings
3. Training data: prototype queries + LLM-augmented paraphrases (1000+ per run)
4. Serialize as pickle/ONNX for Pi5 deployment

#### Phase 3 (optional): Tiny dedicated classifier

If Phase 2 still insufficient: fine-tune gte-small classification variant or TinyBERT/MiniLM with classification head (~56-67MB). Verify total pipeline ≤ 4.5GB before deploying.

---

### 2. /dispatch Response Schema

The endpoint returns a strict typed response. The TypeScript caller must handle all four modes.

```python
@dataclass
class DispatchResult:
    mode: Literal["direct_response", "llm_prompt", "triage_prompt", "ood_response"]

    # mode == "direct_response": Tier 0 hit — return this text directly, no LLM
    # mode == "llm_prompt":      classifier hit — send system_prompt to LLM
    # mode == "triage_prompt":   ambiguous — send system_prompt to LLM (clarifying Q)
    # mode == "ood_response":    off-topic — return this text directly, no LLM

    response_text:  str | None   # for direct_response and ood_response
    system_prompt:  str | None   # for llm_prompt and triage_prompt

    # Telemetry (always populated)
    category:       str | None   # top-1 category ID.
                                 # triage_prompt: ALWAYS set (TS stores as hint).
                                 # ood_response / direct_response: None.
    top3:           list[dict]   # [{"category": str, "score": float}, ...] — top-3 from classifier.
                                 # JSON object entries, not Python tuples.
    score:          float | None # top-1 score (None for Tier 0 hits)
    threshold_path: str          # server-side: "tier0_short" | "tier0_sentence" | "classifier_hit"
                                 #              | "triage" | "ood_floor" | "ood_cluster"
                                 # client-side (synthesized on failure): "network_error"
                                 #              | "service_error" | "invalid_schema"
    latency_ms:     float
    hint_changed_result: bool    # True if triage hint changed the top-1 outcome
```

**TypeScript adapter decision table:**

| mode | ChatFlow behavior |
|------|-------------------|
| `direct_response` | Speak `response_text` directly via TTS. No LLM call. Clear triage hint. |
| `llm_prompt` | Send `system_prompt` to LLM → speak result via TTS. Clear triage hint. |
| `triage_prompt` | Send `system_prompt` to LLM → speak result via TTS. Store `category` as new triage hint with TTL. |
| `ood_response` | Speak `response_text` directly via TTS. No LLM call. Clear triage hint. |

**Client contract:** The classify client must resolve, never reject (same pattern as `oasis-rag-client.ts`). Synthesize a fallback `DispatchResult` locally — never surface an exception to `ChatFlow`:

| Error condition | Synthesized `threshold_path` | Behavior |
|---|---|---|
| Network error (timeout, refused) | `"network_error"` | `mode="ood_response"`, generic help message |
| HTTP 5xx or unexpected status | `"service_error"` | `mode="ood_response"`, generic help message |
| Response body missing required field | `"invalid_schema"` | `mode="ood_response"`, generic help message |

Using distinct `threshold_path` values keeps infrastructure failures visible in logs and separate from real OOD traffic.

---

### 3. Category Definitions (32 categories)

#### Life-threatening emergencies (12)

| # | Category ID | Description | KB Source | Injection Signal |
|---|-------------|-------------|-----------|-----------------|
| 1 | `cpr` | Cardiac arrest, CPR protocol | cpr.md | CARDIAC_ARREST |
| 2 | `choking` | Airway obstruction, Heimlich | airway.md | CHOKING |
| 3 | `bleeding` | Severe bleeding, hemorrhage control | wounds_and_bleeding.md | PANIC_BLOOD |
| 4 | `anaphylaxis` | Severe allergic reaction | bites_and_stings.md | NO_EPIPEN |
| 5 | `stroke` | Stroke, FAST assessment | stroke.md | STROKE |
| 6 | `heart_attack` | Heart attack / ACS (not cardiac arrest) | chest_pain_cardiac.md | HEART_ATTACK |
| 7 | `drowning` | Submersion, near-drowning | submersion.md | DROWNING |
| 8 | `electric_shock` | Electrocution, live wire | electric_shock.md | ELECTRIC_SHOCK |
| 9 | `poisoning` | Poisoning, toxic ingestion | poisoning_overdose.md | POISONING |
| 10 | `opioid_overdose` | Opioid OD, naloxone protocol | poisoning_overdose.md | -- |
| 11 | `chest_wound` | Pneumothorax, sucking chest wound | breathing.md, trauma.md | -- |
| 12 | `spinal_injury` | Spine/neck trauma, do not move | trauma.md | SPINAL |

#### Trauma and injury (8)

| # | Category ID | Description | KB Source | Injection Signal |
|---|-------------|-------------|-----------|-----------------|
| 13 | `fracture` | Broken bones, splinting | bone_and_joint.md | FRACTURE |
| 14 | `sprain_dislocation` | Sprains, strains, dislocations, RICE | bone_and_joint.md | -- |
| 15 | `head_injury` | Concussion, head trauma | trauma.md | -- |
| 16 | `impaled_object` | Penetrating foreign body, do not remove | trauma.md | IMPALED_OBJECT |
| 17 | `abdominal_injury` | Abdominal trauma, evisceration | special_situations.md | -- |
| 18 | `burns` | Thermal/chemical burns | burns.md | BURN |
| 19 | `eye_injury` | Chemical splash, foreign body, impaled eye | special_situations.md | EYE_CHEMICAL |
| 20 | `dental_injury` | Knocked-out tooth, mouth injury | special_situations.md | -- |

#### Environmental emergencies (5)

| # | Category ID | Description | KB Source | Injection Signal |
|---|-------------|-------------|-----------|-----------------|
| 21 | `hypothermia` | Cold exposure, hypothermia | cold_emergencies.md | HYPOTHERMIA |
| 22 | `frostbite` | Frostbite, cold injury to extremities | cold_emergencies.md | FROSTBITE |
| 23 | `heat_stroke` | Heat stroke / heat exhaustion | heat_emergencies.md | HEAT_STROKE |
| 24 | `lightning` | Lightning safety / injury | lightning.md | LIGHTNING |
| 25 | `altitude_sickness` | AMS, HACE, HAPE | altitude.md | -- |

#### Medical emergencies (5)

| # | Category ID | Description | KB Source | Injection Signal |
|---|-------------|-------------|-----------|-----------------|
| 26 | `seizure` | Seizure, epilepsy, convulsions | seizure_epilepsy.md | SEIZURE |
| 27 | `diabetic` | Diabetic emergency, hypo/hyperglycemia | diabetic_emergency.md | -- |
| 28 | `asthma` | Asthma attack, no inhaler | breathing.md | ASTHMA |
| 29 | `shock` | Hypovolemic/circulatory shock | shock.md | SHOCK |
| 30 | `fainting` | Syncope, vasovagal episode | mental_status.md | -- |

#### Other (2)

| # | Category ID | Description | KB Source | Injection Signal |
|---|-------------|-------------|-----------|-----------------|
| 31 | `bites_and_stings` | Snake, spider, scorpion, bee, tick, dog/animal bites | bites_and_stings.md | SNAKEBITE |
| 32 | `pediatric_emergency` | Infant/child CPR, febrile seizure | pediatric_emergency.md | INFANT_CPR |

#### Special: Tier 0 only (not classifier categories)

`panic_hyperventilation`, `childbirth`, `nosebleed` — handled via Tier 0 keyword match. Too rare or simple to warrant a centroid; adding them would blur nearby clusters.

---

### 4. Pre-generated Manuals

Each category gets a single compact manual — pre-written, human-verified, action-oriented.

#### Format

```
Category: [Category Name]

STEPS:
1. [Most critical first action].
2. [Second action].
...

NEVER DO:
- Do NOT [common myth or dangerous mistake].
- Do NOT [another myth].
...
```

- Plain text only — no markdown. The 1b model copies markdown if it sees it.
- 150-250 tokens per manual.
- Both STEPS and NEVER DO are required.
- Content distilled from `data/knowledge/` source documents + `context_injector.py` protocol texts.

#### Example: `bleeding.txt`

```
Category: Severe Bleeding / Hemorrhage Control

STEPS:
1. Call emergency services (911/999/112) immediately.
2. Apply DIRECT PRESSURE to the wound with any clean cloth or your hands.
3. Press hard and do not release. Maintain pressure for at least 10 minutes.
4. If blood soaks through, add more cloth on top. Do not remove the first layer.
5. If bleeding from a limb and direct pressure fails, apply a tourniquet 5-7 cm above the wound.
6. Tighten the tourniquet until bleeding stops completely. Note the time.
7. Lay the person flat. Elevate legs if no spinal injury suspected.
8. Monitor for shock: pale, cold, clammy skin, rapid weak pulse.
9. Keep the person warm and still until help arrives.

NEVER DO:
- Do NOT remove embedded objects. Press around them, not on them.
- Do NOT apply a tourniquet over a joint.
- Do NOT remove the first layer of cloth to check the wound.
- Do NOT use a belt as a tourniquet — it is too wide and does not tighten enough.
```

---

### 5. Multi-label Handling

Some queries involve multiple conditions: "bleeding from a broken leg", "baby not breathing after drowning".

**Strategy: primary manual + short "also check" block, not two full manuals.**

Rationale: two full manuals create conflicting or overloaded instructions under stress. One authoritative protocol with a brief secondary note is safer and more readable.

```python
ALSO_CHECK_TEMPLATE = "ALSO CHECK: {secondary_category} — {one_line_summary}"
```

**One-line summaries are curated, not generated.** Each category has a pre-written one-liner stored in `data/also_check_summaries.json`. `prompt_builder.py` loads this at startup and looks up the secondary category by ID. No medical text is invented at runtime.

```json
{
  "fracture": "Immobilize the limb in its current position while controlling bleeding.",
  "cpr": "If the person becomes unresponsive and stops breathing, start CPR immediately.",
  ...
}
```

**Rules:**
- Classifier returns top-2 categories
- If secondary score >= primary × `MULTI_LABEL_RATIO (0.80)`: append `ALSO_CHECK_TEMPLATE` after primary manual
- Secondary summary is pulled from `also_check_summaries.json` (≤ 20 tokens per entry, human-written)
- Hard token ceiling: combined prompt must not exceed `MAX_PROMPT_TOKENS (400)`. If it does, drop the secondary block entirely — primary protocol takes full precedence
- Priority order: `PRIORITY_CRITICAL` > `PRIORITY_URGENT` > classifier score

**Example combined output:**

```
Category: Severe Bleeding / Hemorrhage Control

STEPS:
1. Call emergency services...
...

NEVER DO:
...

ALSO CHECK: fracture — Immobilize the limb in its current position while controlling bleeding.
```

---

### 6. Triage Protocol

Triage activates only for the ambiguous medical band: `OOD_FLOOR (0.30) <= score < CLASSIFY_THRESHOLD (0.65)`.

```python
TRIAGE_PROMPT = """You are OASIS, a first-aid assistant.
You could not identify the specific emergency type.
Ask ONE short clarifying question to understand the situation better.
If the person sounds panicked about a life-threatening situation, tell them to call emergency services (911/999/112) immediately.

User said: {query}
Response:"""
```

~60 tokens. The next user message (with more detail) should classify at or above threshold.

---

### 7. Conversation Context + Triage Hint

**Design: stateless single-turn with triage carry-forward.**

- No conversation history in the prompt — 0 extra tokens
- The triage hint is a lightweight bias, not context: `prev_triage_hint: str | None`

```python
def classify(query: str, prev_triage_hint: str | None = None) -> DispatchResult:
    scores = cosine_similarity(query_vec, centroids)

    hint_changed_result = False
    if prev_triage_hint and prev_triage_hint in category_index:
        hint_idx = category_index[prev_triage_hint]
        hint_relevance = scores[hint_idx]           # raw cosine sim to hinted category
        if hint_relevance >= TRIAGE_HINT_MIN_RELEVANCE:
            original_top1 = argmax(scores)
            scores[hint_idx] += TRIAGE_HINT_BOOST
            hint_changed_result = (argmax(scores) != original_top1)
        # else: query has drifted too far from hint — ignore boost silently
    ...
```

**Python handles topic-shift, not TypeScript.** If the new query embeds far from the hinted category (e.g., hint was `burns` but query is about choking), the raw cosine similarity to the hint centroid will already be low. The `TRIAGE_HINT_MIN_RELEVANCE` gate skips the boost automatically — no lexical analysis needed in TypeScript.

**Triage hint state — TypeScript responsibilities (declarative only):**

`OasisAdapter.ts` or `ChatFlow.ts` owns `triageHint: { category: string, expiresAt: number } | null`.

Clear the hint only on these conditions:
1. TTL expires (60 seconds of silence)
2. Service returns `mode == "llm_prompt"` — triage resolved, high-confidence result received

Always pass `prev_triage_hint` to the `/dispatch` payload as-is. Do not inspect query text, body-part words, or correction phrases in TypeScript — that analysis belongs in Python.

Log `hint_changed_result: true` responses to monitor how often the hint is changing outcomes. If the rate is unexpectedly high, the boost value, TTL, or `TRIAGE_HINT_MIN_RELEVANCE` may need adjustment.

---

### 8. Prompt Template

```
You are OASIS, a first-aid assistant.
Rules: Follow the MANUAL steps exactly. Numbered list only. One sentence per step. No extra text.

MANUAL:
{manual}

QUESTION: {query}
RESPONSE:
```

~35 tokens template + ~150-250 tokens manual + ~15-30 tokens query = **~200-315 tokens total**.

---

## Threshold Calibration

Static default thresholds (`CLASSIFY_THRESHOLD = 0.65`, `OOD_FLOOR = 0.30`) are starting points. Calibration is required before production deployment.

### Calibration script: `training/calibrate_thresholds.py`

**Inputs:**
- Held-out query set (~200 queries, labelled with ground-truth category)
- OOD query set (~50 queries, labelled "ood")
- Triage query set (~30 queries, labelled "triage" — genuinely ambiguous inputs)

**Metrics computed per threshold sweep:**

| Metric | Target |
|--------|--------|
| Per-category precision/recall | All life-critical categories: recall >= 0.95 |
| OOD false-positive rate | < 5% (medical query misclassified as OOD) |
| OOD false-negative rate | < 10% (non-medical query reaches triage instead of OOD) |
| Triage rate | 5-15% of real queries — below 5% means overconfident, above 15% is poor UX |
| Overall accuracy | >= 90% (Phase 1), >= 95% (Phase 2) |

**Triage rate band:** The triage rate is a UX/safety dial. Too low means the system confidently picks wrong categories. Too high means users are constantly asked clarifying questions instead of getting help. Target 5-15%.

**Per-category thresholds:** Some category pairs are systematically confusing and may need individual thresholds:
- `heart_attack` vs `cpr` — conscious chest pain vs cardiac arrest
- `shock` vs `bleeding` — both present together frequently
- `hypothermia` vs `frostbite` — related but different protocols
- `poisoning` vs `opioid_overdose` — different treatments

Store per-category thresholds in config as `CATEGORY_THRESHOLDS: dict[str, float]`, falling back to `CLASSIFY_THRESHOLD` for categories without a custom value.

### Replay harness: `training/replay_harness.py`

Before any threshold or model update, run the replay harness against anonymized logs from prior sessions. Flags regressions: queries that previously classified correctly and no longer do. Prevents silent accuracy degradation across updates.

---

## Folder Structure

```
python/oasis-classify/
|-- PLAN.md                          # This file
|-- CLAUDE.md                        # Agent guidance for this folder
|-- config.py                        # Thresholds, paths, model config
|-- categories.py                    # Category definitions + metadata
|-- fast_match.py                    # Tier 0: word match + sentence match + ASR normalization
|-- classifier.py                    # Tier 1: centroid-based classifier
|-- manual_store.py                  # Load and serve pre-generated manuals
|-- triage.py                        # Triage prompt template
|-- prompt_builder.py                # Build compact LLM prompt + multi-label resolver
|-- service.py                       # Flask API (:5002), POST /dispatch
|-- build_centroids.py               # Offline: compute centroid embeddings
|-- data/
|   |-- prototypes.json              # Category -> list of prototype queries
|   |-- centroids.npy                # Precomputed centroid embeddings (build artifact)
|   |-- short_queries.json           # Tier 0A: word/keyword matches (<=3 words)
|   |-- sentence_matches.json        # Tier 0B: full sentence matches (>3 words)
|   |-- also_check_summaries.json    # Per-category curated one-liners for multi-label block
|   +-- manuals/                     # Pre-generated manual text files (32 files)
|-- training/
|   |-- generate_data.py             # Synthetic training data via LLM augmentation
|   |-- train_classifier.py          # Train linear head on gte-small embeddings
|   |-- calibrate_thresholds.py      # Threshold sweep: precision/recall/triage-rate metrics
|   +-- replay_harness.py            # Regression check against prior session logs
+-- tests/
    |-- test_fast_match.py           # Tier 0: exact match, sentence match, word count guard, ASR normalization
    |-- test_classifier.py           # Category accuracy, OOD detection, triage band
    |-- test_manuals.py              # Manual format validation (STEPS + NEVER DO, token range)
    |-- test_integration.py          # End-to-end: all four dispatch modes
    |-- test_contract.py             # /dispatch schema stability (TypeScript contract)
    +-- test_adversarial.py          # Mixed-topic, ASR noise, nonsense, foreign language, canary recall
```

---

## Implementation Phases

### Phase 1: Core System

**Goal:** Working prototype — Tier 0 + centroid classifier + manuals + strict /dispatch schema.

1. `config.py` — all thresholds and paths
2. `categories.py` — 32 categories with metadata and priority levels
3. `data/prototypes.json` — 15-30 prototype queries per category, English only
4. `data/manuals/*.txt` — 32 compact manuals, STEPS + NEVER DO format
5. `data/short_queries.json` + `data/sentence_matches.json` — Tier 0 data
6. `build_centroids.py` — embed prototypes, compute centroids, save `.npy`
7. `fast_match.py` — Tier 0 with ASR normalization + edit distance 1 for short tokens
8. `classifier.py` — centroid cosine match, returns `DispatchResult`
9. `manual_store.py` — load manuals at startup
10. `prompt_builder.py` — compact prompt + multi-label conflict resolver + token ceiling
11. `triage.py` — triage prompt template
12. `service.py` — `POST /dispatch` with full typed response schema
13. `tests/test_fast_match.py`, `test_classifier.py`, `test_manuals.py`, `test_integration.py`, `test_contract.py`

**Target:** 90%+ classification accuracy, TTFT < 6s on Pi5, /dispatch schema stable.

### Phase 2: Trained Classifier + Calibration

**Goal:** Improve accuracy and establish calibrated thresholds.

1. `training/generate_data.py` — 1000+ synthetic queries via LLM augmentation
2. `training/train_classifier.py` — LogisticRegression on gte-small embeddings
3. `training/calibrate_thresholds.py` — sweep global + per-category thresholds
4. Establish per-category thresholds for confusing pairs
5. Serialize model for Pi5 deployment (pickle or ONNX)
6. `tests/test_adversarial.py` — adversarial + canary tests

**Target:** 95%+ accuracy, triage rate 5-15%, life-critical category recall >= 0.95.

### Phase 3: Integration + Observability

**Goal:** Connect to the chatbot, instrument telemetry, measure real-world performance.

1. Flask service confirmed on `:5002`
2. Update `OasisAdapter.ts` to call `/dispatch`, handle all four modes per adapter decision table
3. TypeScript triage hint: TTL cache + clear-on-`llm_prompt` + `hint_changed_result` logging (topic-shift handled Python-side)
4. `training/replay_harness.py` — regression check before any threshold/model update
5. Benchmark on Pi5: TTFT, accuracy, triage rate
6. Tune `CLASSIFY_THRESHOLD`, `OOD_FLOOR`, per-category thresholds from real traffic

---

## Key Configuration Parameters

```python
# config.py
CLASSIFY_THRESHOLD    = 0.65   # >= this -> manual; [OOD_FLOOR, this) -> triage
OOD_FLOOR             = 0.30   # < this -> OOD response (not triage)
TIER0_MAX_WORDS       = 3      # Tier 0 word match only for queries <= this many words
MULTI_LABEL_RATIO     = 0.80   # Secondary category included if score >= primary * this
MAX_CATEGORIES        = 2      # Maximum categories per query
MAX_PROMPT_TOKENS     = 400    # Hard ceiling — drop secondary block if exceeded.
                               # Measured with tiktoken (cl100k_base): a tiny pre-compiled
                               # C-extension, ~0µs overhead, ~5% deviation from Gemma count.
                               # Do NOT load transformers.AutoTokenizer here — memory cost on Pi5.
TRIAGE_HINT_BOOST     = 0.05   # Score boost applied when prev_triage_hint matches
TRIAGE_HINT_MIN_RELEVANCE = 0.20  # Skip hint boost if query cosine sim to hint < this.
                               # Handles topic-shift in Python; TS passes hint blindly.
TRIAGE_HINT_TTL_SEC   = 60     # Hint expires after this many seconds (managed in TS)
EMBEDDING_MODEL       = "thenlper/gte-small"

# Per-category threshold overrides (falls back to CLASSIFY_THRESHOLD if not set)
CATEGORY_THRESHOLDS: dict[str, float] = {}   # populated after calibration

# Priority levels for multi-label conflict resolution
PRIORITY_CRITICAL = ["cpr", "choking", "bleeding"]
PRIORITY_URGENT   = ["anaphylaxis", "electric_shock", "poisoning", "drowning"]
```

**Score routing:**

| Score / cluster | Route |
|-----------------|-------|
| `score < OOD_FLOOR` | OOD response |
| `out_of_domain` cluster best hit | OOD response |
| `OOD_FLOOR <= score < CLASSIFY_THRESHOLD` | Triage |
| `score >= CLASSIFY_THRESHOLD` (or per-category override) | Manual → LLM prompt |

---

## Design Decisions

1. **Standalone system.** Does not depend on oasis-rag. Triage handles low-confidence, not RAG fallback.
2. **English only.** No Korean manuals or multilingual classifier.
3. **Stateless single-turn.** No history in prompt. Triage hint is a classifier bias (0 prompt tokens). TS layer owns TTL state only — topic-shift detection is handled in Python via the `TRIAGE_HINT_MIN_RELEVANCE` gate.
4. **STEPS + NEVER DO format.** Both sections required in every manual.
5. **Four-mode /dispatch response.** Strict schema with `mode` field. TypeScript adapter decision table maps each mode to deterministic `ChatFlow` behavior.
6. **Tier 0 splits by word count.** ≤ 3 words: keyword exact match + edit distance 1 (≤ 6 chars). > 3 words: sentence exact match only. No fuzzy word-overlap on full sentences.
7. **ASR normalization at pipeline entry, not just Tier 0.** `normalize(query)` runs once before dispatch. The normalized string is used for Tier 0 lookups and passed to gte-small for Tier 1 embedding — ASR noise degrades cosine scores if uncleaned.
8. **Two-layer OOD defense.** `out_of_domain` cluster + hard floor (0.30). Triage activates only in [0.30, 0.65).
9. **Multi-label: primary + "also check" block, not two full manuals.** Hard token ceiling (400) ensures prompt stays short under all conditions.
10. **Calibrated thresholds, not fixed globals.** Global defaults are starting points. Per-category thresholds tuned from held-out data + replay harness guards against regression.
11. **Also-check summaries are data, not generated text.** `data/also_check_summaries.json` holds a curated one-liner per category. `prompt_builder.py` looks up by category ID at runtime — no medical text is generated dynamically.
12. **tiktoken for the prompt ceiling, not the Gemma tokenizer.** Loading `transformers.AutoTokenizer` in the Flask service costs significant RAM/startup on Pi5. `tiktoken` (cl100k_base) is a tiny pre-compiled C-extension — microsecond overhead, near-zero RAM — and its token count is within ~5% of Gemma's. Acceptable for a soft ceiling enforced by dropping the secondary block.
13. **Client-side `threshold_path` distinguishes infrastructure from OOD.** The client synthesizes `"network_error"`, `"service_error"`, or `"invalid_schema"` into `threshold_path` on failure rather than silently mapping all errors to OOD. This keeps logs actionable.

---

## Open Questions

1. **Manual provenance (deferred):** Each manual could store source file paths, source hashes, reviewer name, and reviewed date in a sidecar `data/manuals_manifest.json`. A CI check could flag manuals whose source `data/knowledge/*.md` has changed since last review. This is a correctness guarantee for production but out of scope for prototype — implement in Phase 3 if the knowledge base is updated frequently.
2. **Context injection redundancy:** Manuals embed critical safety protocols from `context_injector.py`. This is intentional (self-contained). No context injection runs in the classify path.
