# O.A.S.I.S. System Flowcharts

---

## 1. For General Users & Investors

How O.A.S.I.S. turns a spoken emergency question into step-by-step spoken guidance — **entirely offline**.

```mermaid
flowchart TD
    A(["🟢 Press & Hold\nthe Button"])
    B(["🎙️ Speak Your Emergency\n&quot;My friend is not breathing&quot;"])
    C(["👂 Device Listens\nConverts speech to text"])
    D{{"🧠 AI Understands\nWhat kind of emergency is this?"}}
    E(["📋 Finds the Right\nFirst-Aid Protocol\nfrom built-in knowledge"])
    F(["✍️ Generates\nStep-by-Step Guidance"])
    G(["🔊 Speaks the Answer\nAloud to You"])

    A --> B --> C --> D
    D -->|"Recognized emergency\ne.g. cardiac arrest, burns, choking"| E
    D -->|"Unclear — asks one\nclarifying question"| B
    D -->|"Not a medical topic"| G2(["💬 'I am a first-aid assistant.\nPlease describe a medical emergency.'"])
    E --> F --> G

    style A fill:#22c55e,color:#fff,stroke:none
    style B fill:#3b82f6,color:#fff,stroke:none
    style C fill:#3b82f6,color:#fff,stroke:none
    style D fill:#f59e0b,color:#fff,stroke:none
    style E fill:#8b5cf6,color:#fff,stroke:none
    style F fill:#8b5cf6,color:#fff,stroke:none
    style G fill:#10b981,color:#fff,stroke:none
    style G2 fill:#6b7280,color:#fff,stroke:none
```

### Key Points for Investors

| | |
|---|---|
| **Fully offline** | No internet required. Works in wilderness, disaster zones, power outages. |
| **< 3 second response** | Optimized for Raspberry Pi 5 — from button press to first spoken word. |
| **32 emergency categories** | CPR, choking, bleeding, burns, allergic reactions, mental health crises, and more. |
| **Voice in, voice out** | No screen or typing needed — works with gloves, in darkness, under stress. |

---

## 2. For Developers

Full technical pipeline — from button press to TTS output.

```mermaid
flowchart TB

    subgraph HW ["Hardware / Entry"]
        direction TB
        BTN["🔘 Button Press\n(GPIO / PyQt5)"]
        MIC["🎙️ Audio Recording\nWAV file"]
        BTN --> MIC
    end

    subgraph ASR ["Speech-to-Text"]
        W["Whisper STT\n(local, faster-whisper)\nWAV → raw text"]
    end

    subgraph CLASSIFY ["oasis-classify  :5002  (Flask)"]
        direction TB
        NORM["normalize()\nlowercase · strip punctuation\nASR noise removal"]

        subgraph T0 ["Tier 0 — Fast Match  ~0ms"]
            T0A["Tier 0A\nshort_queries.json\n(word count ≤ 3)"]
            T0B["Tier 0B\nsentence_matches.json\n(word count > 3)"]
        end

        subgraph T1 ["Tier 1 — Semantic Classifier  ~10-100ms"]
            EMB["gte-small embed\n384-dim vector"]
            COS["Cosine similarity\nto 33 centroids\n(32 categories + OOD)"]
            EMB --> COS
        end

        SCORE{"Score\nrouting"}

        OOD["ood_response\npre-baked text"]
        TRIAGE["triage_prompt\ntriage.py\nClarifying question"]
        HINT["Store triage hint\n60s TTL →\nnext query boosted"]
        MANUAL["Manual lookup\ndata/manuals/*.txt\n80-140 tokens"]
        PROMPT["prompt_builder.py\nSystem prompt\n~200-350 tokens total"]
        LLP["llm_prompt\nsystem_prompt ready"]

        NORM --> T0A & T0B
        T0A & T0B -->|"No match"| T1
        COS --> SCORE
        SCORE -->|"score < 0.30\nor OOD cluster best"| OOD
        SCORE -->|"0.30 ≤ score < 0.65"| TRIAGE
        TRIAGE --> HINT
        SCORE -->|"score ≥ 0.65"| MANUAL
        MANUAL --> PROMPT --> LLP
    end

    subgraph LLM ["Ollama  :11434"]
        direction TB
        MSG["messages:\n[system: prompt]\n[user: query]"]
        GM["gemma3:1b\nstreaming /api/chat\nnum_predict=200 · temp=0.25\nnum_ctx=512"]
        TOK["Token stream"]
        MSG --> GM --> TOK
    end

    subgraph OUT ["Output Layer"]
        direction LR
        SAN["sanitize_chunk()\nstrip ** ` #"]
        DISP["Display\nPyQt5 ChatWidget\n/ Whisplay HAT LED"]
        TTS["Piper TTS\nText → WAV → Speaker"]
        LOG["log_response()\noasis_logs/YYYY-MM-DD.jsonl"]
        SAN --> DISP & TTS & LOG
    end

    MIC --> W
    W -->|"raw text"| NORM

    T0A & T0B -->|"direct_response\npre-baked text"| SAN
    OOD --> SAN
    HINT --> LLP
    LLP --> MSG
    TRIAGE --> MSG
    TOK --> SAN

    %% TypeScript path note
    TS["TypeScript Layer\nsrc/core/ChatFlow.ts\nOasisAdapter.ts\n(same classify + LLM calls)"]
    TS -.->|"also drives\noasis-classify :5002"| CLASSIFY

    style HW fill:#1e293b,color:#e2e8f0,stroke:#475569
    style ASR fill:#1e3a5f,color:#e2e8f0,stroke:#3b82f6
    style CLASSIFY fill:#1a1a2e,color:#e2e8f0,stroke:#8b5cf6
    style T0 fill:#2d1b69,color:#e2e8f0,stroke:#7c3aed
    style T1 fill:#2d1b69,color:#e2e8f0,stroke:#7c3aed
    style LLM fill:#14532d,color:#e2e8f0,stroke:#22c55e
    style OUT fill:#1c1917,color:#e2e8f0,stroke:#78716c
    style TS fill:#0c1445,color:#93c5fd,stroke:#3b82f6,stroke-dasharray:5 5
```

### Component Map

| Component | File | Port | Language |
|---|---|---|---|
| PyQt5 GUI | `python/oasis-gui/main.py` | — | Python |
| Pipeline orchestration | `python/oasis-gui/core/pipeline_worker.py` | — | Python |
| Classify service | `python/oasis-classify/service.py` | :5002 | Python / Flask |
| Tier 0 fast match | `python/oasis-classify/fast_match.py` | — | Python |
| Tier 1 classifier | `python/oasis-classify/classifier.py` | — | Python (gte-small) |
| Prompt assembly | `python/oasis-classify/prompt_builder.py` | — | Python |
| TypeScript orchestrator | `src/core/ChatFlow.ts` | — | TypeScript |
| TypeScript adapter | `src/core/OasisAdapter.ts` | — | TypeScript |
| Classify HTTP client (TS) | `src/cloud-api/local/oasis-classify-client.ts` | — | TypeScript |
| Classify HTTP client (Py) | `python/oasis-gui/clients/classify_client.py` | — | Python |
| LLM HTTP client (Py) | `python/oasis-gui/clients/llm_client.py` | :11434 | Python |

### Dispatch Mode Decision Table

| Score | Path | Mode | LLM called? |
|---|---|---|---|
| Tier 0 match | `tier0_short` / `tier0_sentence` | `direct_response` | No |
| < 0.30 or OOD cluster | `ood_floor` / `ood_cluster` | `ood_response` | No |
| 0.30 – 0.65 | `triage` | `triage_prompt` | Yes — asks clarifying Q |
| ≥ 0.65 | `classifier_hit` | `llm_prompt` | Yes — step-by-step answer |
