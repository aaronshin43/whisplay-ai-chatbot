# OASIS Desktop GUI Plan — PyQt5

Pi OS main display fullscreen GUI.
Hardware button → voice input → RAG+LLM → token-by-token streaming display.

---

## Architecture

```
Single Python process (oasis-gui/main.py)
├── PyQt5 QApplication (fullscreen, dark theme)
│   ├── Header: title + status indicator
│   ├── Chat area: QTextEdit (read-only, streaming tokens)
│   └── Footer: state message
│
├── GPIO thread
│   └── button press/release → pyqtSignal → state machine
│
├── Pipeline QThread (worker)
│   ├── sox recording
│   ├── ASR (Whisper HTTP / faster-whisper)
│   ├── RAG query (HTTP → Flask :5001)
│   └── Ollama stream (httpx)
│       └── each token → sanitize → pyqtSignal → ChatWidget.append_token()
│
└── TTS QThread (parallel, does NOT block display)
    ├── sentence buffer (accumulates sanitized tokens)
    └── Piper subprocess → wav → playback
```

**Token path (critical for perceived speed):**
```
Ollama JSON line → httpx iter_lines() → sanitize_chunk() → token_received.emit(str)
  → [Qt event queue, <1ms] → ChatWidget.append_token()
  → QTextCursor.insertText(token) + ensureCursorVisible()
```
No batching, no serialization. Each token appears on screen the instant it arrives.

**Memory budget:**
RAG Flask ~300MB + Ollama gemma3:1b ~1.5GB + PyQt5 GUI ~150MB + Piper ~200MB + Whisper ~300MB = **~2.5GB** (within 4.5GB)

---

## Environment Variables

The GUI reads from the project `.env` file (same as the Node.js chatbot):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_ENDPOINT` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `gemma3:1b` | LLM model name |
| `OASIS_RAG_SERVICE_URL` | `http://localhost:5001` | RAG Flask service |
| `OASIS_RAG_TIMEOUT_MS` | `5000` | RAG HTTP timeout |
| `PIPER_BINARY_PATH` | — | Piper TTS binary |
| `PIPER_MODEL_PATH` | — | Piper voice model (.onnx) |
| `SOUND_CARD_INDEX` | `1` | ALSA device index for recording & playback |
| `WHISPER_HOST` | `localhost` | Whisper ASR server host |
| `WHISPER_PORT` | `8804` | Whisper ASR server port |
| `WHISPER_MODEL_SIZE_OR_PATH` | `tiny` | Local Whisper model (if not using HTTP) |

---

## File Structure

```
python/oasis-gui/
├── main.py                  # Entry point: QApplication, fullscreen, wire signals
├── gui/
│   ├── main_window.py       # QMainWindow: layout (header + chat + footer)
│   ├── chat_widget.py       # QTextEdit subclass: append_token(), add_message()
│   ├── header_widget.py     # Status label + title
│   ├── footer_widget.py     # State message bar
│   └── theme.py             # QSS dark theme stylesheet
├── core/
│   ├── state_machine.py     # enum States + transitions + signal routing
│   ├── pipeline_worker.py   # QThread: record → ASR → RAG → LLM stream
│   ├── gpio_handler.py      # RPi.GPIO → pyqtSignal (button press/release)
│   └── tts_worker.py        # Sentence buffer → Piper → audio playback
├── clients/
│   ├── rag_client.py        # httpx POST to Flask :5001 /retrieve
│   ├── llm_client.py        # httpx streaming POST to Ollama /api/chat
│   ├── asr_client.py        # Whisper HTTP or local faster-whisper
│   └── audio.py             # Sox recording start/stop + audio playback
├── utils/
│   ├── sanitizer.py         # sanitize_chunk(): strip markdown from LLM tokens
│   └── logger.py            # JSONL response logging (same format as OasisAdapter)
├── run.sh                   # Launch script
└── requirements.txt         # pyqt5, httpx
```

---

## UI Layout

```
┌──────────────────────────────────────┐
│  ● OASIS                    Ready    │  Header (48px, fixed)
├──────────────────────────────────────┤
│                                      │
│  You:                                │
│  How do I treat a burn?              │
│                                      │
│  OASIS:                              │
│  1. Cool the burn under cool         │
│  running water for at least 10       │
│  minutes.                            │
│  2. Remove any clothing or jewelry   │
│  near the burn unless stuck to it.█  │  Chat area (expanding, scrollable)
│                                      │
├──────────────────────────────────────┤
│  🎤 Press and hold button to speak   │  Footer (40px, fixed)
└──────────────────────────────────────┘
```

**Resolution handling:** Layout uses percentage-based sizing and `QFontMetrics` for dynamic font scaling. Tested targets:
- Pi 7" touchscreen: 800×480
- HDMI monitor: 1920×1080

Font size base: 18px at 800×480, scales proportionally to screen width.

---

## State Machine

```
           button_press
  [idle] ────────────────► [listening]
    ▲                          │ button_release
    │                          ▼
    │                     [processing]  ← ASR + RAG
    │                          │
    │                          ▼
    │                     [streaming]   ← LLM tokens arriving
    │                          │
    │         done             │ button_press (interrupt)
    └──────────────────────────┘
```

| State | Header | Footer | Chat | Action |
|-------|--------|--------|------|--------|
| idle | "Ready" | "Press and hold button to speak" | Previous conversation | — |
| listening | "Listening..." | "Release to send" | (unchanged) | sox recording |
| processing | "Processing..." | "Recognizing..." | User text appears | ASR → RAG → LLM start |
| streaming | "Responding..." | "Press button to interrupt" | Tokens appear one-by-one | Ollama streaming |
| idle (done) | "Ready" | "Press and hold button to speak" | Full response | TTS may still play |

**Interrupt handling (button press during streaming):**
1. Set `pipeline_worker.abort_flag = True`
2. Worker checks flag in `iter_lines()` loop → breaks out, closes httpx response
3. TTS worker: clear sentence queue, kill current Piper subprocess if running, stop sox playback
4. Chat widget: append "[interrupted]" to current response
5. State → `listening` (immediately start recording new query)

---

## Critical Components from Existing System

### Chunk Sanitization (port from OasisAdapter.ts)

```python
# utils/sanitizer.py
import re

def sanitize_chunk(chunk: str) -> str:
    """Strip markdown formatting so tokens display clean and TTS doesn't speak asterisks."""
    chunk = re.sub(r'\*+', '', chunk)        # **bold** / *italic*
    chunk = re.sub(r'`+', '', chunk)          # `code` backticks
    chunk = re.sub(r'^#+\s*', '', chunk, flags=re.MULTILINE)  # ### headings
    return chunk
```

Applied to every token BEFORE display and BEFORE TTS sentence buffer.

### Ollama OASIS Parameters

```python
# clients/llm_client.py — matches existing ollama-llm.ts config
OASIS_OPTIONS = {
    "num_predict": 300,
    "temperature": 0.05,
    "repeat_penalty": 1.3,
}
OASIS_STOP_TOKENS = ["**", "Okay", "Let's", "Here's", "Note:", "Note "]

request_body = {
    "model": OLLAMA_MODEL,
    "messages": messages,
    "stream": True,
    "options": OASIS_OPTIONS,
    "stop": OASIS_STOP_TOKENS,
    "keep_alive": -1,  # keep model loaded permanently
}
```

### Chat History Reset (OASIS mode)

Each query is fully independent — no multi-turn conversation:
```python
# pipeline_worker.py — per-query message construction
messages = []
if system_prompt:
    messages.append({"role": "system", "content": system_prompt})
messages.append({"role": "user", "content": user_text})
# No history carried over. Fresh messages array every time.
```

### Safe Fallback Prompt (RAG unavailable)

```python
# clients/rag_client.py
SAFE_FALLBACK_PROMPT = """You are OASIS, an offline first-aid assistant.
The medical knowledge base is currently unavailable.
Tell the user clearly and calmly:
1. Call emergency services immediately (local emergency number).
2. Stay on the line with the dispatcher — they will guide you.
3. Do not leave the person alone.
Do not provide any specific medical instructions without the knowledge base."""

async def retrieve_system_prompt(query: str) -> str:
    try:
        resp = httpx.post(RAG_URL + "/retrieve", json={"query": query}, timeout=TIMEOUT)
        prompt = resp.json().get("system_prompt", "")
        if prompt.strip():
            return prompt
    except Exception:
        pass
    return SAFE_FALLBACK_PROMPT  # never return empty — always safe
```

### Response Logging

```python
# utils/logger.py — JSONL format matching OasisAdapter.ts
import json, re, os
from datetime import datetime

LOG_DIR = "data/oasis_logs"

def log_response(query: str, response: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    entry = json.dumps({
        "ts": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "steps": len(re.findall(r'^\d+\.', response, re.MULTILINE)),
        "has_markdown": bool(re.search(r'[*_#`]', response)),
    })
    log_file = os.path.join(LOG_DIR, f"oasis_{datetime.now():%Y-%m-%d}.jsonl")
    with open(log_file, "a") as f:
        f.write(entry + "\n")
```

---

## Implementation Steps

### Step 1 — PyQt5 window + chat widget + theme
**Goal:** Fullscreen dark window with header, chat area, footer. Hardcoded test messages.

Files: `main.py`, `gui/main_window.py`, `gui/chat_widget.py`, `gui/header_widget.py`, `gui/footer_widget.py`, `gui/theme.py`

- QMainWindow frameless + `showFullScreen()`
- ChatWidget (QTextEdit, read-only) with `append_token(str)` and `add_message(role, text)`
- Dark QSS theme with dynamic font scaling based on screen resolution
- Test on PC (no GPIO needed): `python main.py`

**Verify:** Window fills screen, text scrolls smoothly, dark theme renders correctly.

---

### Step 2 — State machine + simulated streaming
**Goal:** State transitions work. Fake tokens appear one-by-one via QTimer.

Files: `core/state_machine.py`

- Enum: `idle`, `listening`, `processing`, `streaming`
- Keyboard shortcut (Space) simulates button press/release for PC testing
- QTimer fires fake tokens every 30ms to test `append_token()` throughput
- Header/footer text updates per state table above

**Verify:** Press Space → "Listening..." → release → "Processing..." → tokens stream in → "Ready". Smooth, no flicker.

---

### Step 3 — Ollama streaming client + pre-warm
**Goal:** Real LLM tokens appear in the chat widget. Model loaded on startup.

Files: `clients/llm_client.py`, `core/pipeline_worker.py`

- **Pre-warm on startup:** Send dummy request with `num_predict: 1, stream: false, keep_alive: -1` to load model into memory. Without this, first real query takes 10-20s extra.
- PipelineWorker (QThread) with signals: `token_received(str)`, `user_text_ready(str)`, `state_changed(str)`, `error_occurred(str)`, `finished()`
- httpx streaming to Ollama with OASIS parameters (num_predict=300, temperature=0.05, repeat_penalty=1.3, stop tokens)
- `sanitize_chunk()` applied to each token before emitting signal
- `abort_flag` checked each iteration for interrupt support
- Test: hardcode a query string, see tokens stream in GUI

**Verify:** Tokens appear one-by-one. First token visible within ~500ms of Ollama response start. Second query fast (model warm).

---

### Step 4 — RAG client + safe fallback
**Goal:** System prompt from RAG Flask injected before LLM call.

Files: `clients/rag_client.py`, `utils/sanitizer.py`, `utils/logger.py`

- httpx POST to `http://localhost:5001/retrieve` with `{"query": text, "top_k": 4, "compress": true}`
- Extract `system_prompt` from response
- If RAG unavailable or returns empty → use `SAFE_FALLBACK_PROMPT` (never empty, never unsafe)
- After LLM stream completes → `log_response(query, full_text)` in JSONL
- Fresh messages array per query (no history carryover)

**Verify:** Medical question → response uses OASIS knowledge. Kill RAG Flask → response safely directs to emergency services.

---

### Step 5 — GPIO button handler
**Goal:** Physical button controls state machine on Pi.

Files: `core/gpio_handler.py`

- RPi.GPIO, BOARD pin 11, pull-up, 50ms debounce
- `button_pressed` / `button_released` pyqtSignals
- Connected to state machine transitions
- Keyboard Space still works as fallback (for PC dev)
- Conditional import: `try: import RPi.GPIO except ImportError: GPIO = None` so it runs on PC

**Verify:** On Pi, press button → listening, release → processing. Same flow as Step 2 but real hardware.

---

### Step 6 — Audio recording + ASR
**Goal:** Voice input converted to text, displayed in chat.

Files: `clients/audio.py`, `clients/asr_client.py`

- Sox recording: `sox -d -r 16000 -c 1 -t wav {path}` using ALSA device from `SOUND_CARD_INDEX`
- Start on button_press, terminate on button_release
- ASR via Whisper HTTP POST (`http://{WHISPER_HOST}:{WHISPER_PORT}/asr`)
- Fallback: local faster-whisper if HTTP unavailable
- Skip if recording < 500ms (too short, same as ChatFlow.ts)
- `user_text_ready` signal → chat widget shows "You: {text}"
- Pipeline continues to RAG → LLM

**Verify:** Press button, speak, release → user text appears → LLM response streams.

---

### Step 7 — TTS worker (parallel)
**Goal:** Audio output plays while text continues streaming.

Files: `core/tts_worker.py`

- Receives sanitized tokens via shared signal (same as display path)
- Sentence buffer with `split_sentences()` logic (port from existing TS utils)
- On sentence boundary → Piper subprocess: `{PIPER_BINARY_PATH} --model {PIPER_MODEL_PATH} --output_file {wav_path}`
- Audio playback via sox: `play {wav_path}` using `SOUND_CARD_INDEX`
- Queue-based: sentences play in order
- Display is NEVER blocked by TTS
- Interrupt: clear queue, kill active Piper/sox subprocesses

**Verify:** LLM response streams on screen AND plays as audio. Text appears before audio (visual leads audio).

---

### Step 8 — Auto-launch on Pi boot
**Goal:** Power on Pi → GUI appears automatically.

Desktop autostart:
```
~/.config/autostart/oasis-gui.desktop
[Desktop Entry]
Type=Application
Name=OASIS GUI
Exec=python3 /home/pi/whisplay-ai-chatbot/python/oasis-gui/main.py
```

Dependency services (systemd, start before GUI):
```
ollama.service → oasis-rag.service → oasis-gui (autostart)
```

Boot sequence: Pi power on (~15s) → desktop loads → oasis-gui starts → fullscreen (~2s) → pre-warm Ollama → Ready.

**Verify:** Reboot Pi → GUI appears within ~20s. No manual intervention.

---

### Step 9 — Polish & resilience
**Goal:** Production-ready kiosk experience.

- **Startup health checks:**
  - Ollama: retry connection every 2s, show "Warming up..." until ready
  - RAG Flask: GET /health, show "Knowledge base loading..." until `index_ready: true`
  - Audio device: test sox recording, show error in footer if mic unavailable
- **Typing indicator:** blinking `█` cursor via QTimer toggle during streaming state
- **Screen dimming:** `xset dpms force off` after 60s idle, GPIO interrupt → `xset dpms force on`
- **Graceful shutdown:** SIGTERM → `GPIO.cleanup()`, stop all QThreads, `QApplication.quit()`
- **Conversation display:** Keep last 5 Q&A pairs visible, older ones auto-removed to bound memory

---

## Dependencies

```bash
# Via apt (Pi OS Bookworm)
sudo apt install python3-pyqt5 sox

# Via pip
pip install httpx python-dotenv

# Already available on Pi
# RPi.GPIO (pre-installed on Pi OS)
# Ollama, oasis-rag Flask :5001, Piper TTS (existing system)
```

---

## Dev/Test Strategy

Steps 1–4 can be developed and tested entirely on **PC** (macOS/Linux):
- No GPIO → keyboard Space simulates button
- No sox/mic → hardcoded query text
- Ollama runs on PC too
- RAG Flask runs on PC too

Steps 5–8 require **Pi** hardware.

```
PC dev:   Step 1 → 2 → 3 → 4
Pi dev:   Step 5 → 6 → 7 → 8 → 9
```
