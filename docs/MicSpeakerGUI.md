a# Production Integration Plan: Mic + Speaker + GUI (v2 — Adjusted)

## Context
Python GUI(`oasis-gui`)에 마이크 녹음, ASR, TTS, 스피커 재생을 통합하여 Node.js 파이프라인을 완전히 대체한다.
핵심 목표: **문장 단위 스트리밍 TTS**로 LLM 응답 첫 문장이 생성되자마자 음성 출력을 시작하여 체감 지연시간을 최소화한다.

**v2 변경사항**: 전체 코드베이스 재스캔 결과 반영 — classify_client 4-way dispatch 보존, QThread 안전성, Recorder를 main thread에서 관리, persistent TTSPlaybackWorker 등.

---

## Architecture Overview

```
[GPIO Button / Spacebar]
        │
        ▼
   OasisApp (main thread)
   ├─ Recorder.start/stop (sox subprocess)
   └─ StateMachine ──signal──▶ GUI (header/footer text)
        │
        ▼
   PipelineWorker (QThread)
   ┌─────────────────────────────────┐
   │ 1. asr_client.recognize (HTTP)  │
   │ 2. classify_client.dispatch     │
   │    ├─ direct/ood → emit text    │
   │    └─ llm/triage:               │
   │       3. llm_client.stream      │
   │          ├─ token → GUI display  │
   │          └─ sentence → TTS queue │
   └─────────────────────────────────┘
        │
        ▼
   TTSPlaybackWorker (QThread) — persistent, queue-based
   ┌─────────────────────────────────┐
   │ sentence queue (thread-safe)    │
   │  → Piper binary → WAV file     │
   │  → sox play → speaker           │
   │ _FLUSH sentinel → emit done     │
   │ _STOP sentinel → exit thread    │
   └─────────────────────────────────┘
```

**Thread Model (4 threads):**
- **Main Thread**: Qt event loop, GUI rendering, GPIO callbacks, Recorder start/stop
- **PipelineWorker**: ASR → Classify dispatch → LLM stream (QThread, started per query)
- **TTSPlaybackWorker**: Persistent queue → Piper synthesis → sox playback (QThread, long-lived)
- **PrewarmThread**: One-shot startup model loading (LLM + classify warmup)

---

## Current Codebase State (key existing code to preserve)

| File | Status | Key Logic to Preserve |
|------|--------|----------------------|
| `core/pipeline_worker.py` | EXISTS (114 lines) | 4-way classify dispatch, triage hint TTL, timing instrumentation |
| `clients/classify_client.py` | EXISTS (97 lines) | DispatchResult dataclass, dispatch(), is_healthy() |
| `clients/llm_client.py` | EXISTS (84 lines) | prewarm(), stream() with abort_flag_ref |
| `clients/rag_client.py` | EXISTS (58 lines) | retrieve_system_prompt() — kept as fallback |
| `core/state_machine.py` | EXISTS (59 lines) | 4-state FSM, missing PROCESSING→LISTENING |
| `utils/sanitizer.py` | EXISTS (14 lines) | sanitize_chunk() |
| `utils/logger.py` | EXISTS (31 lines) | log_response() JSONL logging |
| `main.py` | EXISTS (177 lines) | PrewarmThread, KeyFilter, demo queries (to be replaced) |

---

## Memory Budget: ~2.9 GB / 4.5 GB limit

| Component | RAM |
|-----------|-----|
| Ollama gemma3:1b | ~1.5 GB |
| faster-whisper tiny int8 | ~200 MB |
| Classify (gte-small embeddings) | ~800 MB |
| Piper ONNX medium | ~100 MB |
| PyQt5 + Python | ~300 MB |

---

## Implementation Steps (Dependency Order)

### Step 1: Sentence Splitter — `utils/sentence_splitter.py` — NEW
Port from `src/utils/index.ts` splitSentences + purifyTextForTTS.
No dependencies on other new modules — can be built and tested first.

```python
import re

_SENTENCE_RE = re.compile(r'.*?([。！？!?，,]|\.)(?=\s|$)', re.DOTALL)
_EMOJI_SPECIAL_RE = re.compile(r'[*#~]|[\U0001F300-\U0001FAFF\u200d\ufe0f]')

def split_sentences(text: str) -> tuple[list[str], str]:
    """Split text at sentence boundaries. Returns (complete_sentences, remaining_buffer).

    Ported from src/utils/index.ts splitSentences().
    - Boundaries: . ! ? , followed by whitespace or end of string
    - Merges short sentences (≤60 chars combined) for natural TTS pacing
    """
    sentences = []
    last_index = 0
    for match in _SENTENCE_RE.finditer(text):
        sentence = match.group(0).strip()
        if sentence:
            sentences.append(sentence)
            last_index = match.end()

    remaining = text[last_index:].strip()

    # Merge short sentences (≤60 chars) — matches TS implementation
    merged = []
    buf = ""
    for s in sentences:
        candidate = f"{buf}{s} "
        if len(candidate) <= 60:
            buf = candidate
        else:
            if buf:
                merged.append(buf.rstrip())
            buf = f"{s} "
    if buf:
        merged.append(buf.rstrip())

    return merged, remaining


def purify_for_tts(text: str) -> str:
    """Remove characters unsuitable for TTS (emojis, markdown chars).

    Ported from src/utils/index.ts purifyTextForTTS().
    """
    return _EMOJI_SPECIAL_RE.sub("", text).strip()
```

- Add to `utils/__init__.py` exports

### Step 2: Audio Recorder — `audio/recorder.py` — NEW
Sox subprocess for ALSA mic recording. **Called from main thread** (button events).

```python
import os, signal, time, subprocess, platform

IS_PI = platform.machine().startswith("aarch")
SOUND_CARD_INDEX = os.getenv("SOUND_CARD_INDEX", "1")

class Recorder:
    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._output_path: str = ""
        self._start_time: float = 0.0

    def start(self, output_path: str):
        """Spawn sox recording process. Called from main thread on button press."""
        self.stop()  # kill any stale process
        self._output_path = output_path
        self._start_time = time.time()

        if IS_PI:
            src = ["-t", "alsa", f"hw:{SOUND_CARD_INDEX},0"]
        else:
            src = ["-d"]  # default device on PC

        cmd = ["sox"] + src + ["-t", "wav", "-c", "1", "-r", "16000", output_path]
        self._process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                          stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)

    def stop(self) -> tuple[str, float]:
        """Stop recording. Returns (wav_path, duration_seconds).

        Sends SIGINT for graceful WAV finalization.
        Returns ("", 0) if no recording or too short (<0.5s).
        """
        if self._process is None:
            return "", 0.0

        duration = time.time() - self._start_time
        proc = self._process
        self._process = None

        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

        if duration < 0.5:
            # Too short — likely accidental tap, skip ASR
            return "", duration

        return self._output_path, duration

    @property
    def is_recording(self) -> bool:
        return self._process is not None and self._process.poll() is None
```

- `audio/__init__.py` — empty package init
- Minimum 0.5s check prevents empty audio → wasted ASR call
- SIGINT → sox writes proper WAV header before exiting

### Step 3: ASR Client — `clients/asr_client.py` — NEW
HTTP client for faster-whisper service (port 8803). Matches `faster-whisper-host.py` API.

```python
import os, httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

WHISPER_HOST = os.getenv("FASTER_WHISPER_HOST", "localhost")
WHISPER_PORT = os.getenv("FASTER_WHISPER_PORT", "8803")
WHISPER_LANG = os.getenv("FASTER_WHISPER_LANGUAGE", "en")
WHISPER_URL = f"http://{WHISPER_HOST}:{WHISPER_PORT}"
TIMEOUT = 10.0

_client = httpx.Client(timeout=TIMEOUT)

def recognize(wav_path: str) -> str:
    """POST /recognize — returns transcribed text or "" on any failure."""
    try:
        resp = _client.post(
            f"{WHISPER_URL}/recognize",
            json={"filePath": wav_path, "language": WHISPER_LANG},
        )
        resp.raise_for_status()
        text = resp.json().get("recognition", "").strip()
        cost = resp.json().get("time_cost", 0)
        print(f"[ASR] '{text}' ({cost:.2f}s)")
        return text
    except Exception as e:
        print(f"[ASR] Error: {e}")
        return ""
```

### Step 4: TTS + Playback Worker — `audio/tts_playback.py` — NEW
**Persistent queue-based QThread.** Runs for the entire app lifetime.
Uses sentinel values: `_FLUSH` (query done, emit signal) and `_STOP` (shutdown thread).

```python
import os, signal, subprocess, tempfile, platform
from queue import Queue, Empty
from PyQt5.QtCore import QThread, pyqtSignal

IS_PI = platform.machine().startswith("aarch")
PIPER_BINARY = os.getenv("PIPER_BINARY_PATH", "/home/pi/piper/piper")
PIPER_MODEL = os.getenv("PIPER_MODEL_PATH", "/home/pi/piper/voices/en_US-amy-medium.onnx")
SOUND_CARD_INDEX = os.getenv("SOUND_CARD_INDEX", "1")

_FLUSH = "__FLUSH__"
_STOP  = "__STOP__"

# Check if Piper binary exists (PC mode won't have it)
_TTS_AVAILABLE = os.path.isfile(PIPER_BINARY)

class TTSPlaybackWorker(QThread):
    playback_finished = pyqtSignal()  # emitted after _FLUSH when queue drains

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: Queue = Queue()
        self._abort = False
        self._current_process: subprocess.Popen | None = None

    # ── Public API (called from any thread) ───────────────────────────

    def queue_sentence(self, sentence: str):
        if _TTS_AVAILABLE and not self._abort:
            self._queue.put(sentence)

    def flush(self):
        """Signal that LLM stream is done — play remaining, then emit playback_finished."""
        self._queue.put(_FLUSH)

    def cancel(self):
        """Interrupt — stop current playback and drain queue."""
        self._abort = True
        self._kill_current()
        # Drain queue
        while not self._queue.empty():
            try: self._queue.get_nowait()
            except Empty: break
        # Put flush so thread doesn't block forever
        self._queue.put(_FLUSH)

    def reset(self):
        """Prepare for next query (called before new recording starts)."""
        self._abort = False

    def shutdown(self):
        """Graceful shutdown — called on app exit."""
        self._abort = True
        self._kill_current()
        self._queue.put(_STOP)

    # ── Thread loop ───────────────────────────────────────────────────

    def run(self):
        print(f"[TTS] Worker started (piper available: {_TTS_AVAILABLE})")
        while True:
            item = self._queue.get()

            if item == _STOP:
                break

            if item == _FLUSH:
                if not self._abort:
                    self.playback_finished.emit()
                continue

            if self._abort:
                continue

            # Synthesize + play
            wav_path = self._synthesize(item)
            if wav_path and not self._abort:
                self._play(wav_path)
                try: os.unlink(wav_path)
                except OSError: pass

    def _synthesize(self, text: str) -> str | None:
        """Piper binary: stdin text → WAV file."""
        if not _TTS_AVAILABLE:
            return None
        try:
            fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="oasis_tts_")
            os.close(fd)

            proc = subprocess.Popen(
                [PIPER_BINARY, "--model", PIPER_MODEL,
                 "--sentence-silence", "1", "--output_file", wav_path],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()
            proc.wait(timeout=15)

            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 44:
                return wav_path
            return None
        except Exception as e:
            print(f"[TTS] Synthesis error: {e}")
            return None

    def _play(self, wav_path: str):
        """Play WAV via sox `play` command (blocking)."""
        try:
            self._current_process = subprocess.Popen(
                ["play", wav_path, "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._current_process.wait(timeout=30)
        except Exception as e:
            print(f"[TTS] Playback error: {e}")
        finally:
            self._current_process = None

    def _kill_current(self):
        proc = self._current_process
        if proc and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=1)
            except Exception:
                proc.kill()
```

**Key design decisions:**
- **Persistent thread** — never stops between queries (avoids QThread reuse crash)
- **_FLUSH sentinel** — emits `playback_finished` when queue drains after LLM done
- **_STOP sentinel** — clean exit on app shutdown
- **cancel()** — drains queue + kills current sox process instantly
- **reset()** — clears abort flag for next query
- **PC mode without Piper** — `_TTS_AVAILABLE=False`, `queue_sentence()` becomes no-op, `flush()` still emits signal immediately

### Step 5: State Machine Fix — `core/state_machine.py` — MODIFY
Add missing PROCESSING → LISTENING transition (interrupt during ASR/classify).

```python
# In on_button_press(), add:
elif self._state == State.PROCESSING:
    self.transition(State.LISTENING)   # cancel ASR, start new recording
```

This is a **1-line addition** to the existing method.

### Step 6: PipelineWorker Extension — `core/pipeline_worker.py` — MODIFY
Extend existing 4-way dispatch to include: ASR input, sentence-level TTS dispatch.

**Changes to existing code:**
1. Accept `tts_worker` in constructor
2. Add `asr_text_ready` signal (already exists as `user_text_ready`)
3. New `start_from_wav()` method (replaces `start_query()` for real ASR)
4. Keep `start_query()` for PC demo mode
5. Add `_accumulate_and_dispatch_tts()` to on_token callback
6. Add TTS dispatch for `direct_response`/`ood_response` modes (not just LLM streaming)

**Preserve:**
- classify_client 4-way dispatch logic
- triage_hint TTL tracking
- timing instrumentation (classify latency, LLM TTFT)
- sanitize_chunk on every token
- log_response on completion

```python
# New constructor signature:
def __init__(self, tts_worker=None, parent=None):
    super().__init__(parent)
    self._tts_worker = tts_worker  # None = TTS disabled (backward compat)
    self._sentence_buffer = ""
    # ... rest of existing init ...

# New method for real ASR input:
def start_from_wav(self, wav_path: str):
    """Start pipeline from recorded WAV file (production mode)."""
    self._abort_flag = [False]
    self._wav_path = wav_path
    self._response_buffer = []
    self._sentence_buffer = ""
    self._mode = "wav"
    self.start()

# Modified start_query for demo/fallback:
def start_query(self, user_text: str):
    """Start pipeline with pre-set text (demo mode / PC fallback)."""
    self._abort_flag = [False]
    self._user_text = user_text
    self._response_buffer = []
    self._sentence_buffer = ""
    self._mode = "text"
    self.start()

# Modified run():
def run(self):
    if self._mode == "wav":
        # ASR step
        from clients import asr_client
        self.state_changed.emit("Recognizing...")
        text = asr_client.recognize(self._wav_path)
        if not text.strip():
            self.finished.emit()
            return
        self._user_text = text
        self.user_text_ready.emit(text)

    # ... existing classify dispatch logic (unchanged) ...

    # In the on_token callback, ADD TTS accumulation:
    def on_token(token: str):
        # existing sanitize + emit logic...
        clean = sanitize_chunk(token)
        if clean:
            self._response_buffer.append(clean)
            self.token_received.emit(clean)
            self._accumulate_tts(clean)  # NEW

    def on_done():
        self._flush_tts()  # NEW: flush remaining buffer
        if self._response_buffer:
            log_response(self._user_text, "".join(self._response_buffer))
        self.finished.emit()

    # For direct_response/ood_response — also dispatch to TTS:
    if result.mode in ("direct_response", "ood_response"):
        text = result.response_text or classify_client.SAFE_FALLBACK_TEXT
        clean = sanitize_chunk(text)
        if clean:
            self._response_buffer.append(clean)
            self.token_received.emit(clean)
            self._dispatch_tts_sentence(clean)  # NEW: speak direct response
        self._flush_tts()  # NEW
        log_response(self._user_text, text)
        self.finished.emit()
        return

def _accumulate_tts(self, token: str):
    if not self._tts_worker:
        return
    from utils.sentence_splitter import split_sentences, purify_for_tts
    self._sentence_buffer += token
    sentences, remaining = split_sentences(self._sentence_buffer)
    for s in sentences:
        clean = purify_for_tts(s)
        if clean.strip():
            self._tts_worker.queue_sentence(clean)
    self._sentence_buffer = remaining

def _flush_tts(self):
    if not self._tts_worker:
        return
    from utils.sentence_splitter import purify_for_tts
    if self._sentence_buffer.strip():
        clean = purify_for_tts(self._sentence_buffer)
        if clean.strip():
            self._tts_worker.queue_sentence(clean)
    self._sentence_buffer = ""
    self._tts_worker.flush()

def _dispatch_tts_sentence(self, text: str):
    if not self._tts_worker:
        return
    from utils.sentence_splitter import purify_for_tts
    clean = purify_for_tts(text)
    if clean.strip():
        self._tts_worker.queue_sentence(clean)
```

### Step 7: GPIO Handler — `core/gpio_handler.py` — NEW
RPi.GPIO interrupt-based button handler. Pi-only.

```python
from PyQt5.QtCore import QObject, pyqtSignal

class GPIOHandler(QObject):
    button_pressed = pyqtSignal()
    button_released = pyqtSignal()

    BUTTON_PIN = 11  # Whisplay HAT physical pin

    def start(self):
        import RPi.GPIO as GPIO
        self._GPIO = GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.BUTTON_PIN, GPIO.BOTH,
                              callback=self._on_edge, bouncetime=50)

    def _on_edge(self, channel):
        if self._GPIO.input(channel) == self._GPIO.LOW:
            self.button_pressed.emit()
        else:
            self.button_released.emit()

    def cleanup(self):
        self._GPIO.cleanup()
```

### Step 8: main.py Integration — `main.py` — MODIFY
Wire everything together. Remove demo query simulation, add real recording + ASR.

**Key changes:**
1. Create Recorder (main thread) + TTSPlaybackWorker (persistent thread)
2. Pass tts_worker to PipelineWorker
3. GPIO on Pi, spacebar on PC
4. Recording start/stop on state transitions
5. ASR result → chat display → begin streaming
6. TTS playback_finished → IDLE transition
7. Keep PC demo fallback (if no recorder/ASR available)

```python
class OasisApp:
    def __init__(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.window = MainWindow(screen_size=SCREEN_SIZE)
        self.sm = StateMachine()

        # TTS worker — persistent thread, started once
        self.tts_worker = TTSPlaybackWorker()
        self.tts_worker.start()

        # Recorder — managed from main thread
        self.recorder = Recorder()

        # Pipeline worker — uses classify dispatch + TTS
        self.worker = PipelineWorker(tts_worker=self.tts_worker)

        # GPIO (Pi) or keyboard (PC)
        if IS_PI:
            self.gpio = GPIOHandler()
            self.gpio.button_pressed.connect(self._on_button_press)
            self.gpio.button_released.connect(self._on_button_release)
            self.gpio.start()

        # App-level key filter (PC simulation — keep existing)
        self._key_filter = KeyFilter()
        self.app.installEventFilter(self._key_filter)
        self._key_filter.key_pressed.connect(self._on_key_press)
        self._key_filter.key_released.connect(self._on_key_release)
        self._space_held = False

        # Wire signals
        self.sm.state_changed.connect(self._on_state_changed)
        self.worker.token_received.connect(self.window.chat.append_token)
        self.worker.user_text_ready.connect(self._on_asr_done)
        self.worker.finished.connect(self._on_stream_done)
        self.worker.error_occurred.connect(self._on_error)
        self.tts_worker.playback_finished.connect(self._on_playback_done)

        # Prewarm
        self._prewarm_thread = PrewarmThread()
        self._prewarm_thread.done.connect(self._on_prewarm_done)
        self.window.set_status("Warming up...")
        self.window.set_footer("Loading model, please wait...")

        if IS_PI:
            self.window.showFullScreen()
        else:
            self.window.resize(800, 480)
            self.window.show()

        self._prewarm_thread.start()

    def _on_state_changed(self, state):
        self._apply_state(state)

        if state == State.LISTENING:
            # Interrupt any running pipeline + TTS
            self.worker.abort()
            self.tts_worker.cancel()
            self.window.chat.end_oasis_response()
            # Start recording
            self.tts_worker.reset()
            wav_path = os.path.join(_DATA_DIR, "asr", f"recording_{int(time.time()*1000)}.wav")
            self.recorder.start(wav_path)

        elif state == State.PROCESSING:
            wav_path, duration = self.recorder.stop()
            if not wav_path:
                # Too short or failed — return to idle
                self.sm.transition(State.IDLE)
                return
            self.window.chat.begin_oasis_response()
            self.worker.start_from_wav(wav_path)

    def _on_asr_done(self, text):
        """ASR text recognized — show in chat, begin streaming response."""
        self.window.chat.add_message(ROLE_USER, text)
        self.window.chat.begin_oasis_response()
        self.sm.on_pipeline_started()  # PROCESSING → STREAMING

    def _on_stream_done(self):
        """LLM stream complete — wait for TTS to finish playing."""
        self.window.chat.end_oasis_response()
        # If TTS not available, go directly to IDLE
        if not _TTS_AVAILABLE:
            self.sm.on_pipeline_done()

    def _on_playback_done(self):
        """All TTS audio played — now transition to IDLE."""
        self.sm.on_pipeline_done()
```

### Step 9: Startup Script — `run_oasis_gui.sh` — NEW

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Load .env
set -a; [ -f .env ] && source .env; set +a

# Sound card detection (Pi)
card_index=$(awk '/wm8960soundcard/ {print $1}' /proc/asound/cards 2>/dev/null | head -n1)
card_index=${card_index:-${SOUND_CARD_INDEX:-1}}
amixer -c "$card_index" set Speaker "${INITIAL_VOLUME_LEVEL:-114}" 2>/dev/null || true
export SOUND_CARD_INDEX="$card_index"

# Ollama
if [ "${SERVE_OLLAMA:-false}" = "true" ]; then
    OLLAMA_HOST=0.0.0.0:11434 ollama serve &
    OLLAMA_PID=$!
    sleep 2
fi

# Classify service
python3 python/oasis-classify/service.py &
CLASSIFY_PID=$!

# Faster-whisper ASR
python3 python/speech-service/faster-whisper-host.py --port "${FASTER_WHISPER_PORT:-8803}" &
ASR_PID=$!

# Wait for classify service health
for i in $(seq 1 10); do
    curl -sf http://127.0.0.1:5002/health >/dev/null && break
    sleep 3
done

# GUI (main entry point)
python3 python/oasis-gui/main.py
EXIT_CODE=$?

# Cleanup
kill $CLASSIFY_PID $ASR_PID ${OLLAMA_PID:-} 2>/dev/null || true
exit $EXIT_CODE
```

---

## File Summary

### New Files (6)
| File | Purpose |
|------|---------|
| `audio/__init__.py` | Package init |
| `audio/recorder.py` | Sox ALSA recording (main thread) |
| `audio/tts_playback.py` | Persistent Piper TTS + sox playback QThread |
| `clients/asr_client.py` | faster-whisper HTTP client |
| `core/gpio_handler.py` | RPi.GPIO button handler (Pi only) |
| `utils/sentence_splitter.py` | Sentence splitting + TTS text cleanup |

### Modified Files (3)
| File | Changes |
|------|---------|
| `core/state_machine.py` | Add PROCESSING→LISTENING transition (1 line) |
| `core/pipeline_worker.py` | Add ASR step, sentence-level TTS dispatch, preserve classify logic |
| `main.py` | Wire Recorder, TTSWorker, GPIO, remove demo simulation |

### Unchanged (keep as-is)
| File | Reason |
|------|--------|
| `clients/classify_client.py` | Already complete — 4-way dispatch |
| `clients/llm_client.py` | Already complete — streaming + abort |
| `clients/rag_client.py` | Kept as fallback |
| `utils/sanitizer.py` | Already complete |
| `utils/logger.py` | Already complete |
| `gui/*` | No changes needed |

---

## Critical Corrections from v1

| # | Issue | Fix |
|---|-------|-----|
| 1 | Recorder was inside PipelineWorker | Recorder in OasisApp (main thread) — button press/release are main-thread events |
| 2 | QThread reuse crash | PipelineWorker uses `start()` per query (existing pattern). TTSPlaybackWorker is persistent (never restarts) |
| 3 | TTSPlaybackWorker start/stop per query | Persistent queue with _FLUSH/_STOP sentinels |
| 4 | Missing PROCESSING→LISTENING | Added to on_button_press() — can cancel during ASR |
| 5 | PC without Piper | `_TTS_AVAILABLE = os.path.isfile(PIPER_BINARY)` — graceful no-op |
| 6 | No short sentence merging | ≤60 char merge in split_sentences() — matches TS implementation |
| 7 | No minimum recording check | 0.5s minimum in Recorder.stop() |
| 8 | Plan didn't mention classify_client | Preserved existing 4-way dispatch + triage hints |
| 9 | Plan showed rag_client in pipeline | Actually uses classify_client.dispatch() |

---

## Latency Timeline (Target on Pi5)

```
Button Release (t=0)
  ├─ ASR:           t+0 → t+1.5s   (faster-whisper tiny)
  ├─ Classify:      t+1.5 → t+1.6s (100ms — may short-circuit to direct_response)
  ├─ LLM first tok: t+1.6 → t+2.4s (~800ms TTFT, only for llm_prompt/triage_prompt)
  ├─ 1st sentence:  t+2.4 → t+3.9s (~1.5s of tokens)
  ├─ Piper TTS:     t+3.9 → t+4.4s (~500ms synthesis)
  └─ FIRST AUDIO:   t+4.4s ✓
      (LLM continues generating + TTS queues next sentences in parallel)

For direct_response (tier0 match):
  ├─ ASR:           t+0 → t+1.5s
  ├─ Classify:      t+1.5 → t+1.6s → direct_response (no LLM!)
  ├─ Piper TTS:     t+1.6 → t+2.1s
  └─ FIRST AUDIO:   t+2.1s ✓ (extremely fast for common queries)
```

---

## Verification Plan

1. **PC smoke test**: Spacebar → sox -d → ASR → classify → LLM → text display (TTS skipped if no Piper)
2. **Pi full test**: GPIO button → ALSA recording → ASR → classify → LLM → Piper TTS → speaker
3. **Interrupt test**: During streaming, press button → TTS stops, new recording begins
4. **PROCESSING interrupt**: During ASR, press button → cancel ASR, start new recording
5. **Direct response test**: Say "help" → tier0 match → immediate TTS without LLM
6. **Memory check**: `htop` RSS ≤ 4.5 GB on Pi5
7. **Latency log**: Button release → first audio timestamp (target ≤4.5s for LLM path, ≤2.5s for direct)
