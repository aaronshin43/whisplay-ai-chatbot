# oasis-classify Integration Plan for oasis-gui

Replace the `rag_client` → LLM pipeline with the `classify_client` → conditional LLM pipeline.
The GUI output path (token_received signal → ChatWidget → TTS) stays identical for all modes.

---

## What changes

| Component | Current | After |
|-----------|---------|-------|
| `clients/rag_client.py` | POST :5001/retrieve → system_prompt | Keep as-is (fallback option) |
| `clients/classify_client.py` | *(does not exist)* | **New file** — POST :5002/dispatch → DispatchResult |
| `core/pipeline_worker.py` | RAG → LLM always | **Modified** — dispatch → branch on mode |
| Triage hint state | None | `PipelineWorker._triage_hint` — persists across queries |

---

## New file: `clients/classify_client.py`

Mirror the resolve-never-reject contract of `rag_client.py`.

```python
import os, httpx
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

CLASSIFY_URL = os.getenv("OASIS_CLASSIFY_SERVICE_URL", "http://localhost:5002")
TIMEOUT = float(os.getenv("OASIS_CLASSIFY_TIMEOUT_MS", "3000")) / 1000.0

SAFE_FALLBACK_TEXT = (
    "I cannot reach the emergency knowledge system. "
    "Please call 911 immediately for any life-threatening emergency."
)

@dataclass
class DispatchResult:
    mode: Literal["direct_response", "llm_prompt", "triage_prompt", "ood_response"]
    response_text: str | None
    system_prompt: str | None
    category: str | None          # always set for triage_prompt
    score: float | None
    threshold_path: str
    latency_ms: float
    hint_changed_result: bool

def _safe_fallback(threshold_path: str) -> DispatchResult:
    return DispatchResult(
        mode="ood_response",
        response_text=SAFE_FALLBACK_TEXT,
        system_prompt=None,
        category=None,
        score=None,
        threshold_path=threshold_path,
        latency_ms=0.0,
        hint_changed_result=False,
    )

def dispatch(query: str, prev_triage_hint: str | None) -> DispatchResult:
    """POST /dispatch — resolves, never raises.

    On any failure returns a safe ood_response fallback with a
    diagnostic threshold_path so logs distinguish infra failures from real OOD.
    """
    try:
        resp = httpx.post(
            f"{CLASSIFY_URL}/dispatch",
            json={"query": query, "prev_triage_hint": prev_triage_hint},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        d = resp.json()

        # Validate required fields — return invalid_schema fallback if malformed
        required = ("mode", "threshold_path", "latency_ms", "hint_changed_result")
        if not all(k in d for k in required):
            print(f"[Classify] Response missing required fields — invalid_schema")
            return _safe_fallback("invalid_schema")

        result = DispatchResult(
            mode=d["mode"],
            response_text=d.get("response_text"),
            system_prompt=d.get("system_prompt"),
            category=d.get("category"),
            score=d.get("score"),
            threshold_path=d["threshold_path"],
            latency_ms=d["latency_ms"],
            hint_changed_result=d["hint_changed_result"],
        )
        print(
            f"[Classify] mode={result.mode}  cat={result.category}  "
            f"score={f'{result.score:.3f}' if result.score is not None else '—'}  "
            f"path={result.threshold_path}  latency={result.latency_ms:.1f}ms  "
            f"hint_changed={result.hint_changed_result}"
        )
        return result

    except httpx.ConnectError:
        print(f"[Classify] Service not reachable ({CLASSIFY_URL}) — network_error")
        return _safe_fallback("network_error")
    except httpx.TimeoutException:
        print(f"[Classify] Timeout after {TIMEOUT}s — network_error")
        return _safe_fallback("network_error")
    except Exception as e:
        print(f"[Classify] Unexpected error: {e} — service_error")
        return _safe_fallback("service_error")

def is_healthy() -> bool:
    try:
        resp = httpx.get(f"{CLASSIFY_URL}/health", timeout=3.0)
        return resp.json().get("status") == "ok"
    except Exception:
        return False
```

---

## Modified file: `core/pipeline_worker.py`

### Changes from current version

1. Import `classify_client` instead of `rag_client`
2. Add `_triage_hint` instance variable (persists across `start_query()` calls)
3. Add `_clear_triage_hint()` helper with TTL check
4. Replace the linear RAG → LLM block with a 4-mode dispatch branch

```python
import time
from PyQt5.QtCore import QThread, pyqtSignal
from clients import llm_client, classify_client
from utils.sanitizer import sanitize_chunk
from utils.logger import log_response

TRIAGE_HINT_TTL_SEC = 60  # matches Python config.py TRIAGE_HINT_TTL_SEC

class PipelineWorker(QThread):
    token_received  = pyqtSignal(str)   # sanitized token → ChatWidget + TTS
    user_text_ready = pyqtSignal(str)   # ASR result → show in chat
    state_changed   = pyqtSignal(str)   # status label updates
    error_occurred  = pyqtSignal(str)
    finished        = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._user_text = ""
        self._abort_flag = [False]
        self._response_buffer = []
        # Triage hint — survives across queries, cleared by TTL or llm_prompt result
        self._triage_hint: str | None = None
        self._triage_hint_expires: float = 0.0

    def start_query(self, user_text: str):
        """Prepare and start a new query. Triage hint is NOT reset here."""
        self._abort_flag = [False]
        self._user_text = user_text
        self._response_buffer = []
        self.start()

    def abort(self):
        self._abort_flag[0] = True

    def _get_active_hint(self) -> str | None:
        if self._triage_hint and time.time() < self._triage_hint_expires:
            return self._triage_hint
        self._triage_hint = None
        return None

    def run(self):
        # ── Step 1: Dispatch ────────────────────────────────────────────────
        self.state_changed.emit("Classifying...")
        result = classify_client.dispatch(
            query=self._user_text,
            prev_triage_hint=self._get_active_hint(),
        )

        if self._abort_flag[0]:
            self.finished.emit()
            return

        # ── Step 2: Branch on mode ──────────────────────────────────────────
        if result.mode in ("direct_response", "ood_response"):
            # Pre-baked response — no LLM call.
            self._triage_hint = None
            text = result.response_text or classify_client.SAFE_FALLBACK_TEXT
            # Emit as a single token so ChatWidget and TTS handle it identically
            # to streamed responses. No structural changes needed downstream.
            clean = sanitize_chunk(text)
            if clean:
                self._response_buffer.append(clean)
                self.token_received.emit(clean)
            log_response(self._user_text, text)
            self.finished.emit()
            return

        # llm_prompt or triage_prompt — call LLM
        if result.mode == "llm_prompt":
            self._triage_hint = None
        else:  # triage_prompt
            # category is always set by Python for triage_prompt
            self._triage_hint = result.category
            self._triage_hint_expires = time.time() + TRIAGE_HINT_TTL_SEC
            if result.hint_changed_result:
                print(f"[Classify] Triage hint changed top-1 result for category={result.category}")

        system_prompt = result.system_prompt or classify_client.SAFE_FALLBACK_TEXT

        # ── Step 3: Build messages (fresh per query) ─────────────────────────
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": self._user_text},
        ]

        # ── Step 4: Stream LLM tokens ────────────────────────────────────────
        self.state_changed.emit("Generating response...")

        def on_token(token: str):
            clean = sanitize_chunk(token)
            if clean:
                self._response_buffer.append(clean)
                self.token_received.emit(clean)

        def on_done():
            if self._response_buffer:
                log_response(self._user_text, "".join(self._response_buffer))
            self.finished.emit()

        llm_client.stream(
            messages=messages,
            on_token=on_token,
            on_done=on_done,
            abort_flag_ref=self._abort_flag,
        )
```

---

## Environment variable to add to `.env`

```
OASIS_CLASSIFY_SERVICE_URL=http://localhost:5002
OASIS_CLASSIFY_TIMEOUT_MS=3000
```

---

## Health check update (`core/pipeline_worker.py` startup or `main.py`)

The existing startup health check probes RAG (`:5001`). Add classify (`:5002`):

```python
# In main.py or wherever is_healthy() is called at boot
from clients import classify_client

if not classify_client.is_healthy():
    footer.set_text("Knowledge classifier unavailable — limited mode")
```

Both services should be running before the GUI reaches "Ready" state.

---

## State machine / status label changes

| Dispatch mode | `state_changed` label during processing |
|---|---|
| Tier 0 hit (any mode) | `"Classifying..."` then instant finish |
| `llm_prompt` | `"Classifying..."` → `"Generating response..."` |
| `triage_prompt` | `"Classifying..."` → `"Generating response..."` |
| `direct_response` / `ood_response` | `"Classifying..."` then instant finish |

The existing "Retrieving knowledge..." label in `pipeline_worker.py` should be replaced with "Classifying...".

---

## What does NOT change

- `ChatWidget.append_token()` — unchanged. Works for both single-emit (direct/ood) and streaming (llm/triage).
- `TTS worker` — receives `token_received` signal identically. No changes needed.
- `llm_client.py` — unchanged.
- `utils/sanitizer.py`, `utils/logger.py` — unchanged.
- GPIO, audio, ASR pipeline — unchanged.
- State machine transitions — unchanged.

---

## Implementation order

1. Create `clients/classify_client.py`
2. Modify `core/pipeline_worker.py` (replace rag_client with classify_client, add triage hint)
3. Add env vars to `.env`
4. Update startup health check in `main.py`
5. Replace "Retrieving knowledge..." status label string

Files that do NOT need to be touched: `llm_client.py`, `rag_client.py`, all GUI widgets, TTS worker, GPIO handler, sanitizer, logger.
