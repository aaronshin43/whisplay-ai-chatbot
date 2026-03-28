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
        t_start = time.time()

        # ── Step 1: Dispatch ────────────────────────────────────────────────
        self.state_changed.emit("Classifying...")
        result = classify_client.dispatch(
            query=self._user_text,
            prev_triage_hint=self._get_active_hint(),
        )
        t_classify = time.time()
        print(f"[Pipeline] classify round-trip: {(t_classify - t_start)*1000:.1f}ms  (service-side: {result.latency_ms:.1f}ms)")

        if self._abort_flag[0]:
            self.finished.emit()
            return

        # ── Step 2: Branch on mode ──────────────────────────────────────────
        if result.mode in ("direct_response", "ood_response"):
            # Pre-baked response — no LLM call.
            self._triage_hint = None
            text = result.response_text or classify_client.SAFE_FALLBACK_TEXT
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

        t_llm_start = time.time()
        _ttft_printed = [False]

        def on_token(token: str):
            if not _ttft_printed[0]:
                _ttft_printed[0] = True
                print(f"[Pipeline] LLM TTFT: {(time.time() - t_llm_start)*1000:.1f}ms  (total from query: {(time.time() - t_start)*1000:.1f}ms)")
            clean = sanitize_chunk(token)
            if clean:
                self._response_buffer.append(clean)
                self.token_received.emit(clean)

        def on_done():
            if self._response_buffer:
                log_response(self._user_text, "".join(self._response_buffer))
            self.finished.emit()

        print(f"[Pipeline] calling LLM (system_prompt tokens ~{len(system_prompt.split())} words)")
        llm_client.stream(
            messages=messages,
            on_token=on_token,
            on_done=on_done,
            abort_flag_ref=self._abort_flag,
        )
