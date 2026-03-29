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

    def __init__(self, tts_worker=None, parent=None):
        super().__init__(parent)
        self._tts_worker = tts_worker  # None = TTS disabled (backward compat)
        self._user_text = ""
        self._wav_path = ""
        self._mode = "text"  # "text" | "wav"
        self._abort_flag = [False]
        self._response_buffer = []
        self._sentence_buffer = ""
        # Triage hint — survives across queries, cleared by TTL or llm_prompt result
        self._triage_hint: str | None = None
        self._triage_hint_expires: float = 0.0

    def start_from_wav(self, wav_path: str):
        """Start pipeline from recorded WAV file (production mode)."""
        self._abort_flag = [False]
        self._wav_path = wav_path
        self._response_buffer = []
        self._sentence_buffer = ""
        self._mode = "wav"
        self.start()

    def start_query(self, user_text: str):
        """Start pipeline with pre-set text (kept for testing/fallback)."""
        self._abort_flag = [False]
        self._user_text = user_text
        self._response_buffer = []
        self._sentence_buffer = ""
        self._mode = "text"
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

        # ── Step 0: ASR (wav mode only) ──────────────────────────────────────
        if self._mode == "wav":
            from clients import asr_client
            self.state_changed.emit("Recognizing...")
            text = asr_client.recognize(self._wav_path)
            if not text.strip():
                self.finished.emit()
                return
            self._user_text = text
            if not self._abort_flag[0]:
                self.user_text_ready.emit(text)

        if self._abort_flag[0]:
            self.finished.emit()
            return

        # ── Step 1: Dispatch ─────────────────────────────────────────────────
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

        # ── Step 2: Branch on mode ───────────────────────────────────────────
        if result.mode in ("direct_response", "ood_response"):
            # Pre-baked response — no LLM call.
            self._triage_hint = None
            text = result.response_text or classify_client.SAFE_FALLBACK_TEXT
            clean = sanitize_chunk(text)
            if clean:
                self._response_buffer.append(clean)
                self.token_received.emit(clean)
                self._dispatch_tts_sentence(clean)
            self._flush_tts()
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
                self._accumulate_tts(clean)

        def on_done():
            self._flush_tts()
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

    # ── TTS helpers ───────────────────────────────────────────────────────────

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
