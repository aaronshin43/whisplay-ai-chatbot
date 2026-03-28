from PyQt5.QtCore import QThread, pyqtSignal
from clients import llm_client, rag_client
from utils.sanitizer import sanitize_chunk
from utils.logger import log_response


class PipelineWorker(QThread):
    token_received = pyqtSignal(str)   # sanitized token → ChatWidget.append_token()
    user_text_ready = pyqtSignal(str)  # ASR result → show in chat
    state_changed = pyqtSignal(str)    # informational status updates
    error_occurred = pyqtSignal(str)   # error message
    finished = pyqtSignal()            # stream complete (success or abort)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._user_text = ""
        self._abort_flag = [False]     # mutable single-element list for thread-safe flag
        self._response_buffer = []     # accumulates full response for logging

    def start_query(self, user_text: str):
        """Prepare and start a new query. Resets history (OASIS mode: no carry-over)."""
        self._abort_flag = [False]
        self._user_text = user_text
        self._response_buffer = []
        self.start()

    def abort(self):
        """Signal the running stream to stop at the next token boundary."""
        self._abort_flag[0] = True

    def run(self):
        """Runs in worker thread: RAG → LLM stream → log."""
        # Step 1: RAG retrieval (blocking, ~200ms on Pi)
        self.state_changed.emit("Retrieving knowledge...")
        system_prompt = rag_client.retrieve_system_prompt(self._user_text)

        if self._abort_flag[0]:
            self.finished.emit()
            return

        # Step 2: Build messages (fresh per query — OASIS mode, no history carry-over)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._user_text},
        ]

        # Step 3: Stream LLM tokens
        def on_token(token: str):
            clean = sanitize_chunk(token)
            if clean:
                self._response_buffer.append(clean)
                self.token_received.emit(clean)

        def on_done():
            # Log completed response (matches OasisAdapter.ts format)
            if self._response_buffer:
                full_response = "".join(self._response_buffer)
                log_response(self._user_text, full_response)
            self.finished.emit()

        llm_client.stream(
            messages=messages,
            on_token=on_token,
            on_done=on_done,
            abort_flag_ref=self._abort_flag,
        )
