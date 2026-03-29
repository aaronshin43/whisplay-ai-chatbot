import sys
import os
import time
import platform
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QThread, QObject, QEvent, pyqtSignal
import gui.theme as theme

IS_PI = platform.machine().startswith("aarch")

# Pi display is always 800×480 (4.3" Whisplay HAT).
# PC: simulate the same dimensions.
SCREEN_SIZE = (800, 480)
if IS_PI:
    # Detect actual size in case a different display is attached.
    _app_tmp = QApplication.instance() or QApplication(sys.argv)
    _s = _app_tmp.primaryScreen().size()
    _detected = (_s.width(), _s.height())
    print(f"[oasis] detected screen: {_detected}, using: {SCREEN_SIZE}")

# Set portrait width BEFORE any widget imports use get_font_size()
theme.EFFECTIVE_PORTRAIT_WIDTH = min(SCREEN_SIZE)

from gui.main_window import MainWindow
from gui.chat_widget import ROLE_USER
from core.state_machine import StateMachine, State, STATE_UI
from core.pipeline_worker import PipelineWorker
from audio.recorder import Recorder
from audio.tts_playback import TTSPlaybackWorker, _TTS_AVAILABLE
from clients import llm_client, classify_client

_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")


class PrewarmThread(QThread):
    done = pyqtSignal()

    def run(self):
        # Prewarm LLM (loads model into Ollama memory)
        llm_client.prewarm()
        # Prewarm classify service (loads gte-small embedding model on first call)
        classify_client.dispatch("help", None)
        self.done.emit()


class KeyFilter(QObject):
    """App-level event filter: catches key events regardless of which widget has focus."""
    key_pressed  = pyqtSignal(int)
    key_released = pyqtSignal(int)

    def eventFilter(self, _obj, event):
        if event.type() == QEvent.KeyPress and not event.isAutoRepeat():
            self.key_pressed.emit(event.key())
        elif event.type() == QEvent.KeyRelease and not event.isAutoRepeat():
            self.key_released.emit(event.key())
        return False  # never consume — let widgets handle normally too


class OasisApp:
    def __init__(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.window = MainWindow(screen_size=SCREEN_SIZE)
        self.sm = StateMachine()

        # TTS worker — persistent thread, started once at app launch
        self.tts_worker = TTSPlaybackWorker()
        self.tts_worker.start()

        # Recorder — managed from main thread (button events)
        self.recorder = Recorder()

        # Pipeline worker — receives tts_worker for sentence dispatch
        self.worker = PipelineWorker(tts_worker=self.tts_worker)

        # GPIO on Pi, keyboard on PC
        if IS_PI:
            from core.gpio_handler import GPIOHandler
            self.gpio = GPIOHandler()
            self.gpio.button_pressed.connect(self._on_button_press)
            self.gpio.button_released.connect(self._on_button_release)
            self.gpio.start()

        # App-level key filter (PC simulation)
        self._key_filter = KeyFilter()
        self.app.installEventFilter(self._key_filter)
        self._key_filter.key_pressed.connect(self._on_key_press)
        self._key_filter.key_released.connect(self._on_key_release)
        self._space_held = False

        # Wire state machine → UI
        self.sm.state_changed.connect(self._on_state_changed)

        # Wire worker signals → UI + state machine
        self.worker.token_received.connect(self.window.chat.append_token)
        self.worker.user_text_ready.connect(self._on_asr_done)
        self.worker.finished.connect(self._on_stream_done)
        self.worker.error_occurred.connect(self._on_error)

        # TTS playback finished → transition to IDLE
        self.tts_worker.playback_finished.connect(self._on_playback_done)

        # Pre-warm in background thread — GUI stays responsive
        self._prewarm_thread = PrewarmThread()
        self._prewarm_thread.done.connect(self._on_prewarm_done)

        # Initial UI
        self.window.set_status("Warming up...")
        self.window.set_footer("Loading model, please wait...")

        # Pi: fullscreen (800×480, content rotated inside).
        # PC: 800×480 window simulating the Pi screen.
        if IS_PI:
            self.window.showFullScreen()
        else:
            self.window.resize(800, 480)
            self.window.show()

        self._prewarm_thread.start()
        self.app.aboutToQuit.connect(self._shutdown)

    # ── Pre-warm ─────────────────────────────────────────────────────────────

    def _on_prewarm_done(self):
        self._apply_state(State.IDLE)

    # ── State machine ─────────────────────────────────────────────────────────

    def _apply_state(self, state: State):
        header_text, footer_text = STATE_UI[state]
        self.window.set_status(header_text)
        self.window.set_footer(footer_text)

    def _on_state_changed(self, state: State):
        self._apply_state(state)

        if state == State.LISTENING:
            # Interrupt any running pipeline + TTS
            self.worker.abort()
            self.tts_worker.cancel()
            self.window.chat.end_oasis_response()
            # Prepare for new query
            self.tts_worker.reset()
            asr_dir = os.path.join(_DATA_DIR, "asr")
            os.makedirs(asr_dir, exist_ok=True)
            wav_path = os.path.join(asr_dir, f"recording_{int(time.time()*1000)}.wav")
            self.recorder.start(wav_path)

        elif state == State.PROCESSING:
            wav_path, duration = self.recorder.stop()
            if not wav_path:
                # Too short or no audio — return to idle
                self.sm.transition(State.IDLE)
                return
            self.worker.start_from_wav(wav_path)

    def _on_asr_done(self, text: str):
        """ASR text recognized — show in chat, update state to STREAMING."""
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
        """All TTS audio played — transition to IDLE."""
        self.sm.on_pipeline_done()

    def _on_error(self, msg: str):
        self.window.set_status(f"Error: {msg}")
        self.sm.transition(State.IDLE)

    # ── Button handlers (GPIO) ────────────────────────────────────────────────

    def _on_button_press(self):
        self.sm.on_button_press()

    def _on_button_release(self):
        self.sm.on_button_release()

    # ── Keyboard simulation (Space = button, Escape = quit) ───────────────────

    def _on_key_press(self, key: int):
        if key == Qt.Key_Space and not self._space_held:
            self._space_held = True
            self.sm.on_button_press()
        elif key == Qt.Key_Escape:
            self.app.quit()  # aboutToQuit handles _shutdown()

    def _on_key_release(self, key: int):
        if key == Qt.Key_Space:
            self._space_held = False
            self.sm.on_button_release()

    def _shutdown(self):
        self.tts_worker.shutdown()
        if IS_PI and hasattr(self, "gpio"):
            self.gpio.cleanup()

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    OasisApp().run()
