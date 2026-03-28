import sys
import platform
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer, QThread, QObject, QEvent, pyqtSignal
import gui.theme as theme

IS_PI = platform.machine().startswith("aarch")

# PC: simulate Pi 800×480 screen. Pi: use actual screen size.
if IS_PI:
    _app_tmp = QApplication.instance() or QApplication(sys.argv)
    _s = _app_tmp.primaryScreen().size()
    SCREEN_SIZE = (_s.width(), _s.height())
else:
    SCREEN_SIZE = (800, 480)

# Set portrait width BEFORE any widget imports use get_font_size()
theme.EFFECTIVE_PORTRAIT_WIDTH = min(SCREEN_SIZE)

from gui.main_window import MainWindow
from gui.chat_widget import ROLE_USER
from core.state_machine import StateMachine, State, STATE_UI
from core.pipeline_worker import PipelineWorker
from clients import llm_client


DEMO_QUERY = "How do I treat a burn?"


class PrewarmThread(QThread):
    done = pyqtSignal()

    def run(self):
        llm_client.prewarm()
        self.done.emit()


class KeyFilter(QObject):
    """App-level event filter: catches key events regardless of which widget has focus."""
    key_pressed = pyqtSignal(int)
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
        self.window = MainWindow(screen_size=None if IS_PI else SCREEN_SIZE)
        self.sm = StateMachine()
        self.worker = PipelineWorker()
        self._prewarm_thread = PrewarmThread()

        # App-level key filter — works regardless of which widget has focus
        self._key_filter = KeyFilter()
        self.app.installEventFilter(self._key_filter)
        self._key_filter.key_pressed.connect(self._on_key_press)
        self._key_filter.key_released.connect(self._on_key_release)
        self._space_held = False

        # Wire state machine → UI
        self.sm.state_changed.connect(self._on_state_changed)

        # Wire worker signals → UI + state machine
        self.worker.token_received.connect(self.window.chat.append_token)
        self.worker.finished.connect(self._on_stream_done)
        self.worker.error_occurred.connect(self._on_error)

        # Pre-warm in background thread — GUI stays responsive
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

    # ── Pre-warm ────────────────────────────────────────────────────────────

    def _on_prewarm_done(self):
        self._apply_state(State.IDLE)

    # ── State machine ───────────────────────────────────────────────────────

    def _apply_state(self, state: State):
        header_text, footer_text = STATE_UI[state]
        self.window.set_status(header_text)
        self.window.set_footer(footer_text)

    def _on_state_changed(self, state: State):
        self._apply_state(state)

        if state == State.PROCESSING:
            # Simulate ASR delay (500ms) then start real LLM stream
            QTimer.singleShot(500, self._simulate_asr_done)

        elif state == State.LISTENING:
            # Interrupt: abort any running stream
            self.worker.abort()
            self.window.chat.end_oasis_response()

    def _simulate_asr_done(self):
        if self.sm.state != State.PROCESSING:
            return
        self.window.chat.add_message(ROLE_USER, DEMO_QUERY)
        self.sm.on_pipeline_started()
        self.window.chat.begin_oasis_response()
        self.worker.start_query(user_text=DEMO_QUERY)

    def _on_stream_done(self):
        self.window.chat.end_oasis_response()
        self.sm.on_pipeline_done()

    def _on_error(self, msg: str):
        self.window.set_status(f"Error: {msg}")
        self.sm.transition(State.IDLE)

    # ── Keyboard simulation (Space = button press/release) ──────────────────

    def _on_key_press(self, key: int):
        if key == Qt.Key_Space and not self._space_held:
            self._space_held = True
            self.sm.on_button_press()
        elif key == Qt.Key_Escape:
            self.app.quit()

    def _on_key_release(self, key: int):
        if key == Qt.Key_Space:
            self._space_held = False
            self.sm.on_button_release()

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    OasisApp().run()
