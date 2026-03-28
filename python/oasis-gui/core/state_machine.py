from enum import Enum, auto
from PyQt5.QtCore import QObject, pyqtSignal


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    STREAMING = auto()


# Per-state UI text
STATE_UI = {
    State.IDLE:       ("Ready",         "Press and hold button to speak"),
    State.LISTENING:  ("Listening...",  "Release to send"),
    State.PROCESSING: ("Processing...", "Recognizing..."),
    State.STREAMING:  ("Responding...", "Press button to interrupt"),
}


class StateMachine(QObject):
    state_changed = pyqtSignal(object)  # emits State enum value

    def __init__(self):
        super().__init__()
        self._state = State.IDLE

    @property
    def state(self) -> State:
        return self._state

    def transition(self, new_state: State):
        if self._state == new_state:
            return
        self._state = new_state
        self.state_changed.emit(new_state)

    # ── Named transitions (called by GPIO / pipeline signals) ───────────────

    def on_button_press(self):
        if self._state == State.IDLE:
            self.transition(State.LISTENING)
        elif self._state == State.STREAMING:
            self.transition(State.LISTENING)   # interrupt → new query

    def on_button_release(self):
        if self._state == State.LISTENING:
            self.transition(State.PROCESSING)

    def on_pipeline_started(self):
        """Called when ASR+RAG finished and LLM streaming begins."""
        if self._state == State.PROCESSING:
            self.transition(State.STREAMING)

    def on_pipeline_done(self):
        """Called when LLM stream is complete."""
        if self._state == State.STREAMING:
            self.transition(State.IDLE)
