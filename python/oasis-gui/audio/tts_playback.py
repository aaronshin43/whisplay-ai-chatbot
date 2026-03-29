import os
import signal
import subprocess
import tempfile
import platform
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
            try:
                self._queue.get_nowait()
            except Empty:
                break
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
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

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
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
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
            if IS_PI:
                play_cmd = ["play", "-D", f"hw:{SOUND_CARD_INDEX},0", wav_path, "-q"]
            else:
                play_cmd = ["play", wav_path, "-q"]
            self._current_process = subprocess.Popen(
                play_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
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
