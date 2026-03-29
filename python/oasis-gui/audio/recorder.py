import os
import signal
import time
import subprocess
import platform

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
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

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
