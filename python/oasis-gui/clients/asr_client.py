import os
import httpx
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
        if not resp.is_success:
            print(f"[ASR] Server error {resp.status_code}: {resp.text}")
            return ""
        data = resp.json()
        text = data.get("recognition", "").strip()
        cost = data.get("time_cost", 0)
        print(f"[ASR] '{text}' ({cost:.2f}s)")
        return text
    except Exception as e:
        print(f"[ASR] Error: {e}")
        return ""
