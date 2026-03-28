import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

# Persistent client — reuses TCP connection to Ollama across all calls.
# Avoids ~2s per-call overhead from DNS resolution + TCP handshake on Windows.
_client = httpx.Client(timeout=60.0)

# OASIS-mode generation constraints (matches ollama-llm.ts)
_OASIS_OPTIONS = {
    "num_predict": 200,
    "temperature": 0.1,
    "repeat_penalty": 1.3,
}
_OASIS_STOP = ["**", "Okay", "Let's", "Here's"]


def prewarm():
    """Send a minimal request to load the model into memory.

    Blocking call — run once on startup before the GUI becomes Ready.
    Without this the first real query takes 10-20s extra on Pi.
    """
    try:
        _client.post(
            f"{OLLAMA_ENDPOINT}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "keep_alive": -1,
                "options": {"num_predict": 1, "temperature": 0.0},
            },
        )
        print(f"[LLM] Model '{OLLAMA_MODEL}' pre-warmed.")
    except Exception as e:
        print(f"[LLM] Pre-warm failed (Ollama not ready?): {e}")


def stream(messages: list, on_token, on_done, abort_flag_ref: list):
    """Stream tokens from Ollama synchronously (run inside a QThread).

    Args:
        messages:      List of {"role": ..., "content": ...} dicts.
        on_token:      Callable(str) — called for each token.
        on_done:       Callable() — called when stream ends.
        abort_flag_ref: Single-element list [bool] — set [True] to abort.
    """
    try:
        with _client.stream(
            "POST",
            f"{OLLAMA_ENDPOINT}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": True,
                "keep_alive": -1,
                "options": _OASIS_OPTIONS,
                "stop": _OASIS_STOP,
            },
        ) as response:
            for line in response.iter_lines():
                if abort_flag_ref[0]:
                    break
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        on_token(token)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[LLM] Stream error: {e}")
    finally:
        on_done()
