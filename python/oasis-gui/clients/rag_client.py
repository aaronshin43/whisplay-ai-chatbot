import os
import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

RAG_URL = os.getenv("OASIS_RAG_SERVICE_URL", "http://localhost:5001")
TIMEOUT = float(os.getenv("OASIS_RAG_TIMEOUT_MS", "5000")) / 1000.0

# Ported from OasisAdapter.ts — used when RAG is unavailable
SAFE_FALLBACK_PROMPT = """You are OASIS, an offline first-aid assistant.
The medical knowledge base is currently unavailable.
Tell the user clearly and calmly:
1. Call emergency services immediately (local emergency number).
2. Stay on the line with the dispatcher — they will guide you.
3. Do not leave the person alone.
Do not provide any specific medical instructions without the knowledge base."""


def retrieve_system_prompt(query: str) -> str:
    """Fetch system prompt from RAG Flask service.

    Returns SAFE_FALLBACK_PROMPT if the service is unavailable or returns empty.
    Never returns an empty string — always safe for medical use.
    """
    try:
        resp = httpx.post(
            f"{RAG_URL}/retrieve",
            json={"query": query, "top_k": 4, "compress": True},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        prompt = resp.json().get("system_prompt", "")
        latency = resp.json().get("latency_ms", 0)
        chunks = len(resp.json().get("chunks", []))
        if prompt.strip():
            print(f"[RAG] Retrieved {chunks} chunks in {latency:.1f}ms")
            return prompt
        print("[RAG] Empty prompt returned — using safe fallback")
    except httpx.ConnectError:
        print(f"[RAG] Service not reachable ({RAG_URL}) — using safe fallback")
    except httpx.TimeoutException:
        print(f"[RAG] Timeout after {TIMEOUT}s — using safe fallback")
    except Exception as e:
        print(f"[RAG] Unexpected error: {e} — using safe fallback")

    return SAFE_FALLBACK_PROMPT


def is_healthy() -> bool:
    """Check if the RAG service is up and index is ready."""
    try:
        resp = httpx.get(f"{RAG_URL}/health", timeout=3.0)
        data = resp.json()
        return data.get("status") == "ok" and data.get("index_ready", False)
    except Exception:
        return False
