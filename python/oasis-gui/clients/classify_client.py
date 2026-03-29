import os, httpx
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

CLASSIFY_URL = os.getenv("OASIS_CLASSIFY_SERVICE_URL", "http://127.0.0.1:5002")
TIMEOUT = float(os.getenv("OASIS_CLASSIFY_TIMEOUT_MS", "5000")) / 1000.0

# Persistent client — reuses TCP connection across calls.
# Avoids per-call DNS resolution and TCP handshake overhead (~2s on Windows localhost).
_client = httpx.Client(timeout=TIMEOUT)

SAFE_FALLBACK_TEXT = (
    "I cannot reach the emergency knowledge system. "
    "Please call 911 immediately for any life-threatening emergency."
)

@dataclass
class DispatchResult:
    mode: Literal["direct_response", "llm_prompt", "triage_prompt", "ood_response"]
    response_text: str | None
    system_prompt: str | None
    category: str | None          # always set for triage_prompt
    score: float | None
    threshold_path: str
    latency_ms: float
    hint_changed_result: bool

def _safe_fallback(threshold_path: str) -> DispatchResult:
    return DispatchResult(
        mode="ood_response",
        response_text=SAFE_FALLBACK_TEXT,
        system_prompt=None,
        category=None,
        score=None,
        threshold_path=threshold_path,
        latency_ms=0.0,
        hint_changed_result=False,
    )

def dispatch(query: str, prev_triage_hint: str | None) -> DispatchResult:
    """POST /dispatch — resolves, never raises.

    On any failure returns a safe ood_response fallback with a
    diagnostic threshold_path so logs distinguish infra failures from real OOD.
    """
    try:
        resp = _client.post(
            f"{CLASSIFY_URL}/dispatch",
            json={"query": query, "prev_triage_hint": prev_triage_hint},
        )
        resp.raise_for_status()
        d = resp.json()

        # Validate required fields — return invalid_schema fallback if malformed
        required = ("mode", "threshold_path", "latency_ms", "hint_changed_result")
        if not all(k in d for k in required):
            print(f"[Classify] Response missing required fields — invalid_schema")
            return _safe_fallback("invalid_schema")

        result = DispatchResult(
            mode=d["mode"],
            response_text=d.get("response_text"),
            system_prompt=d.get("system_prompt"),
            category=d.get("category"),
            score=d.get("score"),
            threshold_path=d["threshold_path"],
            latency_ms=d["latency_ms"],
            hint_changed_result=d["hint_changed_result"],
        )
        print(
            f"[Classify] mode={result.mode}  cat={result.category}  "
            f"score={f'{result.score:.3f}' if result.score is not None else '—'}  "
            f"path={result.threshold_path}  latency={result.latency_ms:.1f}ms  "
            f"hint_changed={result.hint_changed_result}"
        )
        return result

    except httpx.ConnectError:
        print(f"[Classify] Service not reachable ({CLASSIFY_URL}) — network_error")
        return _safe_fallback("network_error")
    except httpx.TimeoutException:
        print(f"[Classify] Timeout after {TIMEOUT}s — network_error")
        return _safe_fallback("network_error")
    except Exception as e:
        print(f"[Classify] Unexpected error: {e} — service_error")
        return _safe_fallback("service_error")

def is_healthy() -> bool:
    try:
        resp = _client.get(f"{CLASSIFY_URL}/health")
        return resp.json().get("status") == "ok"
    except Exception:
        return False
