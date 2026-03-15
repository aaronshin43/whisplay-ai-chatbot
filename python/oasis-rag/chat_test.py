"""
chat_test.py — O.A.S.I.S. Interactive CLI

Usage:
    python chat_test.py

Requires:
    - RAG service running on localhost:5001  (python service.py)
    - Ollama running on localhost:11434      (ollama serve)
    - gemma3:1b pulled                      (ollama pull gemma3:1b)
"""
from __future__ import annotations

import io
import sys
import time

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import requests

# ── Endpoints ────────────────────────────────────────────────────────────────
RAG_URL    = "http://localhost:5001/retrieve"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "gemma3:1b"

# ── Spinal injury detection ───────────────────────────────────────────────────
_SPINAL_SIGNALS = [
    "cannot feel", "can't feel", "cant feel",
    "paralyz", "numb legs", "numb feet", "numb limb",
    "neck injury", "spinal", "spine",
]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """\
You are OASIS, an offline first-aid assistant.
You respond ONLY based on the REFERENCE below.
Rules:
1. Maximum 5 numbered steps. Plain text only.
2. Each step under 15 words.
3. If supplies unavailable, suggest alternatives.
4. Never diagnose. Never prescribe medication.
5. If unsure: Call emergency services immediately.
6. If panicking: Start with 'Take a deep breath. I will guide you.'
7. Begin directly with step 1. No preambles, disclaimers, or introductions.
8. Answer in your own words. Never copy or reproduce the reference text.

REFERENCE:
{context}\
"""

SAFE_FALLBACK_PROMPT = """\
You are OASIS, an offline first-aid assistant.
The medical knowledge base is currently unavailable.
Tell the user clearly and calmly:
1. Call emergency services immediately (local emergency number).
2. Stay on the line with the dispatcher — they will guide you.
3. Do not leave the person alone.
Do not provide any specific medical instructions without the knowledge base.\
"""


# ── RAG retrieval ─────────────────────────────────────────────────────────────
def retrieve(query: str) -> tuple[str, list[dict], float]:
    """
    Returns (context_str, chunks, rag_latency_ms).
    context_str is empty string if service is down.
    """
    try:
        resp = requests.post(RAG_URL, json={"query": query}, timeout=10)
        resp.raise_for_status()
        data      = resp.json()
        context   = data.get("context", "")
        chunks    = data.get("chunks", [])
        latency   = data.get("latency_ms", 0.0)
        return context, chunks, latency
    except requests.exceptions.ConnectionError:
        return "", [], 0.0
    except Exception as e:
        print(f"  [RAG ERROR] {e}")
        return "", [], 0.0


# ── LLM call ──────────────────────────────────────────────────────────────────
def call_llm(system: str, query: str) -> tuple[str, float]:
    """Returns (response_text, elapsed_seconds)."""
    payload = {
        "model":   MODEL,
        "stream":  False,
        "options": {"num_predict": 200, "temperature": 0.0},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": query},
        ],
    }
    t0   = time.perf_counter()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    elapsed = time.perf_counter() - t0
    return resp.json()["message"]["content"].strip(), elapsed


# ── Preflight check ───────────────────────────────────────────────────────────
def check_services() -> tuple[bool, bool]:
    rag_ok, ollama_ok = False, False
    try:
        r = requests.get("http://localhost:5001/health", timeout=3)
        rag_ok = r.status_code == 200 and r.json().get("index_ready", False)
    except Exception:
        pass
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        pass
    return rag_ok, ollama_ok


# ── Main loop ─────────────────────────────────────────────────────────────────
def main() -> None:
    print()
    print("  O.A.S.I.S. Interactive Test CLI")
    print("  ─────────────────────────────────────────────")

    rag_ok, ollama_ok = check_services()

    if not rag_ok:
        print("  [WARN] RAG service not ready — responses will use safe fallback only")
    else:
        print("  [OK] RAG service  →  localhost:5001")

    if not ollama_ok:
        print("  [ERROR] Ollama not reachable — start with: ollama serve")
        print("  Exiting.")
        sys.exit(1)
    else:
        print(f"  [OK] Ollama       →  localhost:11434  ({MODEL})")

    print()
    print('  Type your query below. Enter "quit" or Ctrl-C to exit.')
    print()

    while True:
        # ── Prompt ────────────────────────────────────────────────────────────
        try:
            query = input("OASIS> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("  Goodbye.")
            break

        print()

        # ── Stage 1: RAG retrieval ────────────────────────────────────────────
        context, chunks, rag_ms = retrieve(query)

        if context:
            # Spinal injection
            q_lower = query.lower()
            if any(sig in q_lower for sig in _SPINAL_SIGNALS):
                context = "CRITICAL: Possible spinal cord injury. Do not move the person.\n\n" + context

            top       = chunks[0] if chunks else {}
            top_src   = top.get("source", "?")
            top_score = top.get("hybrid_score", 0.0)
            print(
                f"  [RAG] {len(chunks)} chunk(s) found ({rag_ms:.0f}ms)"
                f" | top: {top_src} ({top_score:.2f})"
            )
            # All sources
            if len(chunks) > 1:
                others = ", ".join(
                    f"{c.get('source','?')} ({c.get('hybrid_score',0):.2f})"
                    for c in chunks[1:]
                )
                print(f"        also: {others}")

            system = SYSTEM_PROMPT_TEMPLATE.format(context=context)
        else:
            print("  [RAG] unavailable — using safe fallback")
            system = SAFE_FALLBACK_PROMPT

        # ── Stage 2: LLM ──────────────────────────────────────────────────────
        try:
            response, elapsed = call_llm(system, query)
        except requests.exceptions.ConnectionError:
            print("  [ERROR] Ollama not reachable.\n")
            continue
        except Exception as e:
            print(f"  [ERROR] LLM call failed: {e}\n")
            continue

        print(f"  [LLM] {MODEL} ({elapsed:.1f}s)")
        print()
        print(response)
        print()


if __name__ == "__main__":
    main()
