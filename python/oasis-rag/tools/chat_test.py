"""
chat_test.py — O.A.S.I.S. Interactive CLI

Usage:
    python tools/chat_test.py                      # default: gemma3:1b
    python tools/chat_test.py --model qwen3.5:0.8b
    python tools/chat_test.py --model gemma3:4b
    grep -v "^#" tools/test_queries.txt | python tools/chat_test.py -o ./tools/results.txt

Requires:
    - RAG service running on localhost:5001  (python service.py)
    - Ollama running on localhost:11434      (ollama serve)
    - target model pulled                   (ollama pull <model>)
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import time

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import requests

# ── Endpoints ────────────────────────────────────────────────────────────────
RAG_URL    = "http://localhost:5001/retrieve"
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma3:1b"

# ── System prompt (loaded from RAG service — single source: prompt.py) ───────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompt import SAFE_FALLBACK_PROMPT


# ── RAG retrieval ─────────────────────────────────────────────────────────────
def retrieve(query: str) -> tuple[str, str, list[dict], float]:
    """
    Returns (system_prompt, context_str, chunks, rag_latency_ms).
    system_prompt is built by the RAG service from prompt.py (single source of truth).
    """
    try:
        resp = requests.post(RAG_URL, json={"query": query}, timeout=10)
        resp.raise_for_status()
        data          = resp.json()
        system_prompt = data.get("system_prompt", "")
        context       = data.get("context", "")
        chunks        = data.get("chunks", [])
        latency       = data.get("latency_ms", 0.0)
        return system_prompt, context, chunks, latency
    except requests.exceptions.ConnectionError:
        return "", "", [], 0.0
    except Exception as e:
        print(f"  [RAG ERROR] {e}")
        return "", "", [], 0.0


# ── Thinking-mode stripping ────────────────────────────────────────────────────
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks in case any reasoning model leaks them."""
    return _THINK_RE.sub("", text).strip()


# ── LLM call ──────────────────────────────────────────────────────────────────
def call_llm(system: str, query: str, model: str = DEFAULT_MODEL) -> tuple[str, float]:
    """Returns (response_text, elapsed_seconds)."""
    # Qwen3 / reasoning models consume their entire token budget on internal
    # thinking and return an empty content field. Disable thinking mode with
    # the Ollama `think: false` top-level payload key.
    is_thinking_model = any(x in model.lower() for x in ("qwen3", "qwen3.5", "deepseek-r", "phi4"))

    payload = {
        "model":   model,
        "stream":  False,
        "options": {
            "num_predict": 200,
            "temperature": 0.1,
            "repeat_penalty": 1.3,
            "stop": ["**", "Okay", "Let's", "Here's"],
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": query},
        ],
    }
    if is_thinking_model:
        payload["think"] = False
    t0   = time.perf_counter()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    elapsed = time.perf_counter() - t0
    raw = resp.json()["message"]["content"]
    return _strip_think(raw), elapsed


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
    parser = argparse.ArgumentParser(description="O.A.S.I.S. Interactive Test CLI")
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Save all output to FILE (in addition to stdout)",
    )
    args = parser.parse_args()
    model = args.model

    log_file = None
    if args.output:
        log_file = open(args.output, "w", encoding="utf-8")

    def emit(*objects, sep=" ", end="\n"):
        """Print to stdout and optionally mirror to log file."""
        text = sep.join(str(o) for o in objects) + end
        sys.stdout.write(text)
        sys.stdout.flush()
        if log_file:
            log_file.write(text)
            log_file.flush()

    emit()
    emit("  O.A.S.I.S. Interactive Test CLI")
    emit("  ─────────────────────────────────────────────")

    rag_ok, ollama_ok = check_services()

    if not rag_ok:
        emit("  [WARN] RAG service not ready — responses will use safe fallback only")
    else:
        emit("  [OK] RAG service  →  localhost:5001")

    if not ollama_ok:
        emit("  [ERROR] Ollama not reachable — start with: ollama serve")
        emit("  Exiting.")
        if log_file:
            log_file.close()
        sys.exit(1)
    else:
        emit(f"  [OK] Ollama       →  localhost:11434  ({model})")
    if args.output:
        emit(f"  [LOG] Saving output to: {args.output}")

    emit()
    emit('  Type your query below. Enter "quit" or Ctrl-C to exit.')
    emit()

    while True:
        # ── Prompt ────────────────────────────────────────────────────────────
        try:
            query = input("OASIS> ").strip()
        except (KeyboardInterrupt, EOFError):
            emit("\n  Goodbye.")
            break

        if not query or query.startswith("#"):
            continue
        if query.lower() in {"quit", "exit", "q"}:
            emit("  Goodbye.")
            break

        if log_file:
            log_file.write(f"OASIS> {query}\n")
            log_file.flush()

        emit()

        # ── Stage 1: RAG retrieval ────────────────────────────────────────────
        system_prompt, context, chunks, rag_ms = retrieve(query)

        if system_prompt:
            top       = chunks[0] if chunks else {}
            top_src   = top.get("source", "?")
            top_score = top.get("hybrid_score", 0.0)
            emit(
                f"  [RAG] {len(chunks)} chunk(s) found ({rag_ms:.0f}ms)"
                f" | top: {top_src} ({top_score:.2f})"
            )
            if len(chunks) > 1:
                others = ", ".join(
                    f"{c.get('source','?')} ({c.get('hybrid_score',0):.2f})"
                    for c in chunks[1:]
                )
                emit(f"        also: {others}")

            system = system_prompt
        else:
            emit("  [RAG] unavailable — using safe fallback")
            system = SAFE_FALLBACK_PROMPT

        emit(f"  [REF] {context}")

        # ── Stage 2: LLM ──────────────────────────────────────────────────────
        try:
            response, elapsed = call_llm(system, query, model=model)
        except requests.exceptions.ConnectionError:
            emit("  [ERROR] Ollama not reachable.\n")
            continue
        except Exception as e:
            emit(f"  [ERROR] LLM call failed: {e}\n")
            continue

        emit(f"  [LLM] {model} ({elapsed:.1f}s)")
        emit()
        emit(response)
        emit()

    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
