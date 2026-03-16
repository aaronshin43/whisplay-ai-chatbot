"""
chat_test.py — O.A.S.I.S. Interactive CLI

Usage:
    python chat_test.py                      # default: gemma3:1b
    python chat_test.py --model qwen3.5:0.8b
    python chat_test.py --model gemma3:4b

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

# ── Context injection signals ─────────────────────────────────────────────────
_CARDIAC_ARREST_SIGNALS = [
    "collapsed not breathing", "not breathing no pulse", "no pulse not breathing",
    "cardiac arrest", "no pulse no breath", "not breathing and no pulse",
    "collapsed no pulse",
]

_SPINAL_SIGNALS = [
    "cannot feel", "can't feel", "cant feel",
    "paralyz", "numb legs", "numb feet", "numb limb",
    "neck injury", "spinal", "spine",
]

_FROSTBITE_SIGNALS = [
    "frostbite", "frost bite", "frostbitten", "frozen finger", "frozen toe",
]

_PANIC_BLOOD_SIGNALS = [
    "theres so much blood", "there's so much blood", "so much blood",
]

_NO_EPIPEN_SIGNALS = [
    "no epipen", "no epinephrine", "dont have epipen",
    "don't have epipen", "without epipen", "no auto-injector",
]

_LIGHTNING_SIGNALS = [
    "lightning", "thunder", "struck by lightning", "lightning strike",
    "lightning coming", "lightning outside",
]

_BURN_SIGNALS = [
    "burn", "burnt", "scalded", "scald", "boiling water", "hot water on",
    "on fire", "caught fire", "flame",
]

_SNAKEBITE_SIGNALS = [
    "snake", "snakebite", "snake bite", "snake bit", "bitten by snake", "venom",
]

_HYPOTHERMIA_SIGNALS = [
    "hypothermia", "hypothermic", "stopped shivering", "stop shivering",
    "freezing person", "frozen person",
]

_HEAT_STROKE_SIGNALS = [
    "heat stroke", "heatstroke", "heat exhaustion",
    "hot skin", "not sweating", "overheated",
]

_CHOKING_SIGNALS = [
    "choking", "chok", "can't cough", "cant cough", "turning blue",
    "unable to cough", "foreign body airway",
]

_HEART_ATTACK_SIGNALS = [
    "heart attack", "having a heart attack", "think i have a heart attack",
    "think i'm having",
]

_BLOCKED_MEDICAL_TERMS = [
    "antibiotic", "ceftriaxone", "cefotaxime", "amoxicillin",
    "clindamycin", "vancomycin", "broad-spectrum", "azithromycin",
    "penicillin", "cephalosporin", "ciprofloxacin", "metronidazole",
    "tetracycline", "doxycycline",
]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """\
You are OASIS. A person needs first aid RIGHT NOW.

RULES YOU MUST FOLLOW:
- Your response is ONLY numbered steps 1 through 5.
- Do NOT write anything before "1."
- Do NOT write anything after step 5.
- Each step is ONE sentence, maximum 12 words.
- Do NOT use asterisks, bold, markdown, or headers.
- Do NOT ask questions. Give commands only.
- Do NOT say "Okay" or "Let's" or any introduction.

REFERENCE:
{context}

YOUR RESPONSE MUST START WITH "1." AND END AFTER STEP 5. NOTHING ELSE.\
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
            "num_predict": 150,
            "temperature": 0.1,
            "stop": ["6.", "**", "Okay", "Let's", "Here's"],
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
    args = parser.parse_args()
    model = args.model

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
        print(f"  [OK] Ollama       →  localhost:11434  ({model})")

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
            q_lower = query.lower()

            if any(sig in q_lower for sig in _CARDIAC_ARREST_SIGNALS):
                context = (
                    "CARDIAC ARREST PROTOCOL — ACT NOW:\n"
                    "1. CALL emergency services (911/999/112) immediately.\n"
                    "2. BEGIN chest compressions: push hard and fast on centre of chest.\n"
                    "3. Rate: 100-120 compressions per minute. Depth: 5-6 cm.\n"
                    "4. After 30 compressions, give 2 rescue breaths.\n"
                    "5. Continue 30:2 cycle until emergency services arrive.\n\n"
                ) + context

            if any(sig in q_lower for sig in _SPINAL_SIGNALS):
                context = context + (
                    "\n\n⚠ OVERRIDE: SUSPECTED SPINAL CORD INJURY.\n"
                    "STEP 1 MUST BE: Do NOT move the person. Keep head and neck completely still.\n"
                    "Do NOT perform any assessment that requires moving the patient.\n"
                )

            if any(sig in q_lower for sig in _FROSTBITE_SIGNALS):
                context = (
                    "CRITICAL: This is FROSTBITE (cold injury).\n"
                    "Move to WARM shelter immediately. Do NOT move to cool or shaded area.\n"
                    "Rewarm the affected area with WARM (not hot) water 37-39°C.\n"
                    "Do NOT rub the frostbitten area. Do NOT use snow or cold water.\n\n"
                ) + context

            if any(sig in q_lower for sig in _PANIC_BLOOD_SIGNALS):
                context = (
                    "EMERGENCY BLEEDING PROTOCOL:\n"
                    "- CALL EMERGENCY SERVICES immediately.\n"
                    "- Apply direct PRESSURE to the wound with your hands.\n"
                    "- Use cloth, shirt or any fabric and press firmly.\n\n"
                ) + context

            if any(sig in q_lower for sig in _NO_EPIPEN_SIGNALS):
                context = (
                    "CRITICAL: NO EPINEPHRINE available. Epipen NOT available.\n"
                    "MANDATORY FIRST STEPS:\n"
                    "1. Call emergency services immediately (911/999/112).\n"
                    "2. Lay the person flat, legs elevated if no breathing difficulty.\n"
                    "3. Give antihistamine if available.\n\n"
                ) + context

            if any(sig in q_lower for sig in _LIGHTNING_SIGNALS):
                context = (
                    "LIGHTNING SAFETY PROTOCOL:\n"
                    "1. Do NOT stand under trees, poles, or tall objects.\n"
                    "2. CROUCH LOW on balls of feet, feet together, head down.\n"
                    "3. Keep 20 metres from other people.\n"
                    "4. Move to a solid building or hard-topped vehicle if reachable.\n"
                    "5. Stay away from open fields, hilltops, water, and metal objects.\n\n"
                ) + context

            if any(sig in q_lower for sig in _BURN_SIGNALS):
                context = context + (
                    "\n\n⚠ BURN PROTOCOL — MANDATORY STEP 1:\n"
                    "COOL the burn under COOL running water for 20 minutes.\n"
                    "Do NOT skip this step. Do NOT use ice, butter, or warm/hot water.\n"
                    "After cooling: remove jewellery, cover loosely with cling film.\n"
                    "Call emergency services for large, deep, or facial burns.\n"
                )

            if any(sig in q_lower for sig in _CHOKING_SIGNALS):
                context = (
                    "CHOKING PROTOCOL — PERFORM NOW:\n"
                    "1. Give 5 firm BACK BLOWS between shoulder blades with heel of hand.\n"
                    "2. Give 5 ABDOMINAL THRUSTS (Heimlich): stand behind, pull inward and upward.\n"
                    "3. Alternate 5 back blows + 5 abdominal thrusts until object clears.\n"
                    "4. If unconscious: lower to ground, call 911, begin CPR.\n"
                    "5. Do NOT do a blind finger sweep.\n\n"
                ) + context

            if any(sig in q_lower for sig in _SNAKEBITE_SIGNALS):
                context = (
                    "SNAKEBITE PROTOCOL:\n"
                    "1. KEEP THE PERSON STILL and calm — movement spreads venom faster.\n"
                    "2. Immobilize the bitten limb at or below heart level.\n"
                    "3. Remove watches, rings, tight clothing from the affected limb.\n"
                    "4. Call emergency services or transport to hospital URGENTLY.\n"
                    "5. Do NOT cut the wound, suck the venom, or apply tourniquet.\n\n"
                ) + context

            if any(sig in q_lower for sig in _HYPOTHERMIA_SIGNALS):
                context = (
                    "HYPOTHERMIA PROTOCOL — This is COLD INJURY, NOT heat illness:\n"
                    "1. Move the person to WARM shelter immediately.\n"
                    "2. Remove wet clothing; replace with dry insulation (blankets, sleeping bag).\n"
                    "3. Warm the core (trunk/torso) first, not extremities.\n"
                    "4. Give warm fluids ONLY if the person is conscious and can swallow.\n"
                    "5. Handle gently — a cold heart is prone to dangerous arrhythmia.\n"
                    "Do NOT cool this person. Do NOT give cold fluids. Do NOT rub vigorously.\n\n"
                ) + context

            if any(sig in q_lower for sig in _HEAT_STROKE_SIGNALS):
                context = (
                    "HEAT STROKE EMERGENCY — ACT IMMEDIATELY:\n"
                    "1. MOVE the person to shade or a cool area NOW.\n"
                    "2. Remove excess clothing.\n"
                    "3. Cool with cool water — douse, spray, or immerse in cool water.\n"
                    "4. Fan the person while keeping them wet.\n"
                    "5. Call emergency services. Heat stroke is life-threatening.\n\n"
                ) + context

            if any(sig in q_lower for sig in _HEART_ATTACK_SIGNALS):
                context = (
                    "HEART ATTACK PROTOCOL:\n"
                    "1. CALL emergency services IMMEDIATELY (911/999/112).\n"
                    "2. Sit or lie the person down in a comfortable position.\n"
                    "3. Loosen tight clothing around neck and chest.\n"
                    "4. If conscious and not allergic: chew one adult aspirin (300 mg).\n"
                    "5. Do NOT leave the person alone. Monitor breathing.\n\n"
                ) + context

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
            response, elapsed = call_llm(system, query, model=model)
        except requests.exceptions.ConnectionError:
            print("  [ERROR] Ollama not reachable.\n")
            continue
        except Exception as e:
            print(f"  [ERROR] LLM call failed: {e}\n")
            continue

        print(f"  [LLM] {model} ({elapsed:.1f}s)")
        print()
        print(response)
        print()


if __name__ == "__main__":
    main()
