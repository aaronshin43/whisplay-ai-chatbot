"""
prompt.py — Single source of truth for the O.A.S.I.S. system prompt.

Edit the templates here. Both the Node.js chatbot (via /retrieve API)
and the Python CLI (chat_test.py) consume the same prompt.
"""
from __future__ import annotations

import re

# ── Markdown stripping ────────────────────────────────────────────────────────

def strip_markdown(text: str) -> str:
    """Strip markdown formatting so the 1B LLM doesn't copy it."""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)       # headings
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                     # **bold**
    text = re.sub(r"\*(.+?)\*", r"\1", text)                         # *italic*
    text = re.sub(r"^[ \t]*[-*]\s+", "- ", text, flags=re.MULTILINE) # normalise bullets
    text = re.sub(r"\|[^\n]+\|", "", text)                           # markdown tables
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)           # horizontal rules
    text = re.sub(r"\n{3,}", "\n\n", text)                           # collapse blank lines
    return text


# ── System prompt template ────────────────────────────────────────────────────
#
# Placeholders:
#   {context}  — compressed RAG chunks (inserted by build_system_prompt)
#   {query}    — user's raw query
#

SYSTEM_PROMPT_TEMPLATE = """\
You are OASIS, an emergency first aid assistant.

FORMAT RULES (follow exactly):
1. Identify the specific injury in the TASK. Extract steps ONLY for that condition. IGNORE unrelated conditions in the REFERENCE.
2. Numbered list only: 1. 2. 3.
3. Flatten any sub-bullets into the numbered step.
4. Keep exact numbers (depths, rates, ratios) from REFERENCE.
5. If REFERENCE says "Do not...", that MUST be step 1.
6. STOP writing immediately when the steps for the specific condition end. Do NOT pad the list with generic advice like "Stay calm" or "Call EMS" unless in the REFERENCE.
7. No markdown, no headers, no trailing text after the last step.
8. Use ONLY the REFERENCE below. Do NOT add any outside information.

EXAMPLE:
Reference:
Care for Rib Fracture: Do not wrap a band tightly.
Care for Broken Finger: Tape the broken finger to adjacent uninjured fingers with padding.
Task: Write numbered first aid steps for this emergency: I broke my finger
Response:
1. Tape the broken finger to adjacent uninjured fingers with padding.

REFERENCE:
{context}

TASK: Write numbered first aid steps for this emergency: {query}
RESPONSE:\
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


# ── Builder ───────────────────────────────────────────────────────────────────

def build_system_prompt(context: str, query: str) -> str:
    """
    Format the system prompt with RAG context and user query.
    Applies markdown stripping to the context before insertion.
    Returns the safe fallback prompt if context is empty.
    """
    if not context or not context.strip():
        return SAFE_FALLBACK_PROMPT

    clean_context = strip_markdown(context)
    return SYSTEM_PROMPT_TEMPLATE.format(context=clean_context, query=query)
