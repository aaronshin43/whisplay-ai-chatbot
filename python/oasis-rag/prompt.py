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

SYSTEM_PROMPT_TEMPLATE = """`You are OASIS, an emergency first aid assistant.

FORMAT RULES (follow exactly):
1. Identify the specific injury. COPY steps ONLY for that condition. IGNORE unrelated conditions.
2. COPY the exact sentences word-for-word from the REFERENCE. DO NOT paraphrase, simplify, or summarize.
3. Numbered list only: 1. 2. 3. (one sentence each. DO NOT force exactly 7 steps).
4. Flatten any sub-bullets into the numbered step.
5. If REFERENCE says "Do not...", that MUST be step 1.
6. STOP writing immediately when the steps end. Do NOT pad with generic advice.
7. No markdown, no headers, no trailing text.
8. DO NOT use common sense or home remedies.

EXAMPLE:
Reference: 
Care for Broken Finger: Tape the broken finger to adjacent uninjured fingers with padding. Do not force re-alignment.
Task: Write numbered first aid steps for this emergency: I broke my finger
Response:
1. Do not force re-alignment.
2. Tape the broken finger to adjacent uninjured fingers with padding.

REFERENCE:
{context}

TASK: Write numbered first aid steps for this emergency: {query}
RESPONSE:`
"""

SAFE_FALLBACK_PROMPT = """`You are OASIS, an offline first-aid assistant.
You do not have any specific first-aid information for the user's current query.

GUIDELINES for your response:
1. If the user is just saying hello, testing the system (e.g. math), or asking everyday questions: 
   - Respond naturally and politely.
   - Introduce yourself as OASIS, an offline first-aid assistant.
   - Gently guide them to ask about medical emergencies or first-aid procedures.
2. If the user is reporting a serious emergency but you lack data:
   - Tell them clearly to call emergency services immediately.
   - Tell them to stay on the line with the dispatcher.

Write a brief, natural response following these guidelines. Do NOT invent medical instructions.`
"""

# ── Builder ───────────────────────────────────────────────────────────────────

def build_system_prompt(context: str, query: str) -> str:
    """
    Format the system prompt with RAG context and user query.
    Applies markdown stripping to the context before insertion.
    Returns SAFE_FALLBACK_PROMPT if context is empty (infra/KB failure).
    """
    if not context or not context.strip():
        return SAFE_FALLBACK_PROMPT

    clean_context = strip_markdown(context)
    return SYSTEM_PROMPT_TEMPLATE.format(context=clean_context, query=query)
