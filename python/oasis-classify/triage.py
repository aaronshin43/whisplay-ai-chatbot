"""
triage.py — Triage prompt template for ambiguous medical queries.

Activates when the classifier score is in [OOD_FLOOR, CLASSIFY_THRESHOLD).
The LLM uses this prompt to ask a single clarifying question.
"""

from __future__ import annotations

TRIAGE_PROMPT_TEMPLATE = """\
You are OASIS, a first-aid assistant.
You could not identify the specific emergency type.
Ask ONE short clarifying question to understand the situation better.
If the person sounds panicked about a life-threatening situation, tell them to call 911 immediately.

User said: {query}
Response:"""


def build_triage_prompt(query: str) -> str:
    """Build the triage system prompt for a given query."""
    return TRIAGE_PROMPT_TEMPLATE.format(query=query)
