"""Shared utilities for O.A.S.I.S. RAG tools/."""
from __future__ import annotations

import statistics
import textwrap

SEP  = "=" * 68
SEP2 = "-" * 68


def _safe(s: str, width: int = 0) -> str:
    """ASCII-safe string for Windows cp949 console. Optionally truncates to width."""
    out = str(s).encode("ascii", errors="replace").decode("ascii")
    return textwrap.shorten(out, width=width, placeholder="...") if width else out


def _token_count(text: str) -> int:
    """Approximate token count (whitespace split)."""
    return len(text.split())


def _stats(samples: list[float]) -> dict:
    """Compute mean, std, min, max, p95 for a list of latency samples."""
    if len(samples) < 2:
        v = samples[0] if samples else 0.0
        return {"mean": v, "std": 0.0, "min": v, "max": v, "p95": v}
    return {
        "mean": statistics.mean(samples),
        "std":  statistics.stdev(samples),
        "min":  min(samples),
        "max":  max(samples),
        "p95":  sorted(samples)[int(len(samples) * 0.95)],
    }
