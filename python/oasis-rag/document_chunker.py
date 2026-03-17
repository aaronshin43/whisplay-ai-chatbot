"""
document_chunker.py — O.A.S.I.S. RAG Phase 1

Loads Markdown files from a knowledge directory, splits them into
overlapping token-based chunks, and preserves section metadata
(file name, heading hierarchy) for retrieval context.

Chunking strategy:
  - Target chunk size : 400 tokens  (≈ 300 words)
  - Overlap           : 75  tokens  (≈ 56 words)
  - Tokenizer         : whitespace split (no external dependency)
    → swap for tiktoken/transformers tokenizer when available on Pi

Usage:
    from document_chunker import DocumentChunker

    chunker = DocumentChunker()
    chunks  = chunker.load_and_chunk("data/knowledge")
    # chunks: list[dict]  — see Chunk schema below
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE    = 400   # tokens
DEFAULT_CHUNK_OVERLAP = 75    # tokens
SUPPORTED_EXTENSIONS  = {".md", ".txt"}


# ─────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    """A single retrievable unit of text."""
    text:       str
    source:     str          # relative file path
    section:    str          # closest heading above this chunk
    headings:   list[str]    # full heading breadcrumb, outermost → inner
    chunk_idx:  int          # 0-based index within the source document
    token_count: int

    def to_dict(self) -> dict:
        prefix = f"[Source: {self.source} | Section: {self.section}]\n"
        return {
            "text":             self.text,
            "text_with_prefix": prefix + self.text,
            "source":           self.source,
            "section":          self.section,
            "headings":         self.headings,
            "chunk_idx":        self.chunk_idx,
            "token_count":      self.token_count,
        }


# ─────────────────────────────────────────────────────────────
# Tokenizer (whitespace — swap easily)
# ─────────────────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    """Minimal whitespace tokenizer. Replace with tiktoken for precision."""
    return text.split()


def _detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────
# Markdown heading parser
# ─────────────────────────────────────────────────────────────
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _extract_heading_breadcrumb(text_before: str) -> list[str]:
    """
    Return the active heading hierarchy at the point just before `text_before`
    ends.  e.g. ["# Severe Bleeding", "## Direct Pressure Technique"]
    """
    headings: list[tuple[int, str]] = []   # (level, text)
    for m in _HEADING_RE.finditer(text_before):
        level = len(m.group(1))
        title = m.group(2).strip()
        # pop any sibling/child headings that are deeper or equal
        while headings and headings[-1][0] >= level:
            headings.pop()
        headings.append((level, title))
    return [h[1] for h in headings]


# ─────────────────────────────────────────────────────────────
# Core chunker
# ─────────────────────────────────────────────────────────────
class DocumentChunker:
    """
    Loads Markdown/text files and splits them into overlapping chunks.

    Parameters
    ----------
    chunk_size    : target token count per chunk
    chunk_overlap : token overlap between consecutive chunks
    """

    def __init__(
        self,
        chunk_size:    int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_chunk(self, knowledge_dir: str | Path) -> list[Chunk]:
        """
        Walk *knowledge_dir*, load every supported file, and return all chunks.

        Returns
        -------
        list[Chunk]  sorted by (source, chunk_idx)
        """
        knowledge_dir = Path(knowledge_dir)
        if not knowledge_dir.is_dir():
            raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

        all_chunks: list[Chunk] = []
        for file_path in sorted(knowledge_dir.rglob("*")):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            relative = str(file_path.relative_to(knowledge_dir))
            chunks = self._chunk_file(file_path, relative)
            all_chunks.extend(chunks)

        return all_chunks

    def chunk_text(self, text: str, source: str = "<inline>") -> list[Chunk]:
        """Chunk a raw string directly (useful for testing)."""
        return list(self._sliding_window(text, source))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_file(self, path: Path, source: str) -> list[Chunk]:
        text = path.read_text(encoding="utf-8")
        return list(self._sliding_window(text, source))

    def _sliding_window(self, text: str, source: str) -> Iterator[Chunk]:
        """
        Yield chunks using a token-level sliding window.
        Each chunk carries the Markdown heading breadcrumb active at its start.
        """
        tokens = _tokenize(text)
        total  = len(tokens)

        if total == 0:
            return

        step       = self.chunk_size - self.chunk_overlap
        chunk_idx  = 0
        char_pos   = 0   # approximate character cursor for heading lookup

        start = 0
        while start < total:
            end        = min(start + self.chunk_size, total)
            chunk_toks = tokens[start:end]
            chunk_text = _detokenize(chunk_toks)

            # Approximate char position for heading extraction
            # (join tokens up to `start` to find context)
            prefix_text = _detokenize(tokens[:start])
            breadcrumb  = _extract_heading_breadcrumb(prefix_text + "\n" + chunk_text)
            section     = breadcrumb[-1] if breadcrumb else source

            yield Chunk(
                text        = chunk_text,
                source      = source,
                section     = section[:80],   # cap: heading text only, no body
                headings    = breadcrumb,
                chunk_idx   = chunk_idx,
                token_count = len(chunk_toks),
            )

            chunk_idx += 1
            start     += step

            if end == total:
                break


# ─────────────────────────────────────────────────────────────
# CLI smoke-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    knowledge_dir = sys.argv[1] if len(sys.argv) > 1 else "data/knowledge"

    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks  = chunker.load_and_chunk(knowledge_dir)

    def _safe(s: str) -> str:
        return s.encode("ascii", errors="replace").decode("ascii")

    print(f"[DocumentChunker] Loaded {len(chunks)} chunks from '{knowledge_dir}'")
    for c in chunks[:3]:
        print(f"\n--- Chunk {c.chunk_idx} | {c.source} | section: {_safe(c.section[:60])} ---")
        print(f"Tokens: {c.token_count}")
        print(_safe(c.text[:200]) + ("..." if len(c.text) > 200 else ""))
