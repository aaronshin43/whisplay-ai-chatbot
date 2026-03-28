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
# Section-aware chunker
# ─────────────────────────────────────────────────────────────

MIN_SECTION_TOKENS = 80   # sections smaller than this are merged with next

# Pattern matching H1 (#) headings — used for parent tracking when splitting at H2
_H1_RE = re.compile(r"^#{1}(?!#)\s+(.+)$", re.MULTILINE)
# Pattern matching H2 (##) headings — used for parent tracking when splitting at H3
_H2_RE = re.compile(r"^#{2}(?!#)\s+(.+)$", re.MULTILINE)
# Splits at H2 (##) OR H3 (###) boundaries.
# H2-only files (e.g. burns.md) previously produced a single section spanning the
# entire document; splitting at H2 as well gives one chunk per major section.
_SPLIT_RE = re.compile(r"^#{2,3}(?!#)\s+(.+)$", re.MULTILINE)


def _h1_parent(pos: int, text: str) -> str:
    """Return the most recent H1 heading title before *pos* in *text*."""
    parent = ""
    for m in _H1_RE.finditer(text, 0, pos):
        parent = m.group(1).strip()
    return parent


def _h2_parent(pos: int, text: str) -> str:
    """Return the most recent H2 heading title before *pos* in *text*."""
    parent = ""
    for m in _H2_RE.finditer(text, 0, pos):
        parent = m.group(1).strip()
    return parent


class _Section:
    """Represents a markdown section (H3 or lower-level boundary)."""

    __slots__ = ("heading", "h2_parent", "text", "tokens")

    def __init__(self, heading: str, h2_parent: str, text: str) -> None:
        self.heading   = heading
        self.h2_parent = h2_parent
        self.text      = text
        self.tokens    = len(_tokenize(text))


class SectionAwareChunker:
    """
    Splits documents at H3 (###) boundaries, then:
      1. Merges sections smaller than MIN_SECTION_TOKENS with the next
         section, provided they share the same H2 parent.
      2. Falls back to sliding-window chunking for merged sections that
         still exceed chunk_size.

    This prevents tiny 10-20 token sections (common in medical reference
    lists) from degrading embedding quality while still keeping semantically
    distinct conditions in separate chunks.

    Parameters
    ----------
    chunk_size    : maximum tokens per chunk (after merging)
    chunk_overlap : overlap when sliding-window fallback is used
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
        self._fallback     = DocumentChunker(chunk_size, chunk_overlap)

    # ------------------------------------------------------------------
    # Public API (same as DocumentChunker)
    # ------------------------------------------------------------------

    def load_and_chunk(self, knowledge_dir: str | Path) -> list[Chunk]:
        knowledge_dir = Path(knowledge_dir)
        if not knowledge_dir.is_dir():
            raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

        all_chunks: list[Chunk] = []
        for file_path in sorted(knowledge_dir.rglob("*")):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            relative = str(file_path.relative_to(knowledge_dir))
            all_chunks.extend(self._chunk_file(file_path, relative))

        return all_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_file(self, path: Path, source: str) -> list[Chunk]:
        text = path.read_text(encoding="utf-8")
        return self._section_split(text, source)

    def _section_split(self, text: str, source: str) -> list[Chunk]:
        """Split *text* at H3 boundaries, merge tiny sections, yield Chunks."""

        # ── Step 1: Locate all H1/H2/H3 split points ──────────────────
        boundaries: list[int] = [0]
        for m in _SPLIT_RE.finditer(text):
            if m.start() > 0:
                boundaries.append(m.start())
        boundaries.append(len(text))

        # ── Step 2: Build raw sections ────────────────────────────────
        raw: list[_Section] = []
        for i in range(len(boundaries) - 1):
            s_start = boundaries[i]
            s_end   = boundaries[i + 1]
            body    = text[s_start:s_end].strip()
            if not body:
                continue
            # Determine heading and parent for this section.
            # H3 sections use the enclosing H2 as parent.
            # H2 sections use the enclosing H1 as parent.
            first_line = body.split("\n", 1)[0].strip()
            m3 = re.match(r"^#{3}(?!#)\s+(.+)$", first_line)
            m2 = re.match(r"^#{2}(?!#)\s+(.+)$", first_line)
            if m3:
                heading = m3.group(1).strip()
                h2p = _h2_parent(s_start, text)
            elif m2:
                heading = m2.group(1).strip()
                h2p = _h1_parent(s_start, text)
            else:
                heading = source
                h2p = _h1_parent(s_start, text)
            # Skip heading-only sections: an H2 that is immediately followed by
            # H3 sub-sections produces a body consisting of just the heading line.
            # These orphan sections are unembeddable noise; H3 chunks below will
            # carry the semantic content.
            body_content = body[len(first_line):].strip()
            if not body_content:
                continue
            raw.append(_Section(heading=heading, h2_parent=h2p, text=body))

        if not raw:
            return []

        # ── Step 3: Merge tiny sections forward ───────────────────────
        merged: list[_Section] = []
        i = 0
        while i < len(raw):
            sec = raw[i]
            # Accumulate forward while tiny AND:
            #   (a) sibling: next section shares the same H2 parent, OR
            #   (b) child:   next section is an H3 directly under this H2
            #                (its h2_parent equals our heading)
            # Case (b) handles H2 intro paragraphs that are immediately followed
            # by H3 sub-sections — they can't merge sideways but should merge down.
            while (
                sec.tokens < MIN_SECTION_TOKENS
                and i + 1 < len(raw)
                and (
                    raw[i + 1].h2_parent == sec.h2_parent   # (a) sibling
                    or raw[i + 1].h2_parent == sec.heading  # (b) child
                )
            ):
                i += 1
                next_sec = raw[i]
                combined = sec.text + "\n\n" + next_sec.text
                sec = _Section(
                    heading   = sec.heading,   # keep the original heading
                    h2_parent = sec.h2_parent,
                    text      = combined,
                )
            merged.append(sec)
            i += 1

        # ── Step 4: Emit chunks ───────────────────────────────────────
        chunks: list[Chunk] = []
        chunk_idx = 0
        for sec in merged:
            if sec.tokens <= self.chunk_size:
                # Section fits in one chunk
                chunks.append(Chunk(
                    text        = sec.text,
                    source      = source,
                    section     = sec.heading[:80],
                    headings    = [sec.h2_parent, sec.heading] if sec.h2_parent else [sec.heading],
                    chunk_idx   = chunk_idx,
                    token_count = sec.tokens,
                ))
                chunk_idx += 1
            else:
                # Section too large — fall back to sliding window
                sub_chunks = list(self._fallback._sliding_window(sec.text, source))
                for sc in sub_chunks:
                    chunks.append(Chunk(
                        text        = sc.text,
                        source      = source,
                        section     = sec.heading[:80],
                        headings    = [sec.h2_parent, sec.heading] if sec.h2_parent else [sec.heading],
                        chunk_idx   = chunk_idx,
                        token_count = sc.token_count,
                    ))
                    chunk_idx += 1

        return chunks


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
