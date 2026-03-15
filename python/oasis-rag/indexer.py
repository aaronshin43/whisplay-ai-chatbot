"""
indexer.py — O.A.S.I.S. RAG Phase 2

Document Indexing Pipeline.

Reads every Markdown/txt file from data/knowledge/, chunks them with
DocumentChunker, embeds each chunk with gte-small via sentence-transformers,
stores the vectors in a FAISS IndexFlatIP, and builds a keyword→[chunk_ids]
inverted index for Stage 1 lexical pre-filtering.

Artifacts written to data/rag_index/:
  chunks.faiss       — FAISS inner-product index (L2-normalised vectors)
  metadata.json      — chunk metadata list (text, source, section, headings)
  keyword_map.json   — { keyword: [chunk_id, ...] }
  version.txt        — ISO timestamp of last index build

Usage:
    python indexer.py                    # index data/knowledge/
    python indexer.py path/to/knowledge  # custom knowledge dir
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from document_chunker import DocumentChunker
from medical_keywords import detect_keywords

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _build_keyword_map(chunks: list[dict]) -> dict[str, list[int]]:
    """
    Build an inverted index: keyword → list of chunk_ids that contain it.
    Uses medical_keywords.detect_keywords for taxonomy-aware detection.
    """
    kmap: dict[str, list[int]] = {}
    for cid, chunk in enumerate(chunks):
        pairs = detect_keywords(chunk["text"])
        for keyword, _category in pairs:
            kmap.setdefault(keyword, []).append(cid)
    return kmap


def _embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Embed *texts* and return an L2-normalised float32 matrix of shape
    (len(texts), EMBEDDING_DIM).  Normalisation makes IndexFlatIP equivalent
    to cosine similarity.
    """
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,   # L2-normalise → cosine via inner product
        convert_to_numpy=True,
    ).astype("float32")
    return vectors


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

class Indexer:
    """
    Stateful indexing pipeline.

    Parameters
    ----------
    knowledge_dir : directory containing Markdown/txt source files
    index_dir     : output directory for FAISS + metadata artifacts
    model_name    : sentence-transformers model identifier
    """

    def __init__(
        self,
        knowledge_dir: str | Path = config.KNOWLEDGE_DIR,
        index_dir:     str | Path = config.INDEX_DIR,
        model_name:    str        = config.EMBEDDING_MODEL,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.index_dir     = Path(index_dir)
        self.model_name    = model_name

        self._model:       SentenceTransformer | None = None
        self._chunker:     DocumentChunker            = DocumentChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

    # ------------------------------------------------------------------
    # Lazy model loader — avoids loading on import
    # ------------------------------------------------------------------
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            log.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build(self) -> dict:
        """
        Run the full indexing pipeline.

        Returns
        -------
        dict with keys: chunk_count, index_dir, duration_s
        """
        t0 = time.perf_counter()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Chunk ──────────────────────────────────────────────────
        log.info("Chunking documents from: %s", self.knowledge_dir)
        raw_chunks = self._chunker.load_and_chunk(self.knowledge_dir)
        if not raw_chunks:
            raise RuntimeError(f"No documents found in {self.knowledge_dir}")
        log.info("  %d chunks created", len(raw_chunks))

        # ── 2. Build metadata list ────────────────────────────────────
        metadata: list[dict] = [c.to_dict() for c in raw_chunks]

        # ── 3. Embed ──────────────────────────────────────────────────
        log.info("Embedding %d chunks with %s …", len(metadata), self.model_name)
        texts   = [m["text"] for m in metadata]
        vectors = _embed(self.model, texts)    # shape: (N, 384)
        log.info("  Embedding done. Shape: %s", vectors.shape)

        # ── 4. Build FAISS index ──────────────────────────────────────
        index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        index.add(vectors)
        log.info("  FAISS index has %d vectors", index.ntotal)

        # ── 5. Build keyword inverted index ───────────────────────────
        keyword_map = _build_keyword_map(metadata)
        log.info("  Keyword map: %d unique terms", len(keyword_map))

        # ── 6. Persist ────────────────────────────────────────────────
        self._save(index, metadata, keyword_map)

        duration = time.perf_counter() - t0
        log.info("Index built in %.2f s  →  %s", duration, self.index_dir)
        return {
            "chunk_count": len(metadata),
            "index_dir":   str(self.index_dir),
            "duration_s":  round(duration, 3),
        }

    def _save(
        self,
        index:       faiss.IndexFlatIP,
        metadata:    list[dict],
        keyword_map: dict[str, list[int]],
    ) -> None:
        faiss.write_index(index, str(self.index_dir / config.FAISS_INDEX_FILE))

        with open(self.index_dir / config.METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        with open(self.index_dir / config.KEYWORD_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(keyword_map, f, ensure_ascii=False)

        with open(self.index_dir / config.INDEX_VERSION_FILE, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())

        log.info("Artifacts written to %s", self.index_dir)


# ─────────────────────────────────────────────────────────────
# Shared loader (used by retriever.py and service.py)
# ─────────────────────────────────────────────────────────────

class IndexStore:
    """
    Lightweight struct that holds the loaded FAISS index + metadata.
    Retriever uses this instead of loading files directly.
    """

    def __init__(
        self,
        faiss_index:  faiss.Index,
        metadata:     list[dict],
        keyword_map:  dict[str, list[int]],
        model:        SentenceTransformer,
    ) -> None:
        self.faiss_index = faiss_index
        self.metadata    = metadata
        self.keyword_map = keyword_map
        self.model       = model

    @property
    def chunk_count(self) -> int:
        return len(self.metadata)


def load_index(
    index_dir:  str | Path = config.INDEX_DIR,
    model_name: str        = config.EMBEDDING_MODEL,
) -> IndexStore:
    """
    Load a previously-built index from *index_dir* and return an IndexStore.
    Raises FileNotFoundError if the index has not been built yet.
    """
    index_dir = Path(index_dir)
    faiss_path    = index_dir / config.FAISS_INDEX_FILE
    metadata_path = index_dir / config.METADATA_FILE
    kmap_path     = index_dir / config.KEYWORD_MAP_FILE

    for p in (faiss_path, metadata_path, kmap_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Index artifact not found: {p}\n"
                "Run `python indexer.py` to build the index first."
            )

    faiss_index = faiss.read_index(str(faiss_path))
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    with open(kmap_path, encoding="utf-8") as f:
        keyword_map = json.load(f)

    log.info(
        "Index loaded: %d chunks, %d keywords",
        len(metadata), len(keyword_map),
    )

    model = SentenceTransformer(model_name)
    return IndexStore(faiss_index, metadata, keyword_map, model)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    knowledge_dir = sys.argv[1] if len(sys.argv) > 1 else str(config.KNOWLEDGE_DIR)
    indexer = Indexer(knowledge_dir=knowledge_dir)
    result  = indexer.build()
    print(f"\n[Indexer] Done: {result['chunk_count']} chunks indexed in {result['duration_s']}s")
    print(f"          Artifacts: {result['index_dir']}")
