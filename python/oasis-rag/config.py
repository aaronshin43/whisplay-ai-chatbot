"""
config.py — O.A.S.I.S. RAG Phase 2

Central configuration for the RAG pipeline.
All tuneable hyperparameters live here — never hardcoded in logic files.
"""

from __future__ import annotations
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Base paths (relative to project root)
# Override via environment variables when needed.
# ─────────────────────────────────────────────────────────────
import os

_HERE = Path(__file__).parent                          # python/oasis-rag/
_ROOT = _HERE.parent.parent                            # project root

KNOWLEDGE_DIR: Path = Path(os.getenv("OASIS_KNOWLEDGE_DIR",  str(_ROOT / "data" / "knowledge")))
INDEX_DIR:     Path = Path(os.getenv("OASIS_INDEX_DIR",      str(_ROOT / "data" / "rag_index")))

# ─────────────────────────────────────────────────────────────
# Embedding model
# ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("OASIS_EMBED_MODEL", "thenlper/gte-small")
EMBEDDING_DIM:   int = 384   # gte-small output dimension

# ─────────────────────────────────────────────────────────────
# Chunking (must match values used during indexing)
# ─────────────────────────────────────────────────────────────
CHUNK_SIZE:    int = 500
CHUNK_OVERLAP: int = 50

# ─────────────────────────────────────────────────────────────
# Stage 1 — Lexical Pre-filtering
# ─────────────────────────────────────────────────────────────
LEXICAL_CANDIDATE_POOL: int = 50   # max candidates forwarded to Stage 2

# ─────────────────────────────────────────────────────────────
# Stage 2 — Hybrid Semantic Re-ranking
# hybrid_score = ALPHA * cosine_sim + (1-ALPHA) * lexical_score
# ─────────────────────────────────────────────────────────────
ALPHA:           float = 0.6   # semantic weight
SCORE_THRESHOLD:      float = 0.10  # min hybrid score to pass through
CONFIDENCE_THRESHOLD: float = 0.35  # below this → LOW_CONFIDENCE_PROMPT instead of full template
TOP_K:           int   = 1     # final chunks returned to LLM (was 2 — 3 gives richer context)
MAX_PER_SOURCE:  int   = 1     # max chunks from the same source document (was 1)

# ─────────────────────────────────────────────────────────────
# Stage 3 — Selective Context Compression
# ─────────────────────────────────────────────────────────────
COMPRESS_ENABLED:          bool  = False
COMPRESS_MIN_SENTENCES:    int   = 2      # never compress below this
COMPRESS_KEYWORD_WEIGHT:   float = 1.0   # score per keyword hit in sentence
COMPRESS_POSITION_DECAY:   float = 0.05  # small bonus for early sentences
COMPRESS_SENTENCE_THRESHOLD: float = 0.0  # sentences with score > this are kept
                                           # 0.0 = keep any sentence with ≥1 hit
# Target token budget after compression (relative to original)
COMPRESS_MIN_RATIO: float = 0.40   # keep at least 40% of original tokens
COMPRESS_MAX_RATIO: float = 0.70   # target ceiling (30-60% reduction)

# ─────────────────────────────────────────────────────────────
# Flask service
# ─────────────────────────────────────────────────────────────
SERVICE_HOST: str = "0.0.0.0"
SERVICE_PORT: int = 5001

# ─────────────────────────────────────────────────────────────
# Index artifact filenames (inside INDEX_DIR)
# ─────────────────────────────────────────────────────────────
FAISS_INDEX_FILE:   str = "chunks.faiss"
METADATA_FILE:      str = "metadata.json"
KEYWORD_MAP_FILE:   str = "keyword_map.json"
INDEX_VERSION_FILE: str = "version.txt"
