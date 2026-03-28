"""
config.py — All thresholds, paths, and model configuration for oasis-classify.
"""

from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------

CLASSIFY_THRESHOLD        = 0.65   # >= this -> manual; [OOD_FLOOR, this) -> triage
OOD_FLOOR                 = 0.30   # < this -> OOD response (not triage)

# ---------------------------------------------------------------------------
# Tier 0 config
# ---------------------------------------------------------------------------

TIER0_MAX_WORDS           = 3      # Keyword match only for queries <= this many words

# ---------------------------------------------------------------------------
# Multi-label config
# ---------------------------------------------------------------------------

MULTI_LABEL_RATIO         = 0.80   # Secondary included if score >= primary * this
MAX_CATEGORIES            = 2      # Maximum categories per query
MAX_PROMPT_TOKENS         = 400    # Hard ceiling — tiktoken cl100k_base, drop secondary if exceeded

# ---------------------------------------------------------------------------
# Triage hint config
# ---------------------------------------------------------------------------

TRIAGE_HINT_BOOST         = 0.05   # Cosine score boost when prev_triage_hint matches
TRIAGE_HINT_MIN_RELEVANCE = 0.20   # Skip boost if query cosine sim to hint < this
TRIAGE_HINT_TTL_SEC       = 60     # Hint expiry in seconds — managed by TypeScript layer

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

EMBEDDING_MODEL           = "thenlper/gte-small"

# ---------------------------------------------------------------------------
# Per-category threshold overrides (falls back to CLASSIFY_THRESHOLD)
# ---------------------------------------------------------------------------

CATEGORY_THRESHOLDS: dict[str, float] = {}

# ---------------------------------------------------------------------------
# Priority levels for multi-label conflict resolution
# ---------------------------------------------------------------------------

PRIORITY_CRITICAL = ["cpr", "choking", "bleeding"]
PRIORITY_URGENT   = ["anaphylaxis", "electric_shock", "poisoning", "drowning"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR             = os.path.join(_HERE, "data")
MANUALS_DIR          = os.path.join(DATA_DIR, "manuals")
CENTROIDS_PATH       = os.path.join(DATA_DIR, "centroids.npy")
PROTOTYPES_PATH      = os.path.join(DATA_DIR, "prototypes.json")
SHORT_QUERIES_PATH   = os.path.join(DATA_DIR, "short_queries.json")
SENTENCE_MATCHES_PATH = os.path.join(DATA_DIR, "sentence_matches.json")
ALSO_CHECK_PATH      = os.path.join(DATA_DIR, "also_check_summaries.json")

# ---------------------------------------------------------------------------
# Client-side threshold_path constants (synthesized on failure — never sent by server)
# ---------------------------------------------------------------------------

THRESHOLD_PATH_NETWORK_ERROR  = "network_error"
THRESHOLD_PATH_SERVICE_ERROR  = "service_error"
THRESHOLD_PATH_INVALID_SCHEMA = "invalid_schema"
