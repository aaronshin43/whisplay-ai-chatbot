"""
service.py — O.A.S.I.S. RAG Phase 2

Flask HTTP Service (port 5001).

Wraps the Retriever and Indexer in a REST API so the Node.js ChatFlow
can call out to it without embedding Python in the Node process.

Endpoints
---------
GET  /health   — liveness + index status
POST /retrieve — query → compressed context
POST /index    — trigger a full re-index of data/knowledge/

Error responses always carry JSON { "error": "message" }.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request

import config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# Application state (loaded once at startup, replaced on /index)
# ─────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_store      = None   # IndexStore | None
_retriever  = None   # Retriever  | None
_index_error: str | None = None


def _load_index_safe() -> None:
    """Load index from disk into global state. Thread-safe."""
    global _store, _retriever, _index_error
    from indexer   import load_index
    from retriever import Retriever
    try:
        store = load_index()
        ret   = Retriever(store)
        with _state_lock:
            _store      = store
            _retriever  = ret
            _index_error = None
        log.info("Index loaded: %d chunks", store.chunk_count)
    except FileNotFoundError as e:
        with _state_lock:
            _index_error = str(e)
        log.warning("Index not found — call POST /index to build it.")
    except Exception as e:
        with _state_lock:
            _index_error = str(e)
        log.error("Failed to load index: %s", e)


def _run_index_build(knowledge_dir: str) -> dict:
    """Run Indexer.build() and reload into global state."""
    from indexer   import Indexer, load_index
    from retriever import Retriever

    indexer = Indexer(knowledge_dir=knowledge_dir)
    result  = indexer.build()

    store = load_index()
    ret   = Retriever(store)
    with _state_lock:
        global _store, _retriever, _index_error
        _store      = store
        _retriever  = ret
        _index_error = None

    return result


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Returns service liveness + index status.

    Response 200:
        {
          "status":      "ok",
          "index_ready": true,
          "chunk_count": 42,
          "model":       "thenlper/gte-small"
        }
    Response 503 (index not loaded):
        {
          "status":      "degraded",
          "index_ready": false,
          "error":       "Index artifact not found..."
        }
    """
    with _state_lock:
        ready = _store is not None
        count = _store.chunk_count if ready else 0
        err   = _index_error

    if ready:
        return jsonify({
            "status":      "ok",
            "index_ready": True,
            "chunk_count": count,
            "model":       config.EMBEDDING_MODEL,
        })
    return jsonify({
        "status":      "degraded",
        "index_ready": False,
        "error":       err or "Index not loaded",
    }), 503


@app.post("/retrieve")
def retrieve():
    """
    Retrieve compressed context for a user query.

    Request body (JSON):
        { "query": "patient is bleeding heavily from the leg" }

    Optional fields:
        "top_k"    : int   — override TOP_K for this request
        "compress" : bool  — override compression toggle

    Response 200:
        {
          "context":       "...",    // text ready for LLM system prompt
          "chunks": [
            {
              "source":        "severe_bleeding.md",
              "section":       "Step 5 — Tourniquet",
              "hybrid_score":  0.72,
              "cosine_score":  0.81,
              "lexical_score": 0.55,
              "compressed_text": "..."
            },
            ...
          ],
          "stage1_candidates": 12,
          "stage2_passing":    3,
          "latency_ms":        87.4
        }
    """
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Missing required field: query"}), 400

    with _state_lock:
        retriever = _retriever

    if retriever is None:
        return jsonify({
            "error": "Index not ready. Call POST /index first.",
            "detail": _index_error,
        }), 503

    # Per-request overrides
    if "top_k" in body:
        retriever.top_k = int(body["top_k"])
    if "compress" in body:
        retriever.compress = bool(body["compress"])

    try:
        result = retriever.retrieve(query)
    except Exception as e:
        log.exception("Retrieval error")
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "context": result.context,
        "chunks": [
            {
                "source":          c.source,
                "section":         c.section,
                "hybrid_score":    round(c.hybrid_score, 4),
                "cosine_score":    round(c.cosine_score, 4),
                "lexical_score":   round(c.lexical_score, 4),
                "compressed_text": c.compressed_text,
            }
            for c in result.chunks
        ],
        "stage1_candidates": result.stage1_count,
        "stage2_passing":    result.stage2_count,
        "latency_ms":        result.latency_ms,
    })


@app.post("/index")
def index():
    """
    Trigger a full re-index of the knowledge directory.

    Request body (JSON, all optional):
        { "knowledge_dir": "data/knowledge" }

    This call is synchronous and may take 30-120 s on first run
    (model download + embedding).  Run it once after adding new documents.

    Response 200:
        {
          "status":      "ok",
          "chunk_count": 42,
          "duration_s":  14.7,
          "index_dir":   "data/rag_index"
        }
    """
    body          = request.get_json(silent=True) or {}
    knowledge_dir = body.get("knowledge_dir", str(config.KNOWLEDGE_DIR))

    if not Path(knowledge_dir).is_dir():
        return jsonify({"error": f"knowledge_dir not found: {knowledge_dir}"}), 400

    log.info("POST /index  knowledge_dir=%s", knowledge_dir)
    try:
        result = _run_index_build(knowledge_dir)
    except Exception as e:
        log.exception("Indexing error")
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "status":      "ok",
        "chunk_count": result["chunk_count"],
        "duration_s":  result["duration_s"],
        "index_dir":   result["index_dir"],
    })


# ─────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────

def create_app() -> Flask:
    """Factory — load index in background thread so startup is instant."""
    thread = threading.Thread(target=_load_index_safe, daemon=True)
    thread.start()
    return app


if __name__ == "__main__":
    log.info("Starting O.A.S.I.S. RAG service on port %d …", config.SERVICE_PORT)
    _load_index_safe()   # blocking load before serving
    app.run(
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        debug=False,
        threaded=True,
    )
