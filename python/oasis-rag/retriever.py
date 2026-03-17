"""
retriever.py — O.A.S.I.S. RAG Phase 2

3-Stage Hybrid Retrieval Pipeline.

Stage 1 — Lexical Pre-filtering
    Detect medical keywords in the query, look them up in the inverted
    keyword_map, and narrow the search space to at most LEXICAL_CANDIDATE_POOL
    candidates.  Falls back to all chunks when no keywords match.

Stage 2 — Semantic Re-ranking
    Embed the query with gte-small.  For each Stage-1 candidate compute:
        hybrid_score = ALPHA * cosine_similarity + (1-ALPHA) * lexical_score
    Keep top TOP_K chunks that exceed SCORE_THRESHOLD.

Stage 3 — Selective Context Compression
    Pass each top chunk through compressor.compress_chunk() to remove
    sentences not relevant to the query.  Target 20-40% token reduction.

Usage:
    from indexer   import load_index
    from retriever import Retriever

    store     = load_index()          # loads FAISS + metadata from disk
    retriever = Retriever(store)

    result = retriever.retrieve("patient is bleeding from the leg")
    # result.context   → compressed text ready for the LLM system prompt
    # result.chunks    → list[RetrievedChunk] for inspection/logging
    # result.latency_ms
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

import config
from compressor import compress_chunk
from indexer import IndexStore
from medical_keywords import detect_keywords, expand_query
from query_classifier import classify_query, _UPPER_EXTREMITY, _LOWER_EXTREMITY, _AXIAL

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:         int
    text:             str
    compressed_text:  str
    source:           str
    section:          str
    headings:         list[str]
    cosine_score:     float
    lexical_score:    float
    hybrid_score:     float


@dataclass
class RetrievalResult:
    context:      str                  # final text injected into LLM prompt
    chunks:       list[RetrievedChunk]
    query:        str
    latency_ms:   float
    stage1_count: int                  # candidates after Stage 1
    stage2_count: int                  # chunks that passed SCORE_THRESHOLD


# ─────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────

class Retriever:
    """
    3-Stage Hybrid Retriever.

    Parameters
    ----------
    store        : IndexStore loaded by indexer.load_index()
    alpha        : semantic weight in hybrid score (default from config)
    top_k        : final chunk count (default from config)
    threshold    : minimum hybrid score (default from config)
    candidate_k  : Stage-1 candidate pool size (default from config)
    compress     : enable Stage-3 compression (default from config)
    """

    def __init__(
        self,
        store:       IndexStore,
        alpha:       float = config.ALPHA,
        top_k:       int   = config.TOP_K,
        threshold:   float = config.SCORE_THRESHOLD,
        candidate_k: int   = config.LEXICAL_CANDIDATE_POOL,
        compress:    bool  = config.COMPRESS_ENABLED,
    ) -> None:
        self.store       = store
        self.alpha       = alpha
        self.top_k       = top_k
        self.threshold   = threshold
        self.candidate_k = candidate_k
        self.compress    = compress

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        compress: bool | None = None,
    ) -> RetrievalResult:
        """
        Retrieve context for *query*.

        Per-call *top_k* and *compress* override instance defaults without
        mutating shared state, making concurrent requests safe.
        """
        effective_top_k  = top_k    if top_k    is not None else self.top_k
        effective_compress = compress if compress is not None else self.compress

        t0 = time.perf_counter()

        # ── Stage 1: Lexical pre-filtering ────────────────────────────
        candidates, query_terms = self._stage1_lexical(query)
        stage1_count = len(candidates)
        log.debug("Stage 1: %d candidates", stage1_count)

        # ── Stage 2: Semantic re-ranking ──────────────────────────────
        ranked = self._stage2_semantic(query, candidates, query_terms)
        passing = [c for c in ranked if c["hybrid_score"] >= self.threshold]
        stage2_count = len(passing)

        # Source diversity: limit chunks from the same document to MAX_PER_SOURCE.
        # Prevents a single broad document (e.g. redcross_bone_joint.md) from
        # filling all top_k slots, while still allowing complementary chunks
        # from the same source (e.g. two who_bec_module5 chunks that together
        # provide complete context).
        source_count: dict[str, int] = {}
        diverse: list[dict] = []
        for c in passing:
            src = self.store.metadata[c["chunk_id"]].get("source", "")
            if source_count.get(src, 0) < config.MAX_PER_SOURCE:
                source_count[src] = source_count.get(src, 0) + 1
                diverse.append(c)
            if len(diverse) >= effective_top_k:
                break
        top = diverse
        log.debug("Stage 2: %d above threshold, %d after source-diversity, taking top %d",
                  stage2_count, len(diverse), len(top))

        # ── Stage 3: Context compression ──────────────────────────────
        retrieved: list[RetrievedChunk] = []
        for c in top:
            meta = self.store.metadata[c["chunk_id"]]
            if effective_compress:
                compressed = compress_chunk(
                    chunk_text=meta["text"],
                    query=query,
                    section=meta.get("section", ""),
                )
            else:
                compressed = meta["text"]

            retrieved.append(RetrievedChunk(
                chunk_id        = c["chunk_id"],
                text            = meta["text"],
                compressed_text = compressed,
                source          = meta.get("source", ""),
                section         = meta.get("section", ""),
                headings        = meta.get("headings", []),
                cosine_score    = c["cosine_score"],
                lexical_score   = c["lexical_score"],
                hybrid_score    = c["hybrid_score"],
            ))

        context = self._build_context(retrieved)
        latency_ms = (time.perf_counter() - t0) * 1000

        log.info(
            "Retrieve done: query=%r  stage1=%d  stage2=%d  top=%d  %.1f ms",
            query[:60], stage1_count, stage2_count, len(retrieved), latency_ms,
        )

        return RetrievalResult(
            context      = context,
            chunks       = retrieved,
            query        = query,
            latency_ms   = round(latency_ms, 2),
            stage1_count = stage1_count,
            stage2_count = stage2_count,
        )

    # ------------------------------------------------------------------
    # Stage 1 — Lexical pre-filtering
    # ------------------------------------------------------------------

    def _query_lexical_score(self, chunk_text: str, query_terms: frozenset[str]) -> float:
        """
        Query-specific lexical score: fraction of query terms found in chunk.
        Capped at 1.0.  Unlike the global keyword_score, this is query-aware
        so it actually discriminates between chunks for a given query.
        """
        if not query_terms:
            return 0.0
        text_lower = chunk_text.lower()
        hits = sum(1 for t in query_terms if t in text_lower)
        return min(hits / len(query_terms), 1.0)

    def _stage1_lexical(self, query: str) -> tuple[list[int], frozenset[str]]:
        """
        Return (candidate_chunk_ids, query_terms) for Stage 2.

        Strategy:
        1. Detect medical keywords in the query + expand to category terms.
        2. Union chunk IDs from keyword_map for all matched terms.
        3. Score by query-specific lexical overlap.
        4. Return top CANDIDATE_K, or full corpus fallback.
        """
        kmap    = self.store.keyword_map
        n_total = self.store.chunk_count

        detected   = [kw for kw, _ in detect_keywords(query)]
        expanded   = expand_query(query)
        all_terms  = list(dict.fromkeys(detected + [t.lower() for t in expanded]))
        query_terms = frozenset(all_terms)

        candidate_set: set[int] = set()
        for term in all_terms:
            for cid in kmap.get(term, []):
                candidate_set.add(cid)

        if not candidate_set:
            log.debug("Stage 1: no keyword match -- using full corpus (%d)", n_total)
            return list(range(n_total)), query_terms

        candidates = sorted(
            candidate_set,
            key=lambda cid: self._query_lexical_score(
                self.store.metadata[cid]["text"], query_terms
            ),
            reverse=True,
        )
        return candidates[: self.candidate_k], query_terms

    # ------------------------------------------------------------------
    # Stage 2 — Semantic re-ranking
    # ------------------------------------------------------------------

    def _stage2_semantic(
        self,
        query:       str,
        candidates:  list[int],
        query_terms: frozenset[str],
    ) -> list[dict]:
        """
        Compute hybrid scores for each candidate chunk.

        Returns list[dict] sorted by hybrid_score descending.
        """
        if not candidates:
            return []

        # Embed query (L2-normalised → inner product = cosine)
        q_vec = self.store.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")   # shape: (1, 384)

        # Retrieve all candidate vectors from FAISS at once
        # FAISS reconstruct is O(1) per vector for IndexFlatIP
        cand_vectors = np.array(
            [self.store.faiss_index.reconstruct(cid) for cid in candidates],
            dtype="float32",
        )   # shape: (|candidates|, 384)

        # Cosine similarities: (1,384) × (384,|candidates|) → (|candidates|,)
        cosine_sims = (q_vec @ cand_vectors.T).flatten()

        # Classify query for body-part filtering
        qc = classify_query(query)

        # Body-part mismatch penalty sets
        # When query is upper-extremity-only, penalise chest/leg chunks (and vice versa)
        _PENALTY = 0.15
        penalise_for_upper = qc.is_upper_extremity_only
        penalise_for_lower = qc.is_lower_extremity_only

        _CHEST_RIB_TERMS  = {"chest", "rib", "thorax", "sternum", "lung", "pleura"}
        _LEG_TERMS        = {"leg", "thigh", "femur", "tibia", "fibula", "shin", "calf"}
        _ARM_HAND_TERMS   = {"arm", "hand", "finger", "wrist", "elbow", "forearm"}

        # Query-specific lexical scores for each candidate
        results = []
        for i, cid in enumerate(candidates):
            cos    = float(cosine_sims[i])
            lex    = self._query_lexical_score(self.store.metadata[cid]["text"], query_terms)
            hybrid = self.alpha * cos + (1.0 - self.alpha) * lex

            # Apply body-part mismatch penalty
            chunk_text_lower = self.store.metadata[cid]["text"].lower()
            if penalise_for_upper:
                if any(t in chunk_text_lower for t in _CHEST_RIB_TERMS) or \
                   any(t in chunk_text_lower for t in _LEG_TERMS):
                    hybrid = max(0.0, hybrid - _PENALTY)
            elif penalise_for_lower:
                if any(t in chunk_text_lower for t in _ARM_HAND_TERMS):
                    hybrid = max(0.0, hybrid - _PENALTY)

            results.append({
                "chunk_id":     cid,
                "cosine_score": cos,
                "lexical_score": lex,
                "hybrid_score": hybrid,
            })

        results.sort(key=lambda r: r["hybrid_score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Merge compressed chunk texts into a single context string for the LLM.
        Separates chunks with a clear divider and includes source attribution.
        """
        if not chunks:
            return ""

        parts: list[str] = []
        for i, c in enumerate(chunks, start=1):
            source_tag = f"[Source {i}: {c.source}]"
            parts.append(f"{source_tag}\n{c.compressed_text}")

        return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────
# CLI smoke-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from indexer import load_index

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "patient has severe bleeding from the leg, how to apply tourniquet"

    print(f"[Retriever] Query: {query!r}\n")

    try:
        store     = load_index()
        retriever = Retriever(store)
        result    = retriever.retrieve(query)

        print(f"Stage 1 candidates : {result.stage1_count}")
        print(f"Stage 2 passing    : {result.stage2_count}")
        print(f"Top chunks         : {len(result.chunks)}")
        print(f"Latency            : {result.latency_ms} ms\n")

        print("=== CONTEXT (LLM input) ===")
        print(result.context)

        print("\n=== CHUNK SCORES ===")
        for c in result.chunks:
            print(
                f"  [{c.source}] cosine={c.cosine_score:.3f}  "
                f"lex={c.lexical_score:.3f}  hybrid={c.hybrid_score:.3f}"
            )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
