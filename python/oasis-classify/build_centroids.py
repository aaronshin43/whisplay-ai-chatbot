"""
build_centroids.py — Offline build script.

Embeds all prototype queries from data/prototypes.json with gte-small,
computes per-category centroid embeddings, and saves to data/centroids.npy.

Must be run after:
  - Adding or removing categories in categories.py
  - Editing data/prototypes.json
  - Changing EMBEDDING_MODEL in config.py

Usage:
    cd python/oasis-classify
    python build_centroids.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, PROTOTYPES_PATH, CENTROIDS_PATH
from categories import CATEGORY_IDS


def build_centroids() -> None:
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Loading prototypes from: {PROTOTYPES_PATH}")
    with open(PROTOTYPES_PATH, encoding="utf-8") as fh:
        prototypes: dict[str, list[str]] = json.load(fh)

    # Verify all category IDs are covered
    missing = [cat_id for cat_id in CATEGORY_IDS if cat_id not in prototypes]
    if missing:
        print(f"ERROR: Missing prototype entries for categories: {missing}", file=sys.stderr)
        sys.exit(1)

    extra = [k for k in prototypes if k not in CATEGORY_IDS]
    if extra:
        print(f"WARNING: Prototype keys not in CATEGORY_IDS (will be skipped): {extra}")

    centroids = np.zeros((len(CATEGORY_IDS), 384), dtype=np.float32)

    for idx, cat_id in enumerate(CATEGORY_IDS):
        queries = prototypes[cat_id]
        if not queries:
            print(f"WARNING: No prototypes for category '{cat_id}' — centroid will be zero vector")
            continue
        embeddings = model.encode(queries, normalize_embeddings=False, show_progress_bar=False)
        centroid = embeddings.mean(axis=0)
        centroids[idx] = centroid.astype(np.float32)
        print(f"  [{idx+1:2d}/{len(CATEGORY_IDS)}] {cat_id}: {len(queries)} prototypes")

    np.save(CENTROIDS_PATH, centroids)
    print(f"\nCentroids saved to: {CENTROIDS_PATH}")
    print(f"Shape: {centroids.shape}  dtype: {centroids.dtype}")


if __name__ == "__main__":
    build_centroids()
