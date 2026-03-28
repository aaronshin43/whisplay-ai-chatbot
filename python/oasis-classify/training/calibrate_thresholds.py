"""
training/calibrate_thresholds.py — Sweep CLASSIFY_THRESHOLD and OOD_FLOOR to find optimal values.

Computes per-category precision/recall, life-critical recall, OOD FPR/FNR,
triage rate, and overall accuracy for a grid of threshold combinations.

Usage:
    python training/calibrate_thresholds.py
    python training/calibrate_thresholds.py --data training/data/synthetic_queries.csv
    python training/calibrate_thresholds.py --embeddings training/data/embeddings.npy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CLASSIFY_DIR = os.path.dirname(_SCRIPT_DIR)

DEFAULT_DATA = os.path.join(_SCRIPT_DIR, "data", "synthetic_queries.csv")
DEFAULT_OUTPUT = os.path.join(_SCRIPT_DIR, "models", "threshold_sweep.json")
EMBEDDING_MODEL = "thenlper/gte-small"
CENTROIDS_PATH = os.path.join(_CLASSIFY_DIR, "data", "centroids.npy")

# Life-critical categories
LIFE_CRITICAL = ["cpr", "choking", "bleeding"]

# OOD category ID
OOD_CATEGORY = "out_of_domain"

# Threshold sweep ranges
CLASSIFY_THRESHOLDS = [round(0.50 + i * 0.05, 2) for i in range(7)]  # 0.50 to 0.80
OOD_FLOORS = [round(0.15 + i * 0.05, 2) for i in range(5)]            # 0.15 to 0.35


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _check_dependencies() -> None:
    missing = []
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        missing.append("sentence-transformers")

    if missing:
        print(
            f"ERROR: Missing required packages: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> tuple[list[str], list[str]]:
    """Load query,category CSV."""
    import csv

    queries: list[str] = []
    labels: list[str] = []

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            q = row.get("query", "").strip()
            c = row.get("category", "").strip()
            if q and c:
                queries.append(q)
                labels.append(c)

    return queries, labels


# ---------------------------------------------------------------------------
# Centroid loading and cosine similarity
# ---------------------------------------------------------------------------

def _load_centroids() -> "np.ndarray":
    import numpy as np

    if not os.path.exists(CENTROIDS_PATH):
        print(
            "ERROR: data/centroids.npy not found. "
            "Run 'python build_centroids.py' before calibrating.",
            file=sys.stderr,
        )
        sys.exit(1)
    return np.load(CENTROIDS_PATH)


def _cosine_scores(query_vecs: "np.ndarray", centroids: "np.ndarray") -> "np.ndarray":
    """Compute cosine similarity: (N, 384) x (33, 384) -> (N, 33)."""
    import numpy as np

    q_norm = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-10)
    c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    return q_norm @ c_norm.T  # (N, 33)


def _embed_queries(queries: list[str], model_name: str) -> "np.ndarray":
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print(f"Loading embedding model: {model_name}", file=sys.stderr)
    model = SentenceTransformer(model_name)
    print(f"Embedding {len(queries)} queries...", file=sys.stderr)
    emb = model.encode(queries, batch_size=64, show_progress_bar=True, normalize_embeddings=False)
    return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Load category IDs from categories.py
# ---------------------------------------------------------------------------

def _load_category_ids() -> list[str]:
    sys.path.insert(0, _CLASSIFY_DIR)
    from categories import CATEGORY_IDS
    return CATEGORY_IDS


# ---------------------------------------------------------------------------
# Classification at a given threshold pair
# ---------------------------------------------------------------------------

def _classify_batch(
    scores: "np.ndarray",
    category_ids: list[str],
    classify_threshold: float,
    ood_floor: float,
) -> list[tuple[str, str]]:
    """Classify each query given cosine scores.

    Returns list of (predicted_category_or_mode, route) where route is one of:
    'ood_floor', 'ood_cluster', 'triage', 'classifier_hit'
    """
    import numpy as np

    ood_idx = category_ids.index(OOD_CATEGORY)
    results: list[tuple[str, str]] = []

    for i in range(len(scores)):
        row = scores[i]
        best_idx = int(np.argmax(row))
        best_score = float(row[best_idx])
        best_cat = category_ids[best_idx]

        if best_score < ood_floor:
            results.append((OOD_CATEGORY, "ood_floor"))
        elif best_cat == OOD_CATEGORY:
            results.append((OOD_CATEGORY, "ood_cluster"))
        elif best_score >= classify_threshold:
            results.append((best_cat, "classifier_hit"))
        else:
            results.append((best_cat, "triage"))

    return results


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    results: list[tuple[str, str]],
    true_labels: list[str],
    classify_threshold: float,
    ood_floor: float,
) -> dict[str, Any]:
    """Compute metrics for a threshold combination."""
    n = len(results)
    assert n == len(true_labels)

    # Separate OOD from medical queries
    medical_mask = [lbl != OOD_CATEGORY for lbl in true_labels]
    ood_mask = [lbl == OOD_CATEGORY for lbl in true_labels]

    n_medical = sum(medical_mask)
    n_ood = sum(ood_mask)

    # Route distribution
    triage_count = sum(1 for _, route in results if route == "triage")
    classifier_hit_count = sum(1 for _, route in results if route == "classifier_hit")
    ood_count = sum(1 for pred, _ in results if pred == OOD_CATEGORY)

    triage_rate = triage_count / n if n > 0 else 0.0

    # Overall accuracy (only for classifier_hit routes where we predicted a category)
    correct = sum(
        1 for (pred, route), true in zip(results, true_labels)
        if route == "classifier_hit" and pred == true
    )
    total_classified = classifier_hit_count
    accuracy = correct / total_classified if total_classified > 0 else 0.0

    # Life-critical recall: fraction of life-critical queries that are NOT routed to OOD
    # (they may be triage or classifier_hit — both acceptable for safety)
    lc_recall: dict[str, float] = {}
    for cat in LIFE_CRITICAL:
        cat_mask = [lbl == cat for lbl in true_labels]
        cat_total = sum(cat_mask)
        if cat_total == 0:
            lc_recall[cat] = 1.0
            continue
        # Count how many are NOT sent to OOD
        not_ood = sum(
            1 for (pred, _), is_cat in zip(results, cat_mask)
            if is_cat and pred != OOD_CATEGORY
        )
        lc_recall[cat] = not_ood / cat_total

    min_lc_recall = min(lc_recall.values()) if lc_recall else 0.0

    # OOD FPR: medical queries classified as OOD
    ood_fp = sum(
        1 for (pred, _), is_medical in zip(results, medical_mask)
        if is_medical and pred == OOD_CATEGORY
    )
    ood_fpr = ood_fp / n_medical if n_medical > 0 else 0.0

    # OOD FNR: non-medical queries that reach triage (not OOD)
    ood_fn = sum(
        1 for (pred, route), is_ood in zip(results, ood_mask)
        if is_ood and pred != OOD_CATEGORY
    )
    ood_fnr = ood_fn / n_ood if n_ood > 0 else 0.0

    # Per-category precision and recall (for classifier_hit only)
    all_cats = list(set(true_labels) - {OOD_CATEGORY})
    per_category: dict[str, dict[str, float]] = {}
    for cat in sorted(all_cats):
        tp = sum(
            1 for (pred, route), true in zip(results, true_labels)
            if route == "classifier_hit" and pred == cat and true == cat
        )
        fp = sum(
            1 for (pred, route), true in zip(results, true_labels)
            if route == "classifier_hit" and pred == cat and true != cat
        )
        fn = sum(
            1 for (pred, route), true in zip(results, true_labels)
            if true == cat and (pred != cat or route != "classifier_hit")
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_category[cat] = {"precision": round(precision, 4), "recall": round(recall, 4)}

    return {
        "classify_threshold": classify_threshold,
        "ood_floor": ood_floor,
        "accuracy": round(accuracy, 4),
        "triage_rate": round(triage_rate, 4),
        "ood_fpr": round(ood_fpr, 4),
        "ood_fnr": round(ood_fnr, 4),
        "life_critical_recall": {k: round(v, 4) for k, v in lc_recall.items()},
        "min_life_critical_recall": round(min_lc_recall, 4),
        "triage_count": triage_count,
        "classifier_hit_count": classifier_hit_count,
        "ood_count": ood_count,
        "n_total": n,
        "per_category": per_category,
    }


# ---------------------------------------------------------------------------
# Best combination selector
# ---------------------------------------------------------------------------

def _select_best(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select best threshold combination: LC recall >= 0.95, then accuracy, then triage rate in [0.05, 0.15]."""
    candidates = [r for r in results if r["min_life_critical_recall"] >= 0.95]
    if not candidates:
        # Relax: just pick highest LC recall
        candidates = sorted(results, key=lambda r: r["min_life_critical_recall"], reverse=True)
        if not candidates:
            return None

    # Among candidates with triage in target range
    on_target = [r for r in candidates if 0.05 <= r["triage_rate"] <= 0.15]
    if on_target:
        candidates = on_target

    # Sort by accuracy desc
    candidates = sorted(candidates, key=lambda r: r["accuracy"], reverse=True)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    data_path: str,
    output_path: str,
    embeddings_path: str | None,
    model_name: str,
) -> None:
    import numpy as np

    category_ids = _load_category_ids()
    centroids = _load_centroids()

    print(f"Loading data from {data_path}", file=sys.stderr)
    queries, true_labels = _load_csv(data_path)
    print(f"Loaded {len(queries)} samples.", file=sys.stderr)

    if len(queries) == 0:
        print("ERROR: No data found. Run generate_data.py first.", file=sys.stderr)
        sys.exit(1)

    # Load or compute embeddings
    if embeddings_path and os.path.exists(embeddings_path):
        print(f"Loading pre-computed embeddings from {embeddings_path}", file=sys.stderr)
        embeddings = np.load(embeddings_path)
        if embeddings.shape[0] != len(queries):
            print(
                f"ERROR: Embeddings shape {embeddings.shape} does not match "
                f"data length {len(queries)}.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        embeddings = _embed_queries(queries, model_name)
        if embeddings_path:
            os.makedirs(os.path.dirname(os.path.abspath(embeddings_path)), exist_ok=True)
            np.save(embeddings_path, embeddings)
            print(f"Saved embeddings to {embeddings_path}", file=sys.stderr)

    # Compute cosine scores once against all centroids
    print("Computing cosine similarity scores...", file=sys.stderr)
    all_scores = _cosine_scores(embeddings, centroids)  # (N, 33)

    # Grid sweep
    all_results: list[dict[str, Any]] = []

    total_combos = len(CLASSIFY_THRESHOLDS) * len(OOD_FLOORS)
    print(f"\nRunning {total_combos} threshold combinations...", file=sys.stderr)

    for ct in CLASSIFY_THRESHOLDS:
        for oof in OOD_FLOORS:
            if oof >= ct:
                continue  # OOD floor must be below classify threshold
            batch_results = _classify_batch(all_scores, category_ids, ct, oof)
            metrics = _compute_metrics(batch_results, true_labels, ct, oof)
            all_results.append(metrics)

    # Sort by LC recall desc, then accuracy desc
    all_results.sort(key=lambda r: (r["min_life_critical_recall"], r["accuracy"]), reverse=True)

    # Print summary table
    print("\n" + "=" * 100, file=sys.stderr)
    print(
        f"{'CT':>6} {'OOD_F':>7} {'Acc':>7} {'Triage%':>8} {'LC_Recall':>10} "
        f"{'OOD_FPR':>9} {'OOD_FNR':>9}",
        file=sys.stderr,
    )
    print("-" * 100, file=sys.stderr)
    for r in all_results[:20]:  # top 20
        print(
            f"{r['classify_threshold']:>6.2f} "
            f"{r['ood_floor']:>7.2f} "
            f"{r['accuracy']:>7.4f} "
            f"{r['triage_rate']:>8.4f} "
            f"{r['min_life_critical_recall']:>10.4f} "
            f"{r['ood_fpr']:>9.4f} "
            f"{r['ood_fnr']:>9.4f}",
            file=sys.stderr,
        )
    print("=" * 100, file=sys.stderr)

    # Save full results
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nSaved full sweep results to {output_path}", file=sys.stderr)

    # Recommend best
    best = _select_best(all_results)
    if best:
        print("\n" + "=" * 60)
        print("RECOMMENDED THRESHOLD PAIR:")
        print(f"  CLASSIFY_THRESHOLD = {best['classify_threshold']}")
        print(f"  OOD_FLOOR          = {best['ood_floor']}")
        print(f"  Accuracy:          {best['accuracy']:.4f}")
        print(f"  Triage rate:       {best['triage_rate']:.4f}")
        print(f"  Min LC recall:     {best['min_life_critical_recall']:.4f}")
        print(f"  OOD FPR:           {best['ood_fpr']:.4f}")
        print(f"  OOD FNR:           {best['ood_fnr']:.4f}")
        print("=" * 60)

        lc = best["life_critical_recall"]
        print("\nLife-critical recall by category:")
        for cat in LIFE_CRITICAL:
            print(f"  {cat}: {lc.get(cat, 'N/A'):.4f}")
    else:
        print("\nWARNING: No combination met all targets. Review sweep results.", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep CLASSIFY_THRESHOLD and OOD_FLOOR to find optimal values."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"Labeled CSV path (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path for sweep results (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--embeddings",
        default=None,
        help="Path to pre-computed embeddings .npy file (skip re-embedding if provided)",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help=f"Sentence-transformers model name (default: {EMBEDDING_MODEL})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _check_dependencies()
    args = _parse_args()

    if not os.path.exists(args.data):
        print(
            f"ERROR: Data file not found: {args.data}\n"
            "Run 'python training/generate_data.py' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_sweep(
        data_path=args.data,
        output_path=args.output,
        embeddings_path=args.embeddings,
        model_name=args.embedding_model,
    )
