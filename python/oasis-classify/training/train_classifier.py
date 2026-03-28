"""
training/train_classifier.py — Train a LogisticRegression classifier on gte-small embeddings.

The trained model is an optional upgrade path for classifier.py.
If training/models/classifier.pkl is present, classifier.py can load it as a
faster/more-accurate alternative to centroid cosine matching.

Usage:
    python training/train_classifier.py
    python training/train_classifier.py --data training/data/synthetic_queries.csv
    python training/train_classifier.py --output-dir training/models --c 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CLASSIFY_DIR = os.path.dirname(_SCRIPT_DIR)

DEFAULT_DATA = os.path.join(_SCRIPT_DIR, "data", "synthetic_queries.csv")
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "models")
CENTROIDS_PATH = os.path.join(_CLASSIFY_DIR, "data", "centroids.npy")
EMBEDDING_MODEL = "thenlper/gte-small"


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
        import sklearn  # noqa: F401
    except ImportError:
        missing.append("scikit-learn")
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
    """Load query,category CSV. Returns (queries, labels)."""
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
# Embedding
# ---------------------------------------------------------------------------

def _embed_queries(queries: list[str], model_name: str) -> "np.ndarray":
    """Embed all queries with gte-small. Returns (N, 384) float32 array."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print(f"Loading embedding model: {model_name}", file=sys.stderr)
    model = SentenceTransformer(model_name)

    print(f"Embedding {len(queries)} queries...", file=sys.stderr)
    embeddings = model.encode(
        queries,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    output_dir: str,
    c_param: float = 1.0,
    model_name: str = EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Train classifier and save artifacts. Returns eval metrics dict."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score

    # Check centroids artifact exists (ensures model is consistent)
    if not os.path.exists(CENTROIDS_PATH):
        print(
            "ERROR: data/centroids.npy not found. "
            "Run 'python build_centroids.py' before training.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load data
    print(f"Loading data from {data_path}", file=sys.stderr)
    queries, labels = _load_csv(data_path)
    print(f"Loaded {len(queries)} samples across {len(set(labels))} categories.", file=sys.stderr)

    if len(queries) == 0:
        print("ERROR: No data found. Run generate_data.py first.", file=sys.stderr)
        sys.exit(1)

    # Embed
    embeddings = _embed_queries(queries, model_name)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42, stratify=y
    )

    print(
        f"Train: {len(X_train)} samples, Test: {len(X_test)} samples",
        file=sys.stderr,
    )

    # Train LogisticRegression
    print(f"Training LogisticRegression (C={c_param}, max_iter=1000)...", file=sys.stderr)
    clf = LogisticRegression(
        max_iter=1000,
        C=c_param,
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    print(f"\nOverall accuracy: {accuracy:.4f}", file=sys.stderr)

    # Per-category metrics
    target_names = [str(le.classes_[i]) for i in range(len(le.classes_))]
    report_str = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    print("\nPer-category report:", file=sys.stderr)
    print(report_str, file=sys.stderr)

    # Parse classification report into dict
    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0, output_dict=True
    )

    # Save artifacts
    os.makedirs(output_dir, exist_ok=True)

    clf_path = os.path.join(output_dir, "classifier.pkl")
    le_path = os.path.join(output_dir, "label_encoder.pkl")
    report_path = os.path.join(output_dir, "eval_report.json")

    with open(clf_path, "wb") as fh:
        pickle.dump(clf, fh)
    print(f"Saved classifier to {clf_path}", file=sys.stderr)

    with open(le_path, "wb") as fh:
        pickle.dump(le, fh)
    print(f"Saved label encoder to {le_path}", file=sys.stderr)

    eval_report = {
        "accuracy": accuracy,
        "num_train": int(len(X_train)),
        "num_test": int(len(X_test)),
        "num_categories": int(len(le.classes_)),
        "c_param": c_param,
        "embedding_model": model_name,
        "per_category": report_dict,
    }

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(eval_report, fh, indent=2)
    print(f"Saved eval report to {report_path}", file=sys.stderr)

    return eval_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LogisticRegression classifier on gte-small embeddings."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"Path to training CSV (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save model artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Regularization parameter C for LogisticRegression (default: 1.0)",
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
            f"ERROR: Training data not found: {args.data}\n"
            "Run 'python training/generate_data.py' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    report = train(
        data_path=args.data,
        output_dir=args.output_dir,
        c_param=args.c,
        model_name=args.embedding_model,
    )

    print(f"\nTraining complete. Accuracy: {report['accuracy']:.4f}")
