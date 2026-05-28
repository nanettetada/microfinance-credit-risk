"""Train a credit risk model and persist the pipeline + metadata.

Usage:
    python -m src.train
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import __version__
from .data import load_or_generate, train_val_test_split
from .features import build_preprocessor

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    # Catch *any* failure, not just ImportError: mlflow pulls in protobuf /
    # opentelemetry, which raise a TypeError (not ImportError) on newer Python
    # versions. Experiment tracking is optional, so degrade gracefully.
    MLFLOW_AVAILABLE = False


MODEL_PATH = Path("models/model.joblib")
META_PATH = Path("models/metadata.json")


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov statistic between positive and negative score distributions."""
    pos = np.sort(y_score[y_true == 1])
    neg = np.sort(y_score[y_true == 0])
    if pos.size == 0 or neg.size == 0:
        return 0.0
    thresholds = np.unique(np.concatenate([pos, neg]))
    tpr = np.searchsorted(pos, thresholds, side="right") / pos.size
    fpr = np.searchsorted(neg, thresholds, side="right") / neg.size
    return float(np.max(np.abs(tpr - fpr)))


def train() -> dict:
    print("Loading data...")
    df = load_or_generate()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pipeline = Pipeline([
        ("prep", build_preprocessor()),
        ("model", XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=pos_weight,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("Training...")
    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "val_auc": float(roc_auc_score(y_val, val_proba)),
        "test_auc": float(roc_auc_score(y_test, test_proba)),
        "test_brier": float(brier_score_loss(y_test, test_proba)),
        "test_ks": _ks_statistic(y_test.to_numpy(), test_proba),
    }

    metadata = {
        "model_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "risk_bands": {"LOW": 0.20, "MEDIUM": 0.50, "HIGH": 1.01},
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with META_PATH.open("w") as f:
        json.dump(metadata, f, indent=2)

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("credit-risk")
        with mlflow.start_run():
            mlflow.log_params({
                "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
                "scale_pos_weight": float(pos_weight),
            })
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(MODEL_PATH))
            mlflow.log_artifact(str(META_PATH))

    print("Done.")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  model -> {MODEL_PATH}")
    return metadata


if __name__ == "__main__":
    train()
