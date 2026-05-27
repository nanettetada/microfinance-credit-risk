"""Shared pytest fixtures — train a small model once per test session."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _train_small_model(tmp_path_factory):
    """Train and persist a small model into models/ if none exists."""
    cwd = Path.cwd()
    model_path = cwd / "models" / "model.joblib"
    if model_path.exists():
        yield
        return

    # Train a tiny model so tests don't depend on a pre-trained artefact.
    from src.data import load_or_generate
    from src.features import build_preprocessor
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    import joblib
    import json

    df = load_or_generate(n=2000)
    X = df.drop(columns=["default"])
    y = df["default"]
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            eval_metric="auc", random_state=42, n_jobs=1,
        )),
    ])
    pipe.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    (cwd / "models" / "metadata.json").write_text(json.dumps({
        "model_version": "test-1.0.0",
        "risk_bands": {"LOW": 0.20, "MEDIUM": 0.50, "HIGH": 1.01},
    }))
    yield
