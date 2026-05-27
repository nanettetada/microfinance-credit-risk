"""Loaded-model prediction helper used by the API."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from . import __version__

MODEL_PATH = Path("models/model.joblib")
META_PATH = Path("models/metadata.json")


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run `python -m src.train` first."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_metadata() -> dict[str, Any]:
    if META_PATH.exists():
        with META_PATH.open() as f:
            return json.load(f)
    return {
        "model_version": __version__,
        "risk_bands": {"LOW": 0.20, "MEDIUM": 0.50, "HIGH": 1.01},
    }


def _to_band(prob: float, bands: dict[str, float]) -> str:
    for name in ("LOW", "MEDIUM", "HIGH"):
        if prob <= bands.get(name, 1.0):
            return name
    return "HIGH"


def predict_one(record: dict[str, Any]) -> dict[str, Any]:
    return predict_many([record])[0]


def predict_many(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model = get_model()
    meta = get_metadata()
    df = pd.DataFrame(records)
    probs = model.predict_proba(df)[:, 1]
    return [
        {
            "default_probability": round(float(p), 4),
            "risk_band": _to_band(float(p), meta["risk_bands"]),
            "model_version": meta.get("model_version", __version__),
        }
        for p in probs
    ]
