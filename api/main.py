"""FastAPI service exposing the trained credit risk model."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.predict import get_metadata, get_model, predict_many, predict_one

from .schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    LoanApplication,
    ModelInfoResponse,
    PredictionResponse,
)

app = FastAPI(
    title="Credit Risk API",
    description="Predict the probability a loan application will default.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    try:
        get_model()
        return HealthResponse(status="ok", model_loaded=True)
    except FileNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
def model_info() -> ModelInfoResponse:
    meta = get_metadata()
    return ModelInfoResponse(
        model_version=meta.get("model_version", "unknown"),
        trained_at=meta.get("trained_at"),
        metrics=meta.get("metrics"),
        risk_bands=meta.get("risk_bands", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(application: LoanApplication) -> PredictionResponse:
    try:
        result = predict_one(application.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
def predict_batch(req: BatchRequest) -> BatchResponse:
    try:
        results = predict_many([a.model_dump() for a in req.applications])
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return BatchResponse(predictions=[PredictionResponse(**r) for r in results])
