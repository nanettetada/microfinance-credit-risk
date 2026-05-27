"""Pydantic request/response schemas for the credit risk API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HomeOwnership = Literal["RENT", "OWN", "MORTGAGE", "LODGER", "OTHER"]
Purpose = Literal[
    "SCHOOL_FEES",
    "DEBT_CONSOLIDATION",
    "MEDICAL",
    "AGRIC_INPUTS",
    "FUNERAL",
    "BUSINESS",
    "CAR",
    "HOME_IMPROVEMENT",
    "SOLAR_BACKUP",
    "OTHER",
]


class LoanApplication(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    loan_term_months: int = Field(..., ge=6, le=120)
    employment_years: int = Field(..., ge=0, le=60)
    credit_history_months: int = Field(..., ge=0, le=720)
    existing_loans: int = Field(..., ge=0, le=20)
    home_ownership: HomeOwnership
    purpose: Purpose


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., ge=0, le=1)
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]
    model_version: str


class BatchRequest(BaseModel):
    applications: list[LoanApplication] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    trained_at: str | None = None
    metrics: dict | None = None
    risk_bands: dict
