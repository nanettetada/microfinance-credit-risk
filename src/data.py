"""Synthetic credit risk data generation and loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES = [
    "age",
    "income",
    "loan_amount",
    "loan_term_months",
    "employment_years",
    "credit_history_months",
    "existing_loans",
]
CATEGORICAL_FEATURES = ["home_ownership", "purpose"]
TARGET = "default"

HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE", "LODGER", "OTHER"]
# Zimbabwean microfinance loan purposes — what people actually borrow for here:
# SCHOOL_FEES = three-term school fee cycle; AGRIC_INPUTS = smallholder seed/fertiliser;
# SOLAR_BACKUP = generator + solar (load-shedding response); FUNERAL = covered separately
# from medical because Zimbabwean lenders price funeral loans differently.
PURPOSE = [
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


def generate(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic credit risk dataset with realistic default signal."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n)
    income = rng.lognormal(mean=10.8, sigma=0.55, size=n).clip(12_000, 400_000).round(0)
    loan_amount = rng.lognormal(mean=9.0, sigma=0.65, size=n).clip(1_000, 100_000).round(0)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n, p=[0.10, 0.20, 0.35, 0.20, 0.15])
    employment_years = rng.integers(0, 35, n)
    credit_history_months = (age - 18).clip(min=0) * 12 - rng.integers(0, 60, n).clip(min=0)
    credit_history_months = credit_history_months.clip(min=0)
    existing_loans = rng.poisson(0.8, n).clip(0, 8)
    # Zim ownership mix — formal mortgages are rarer here than the US/EU,
    # lodgers (room-renting within someone else's house) are common.
    home_ownership = rng.choice(HOME_OWNERSHIP, n, p=[0.36, 0.27, 0.10, 0.22, 0.05])
    purpose = rng.choice(PURPOSE, n)

    debt_to_income = loan_amount / (income + 1)
    logit = (
        -2.3
        + 2.5 * debt_to_income
        + 0.4 * (existing_loans >= 3)
        - 0.05 * employment_years
        - 0.005 * credit_history_months
        + 0.4 * (home_ownership == "RENT")
        + 0.5 * (home_ownership == "LODGER")  # higher risk than rental tenants
        - 0.3 * (home_ownership == "OWN")
        + 0.5 * (purpose == "FUNERAL")         # urgent, often not repaid on schedule
        + 0.3 * (purpose == "AGRIC_INPUTS")    # rainfall-dependent
        + 0.2 * (purpose == "OTHER")
        - 0.2 * (purpose == "SOLAR_BACKUP")    # asset-backed, lower risk
        - 0.15 * ((age - 35) / 10.0)
    )
    prob = 1 / (1 + np.exp(-logit))
    default = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "employment_years": employment_years,
        "credit_history_months": credit_history_months,
        "existing_loans": existing_loans,
        "home_ownership": home_ownership,
        "purpose": purpose,
        TARGET: default,
    })


def load_or_generate(path: Path | str = "data/credit_risk.csv", n: int = 20_000) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate(n=n).to_csv(path, index=False)
    return pd.read_csv(path)


def train_val_test_split(df: pd.DataFrame, seed: int = 42):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=seed
    )  # ~70/15/15 overall
    return X_train, X_val, X_test, y_train, y_val, y_test
