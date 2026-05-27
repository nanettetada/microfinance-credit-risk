from src.predict import predict_many, predict_one

SAMPLE = {
    "age": 35,
    "income": 65000,
    "loan_amount": 15000,
    "loan_term_months": 36,
    "employment_years": 5,
    "credit_history_months": 84,
    "existing_loans": 1,
    "home_ownership": "OWN",
    "purpose": "DEBT_CONSOLIDATION",
}


def test_predict_one_returns_valid_shape():
    out = predict_one(SAMPLE)
    assert set(out) == {"default_probability", "risk_band", "model_version"}
    assert 0.0 <= out["default_probability"] <= 1.0
    assert out["risk_band"] in {"LOW", "MEDIUM", "HIGH"}


def test_predict_many_matches_length():
    out = predict_many([SAMPLE, SAMPLE, SAMPLE])
    assert len(out) == 3
    for o in out:
        assert 0.0 <= o["default_probability"] <= 1.0


def test_risky_application_scores_higher_than_safe():
    safe = SAMPLE
    risky = {
        **SAMPLE,
        "income": 18000,
        "loan_amount": 40000,
        "employment_years": 0,
        "existing_loans": 5,
        "home_ownership": "LODGER",
        "purpose": "FUNERAL",
    }
    p_safe = predict_one(safe)["default_probability"]
    p_risky = predict_one(risky)["default_probability"]
    assert p_risky > p_safe
