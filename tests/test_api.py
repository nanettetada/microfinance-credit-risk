from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID = {
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


def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_score():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_band"] in {"LOW", "MEDIUM", "HIGH"}


def test_predict_rejects_invalid_input():
    bad = {**VALID, "age": -5}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_unknown_purpose():
    bad = {**VALID, "purpose": "INTERSTELLAR_TRAVEL"}  # not in the Zim purpose list
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_batch_endpoint():
    r = client.post("/predict/batch", json={"applications": [VALID, VALID]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 2


def test_model_info_endpoint():
    r = client.get("/model/info")
    assert r.status_code == 200
    assert "model_version" in r.json()
