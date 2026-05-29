<div align="center">

# The Loan Desk

**A credit decision tool for Zimbabwean microfinance.**
Fill in a loan application, get a verdict, the reasons behind it in plain English, and a recommended offer — instantly.

<a href="https://microfinance-credit-risk.streamlit.app"><img src="https://img.shields.io/badge/▶%20Try%20it%20live-The%20Loan%20Desk-16794C?style=for-the-badge&logo=streamlit&logoColor=white" /></a>

<sub>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/XGBoost-006400?style=flat-square&logo=xgboost&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" />
<img src="https://img.shields.io/badge/Pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white" />
</sub>

<br/>

<img src="docs/screenshots/01_decision.png" alt="The Loan Desk — the decision screen" width="90%"/>

</div>

---

## What this is

A loan officer opens one screen, types in an applicant, and immediately sees whether to **approve**, **take a closer look**, or **decline** — along with *why*, and what to actually offer. No accuracy printout, no wall of charts. A decision they can defend to the borrower sitting across the desk.

Behind that screen is the full path from data to a running service: a leakage-proof XGBoost pipeline, a FastAPI endpoint, a test suite, and a Docker image. I built it to find out what it really takes to turn a model into something a lending team could run — not just a notebook that ends with a number.

## Built for how lending actually works in Zimbabwe

Generic credit datasets don't reflect the people a Zim microfinance officer sees. So the data is shaped around them:

- **School fees** — the three-term cycle every parent budgets around
- **Agricultural inputs** for smallholder farmers
- **Funeral assistance**, **medical**, **business capital**, **solar backup** for load-shedding
- **LODGER** as a home-ownership status — far more common here than a US-style mortgage

Offers are framed in **USD with ZiG context**, and the book-level view includes cuts that matter locally: group vs. individual lending and repayment channel. The model is calibrated against those realities, not a textbook population.

## Results

Measured on a held-out test set:

| Metric | Score | What it means |
|---|---|---|
| **ROC-AUC** | **0.79** | Tells a good loan from a bad one ~79% of the time |
| **KS statistic** | **0.45** | Clear separation between defaulters and non-defaulters |
| **Top-decile lift** | **~8×** | The riskiest 10% of scores default about 8× more than the book average |

> **Honesty note:** the dataset is *synthetic*, generated to mirror Zimbabwean microfinance lending. The engineering — the pipeline, serving, tests, and the decision tool — is real and production-shaped. Swapping in a real dataset (LendingClub / Home Credit) is the first item on the roadmap.

## The three screens

<table>
  <tr>
    <td width="50%" valign="top">
      <p align="center"><b>Make a decision</b><br/><sub>A verdict, the reasons for and against, and a recommended offer — updating live as you type</sub></p>
      <img src="docs/screenshots/01_decision.png" alt="Make a decision" width="100%"/>
    </td>
    <td width="50%" valign="top">
      <p align="center"><b>The whole book</b><br/><sub>Score distribution, the lift chart, default rates by loan purpose, and the Zim-specific cuts</sub></p>
      <img src="docs/screenshots/02_book.png" alt="The whole book" width="100%"/>
    </td>
  </tr>
  <tr>
    <td colspan="2" valign="top">
      <p align="center"><b>Set the rules</b><br/><sub>Drag the decline threshold and watch approval rate, default rate, bad loans caught, and margin move together</sub></p>
      <img src="docs/screenshots/03_rules.png" alt="Set the rules" width="100%"/>
    </td>
  </tr>
</table>

The dashboard reads the model directly through `src.predict`, so it works whether or not the FastAPI service is running.

## How it fits together

```
       Raw / synthetic data
                 |
                 v
       src/data.py        load + validate + train/val/test split
                 |
                 v
       src/features.py    ColumnTransformer pipeline (leakage-proof)
                 |
                 v
       src/train.py       XGBoost (+ optional MLflow tracking) -> models/*.joblib
                 |
       +---------+---------+
       |                   |
       v                   v
   api/main.py        dashboard.py
   FastAPI            Streamlit ("The Loan Desk")
       |                   |
       v                   |
   Docker container        |
       |                   |
       +---------+---------+
                 |
                 v
            Loan officers
```

## Run it yourself

```bash
# Install
pip install -r requirements.txt

# Train the model (writes models/model.joblib + metadata.json)
python -m src.train

# Open The Loan Desk
streamlit run dashboard.py

# Or serve the model as an API
uvicorn api.main:app --reload
```

## Hit the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "income": 65000, "loan_amount": 15000, "loan_term_months": 36, "employment_years": 5, "credit_history_months": 84, "existing_loans": 1, "home_ownership": "OWN", "purpose": "BUSINESS"}'
```

```json
{
  "default_probability": 0.087,
  "risk_band": "LOW",
  "model_version": "1.0.0"
}
```

Endpoints: `/predict`, `/predict/batch`, `/health`, `/model/info`. Pydantic v2 validates every request, so bad input gets a clear 422 instead of a 500. Swagger UI lives at `/docs`.

## Run with Docker

```bash
docker-compose up
```

## Tests

```bash
pytest -v
```

12 tests covering the parts that actually break:
- the data generator produces valid shapes and types
- train/val/test splits don't overlap
- the preprocessor survives unseen categories at inference
- the model loads and scores a single applicant and a batch
- a risky application really does score higher than a safe one
- the API returns 200 for valid input and 422 for invalid input

## Project layout

```
microfinance-credit-risk/
├── dashboard.py            The Loan Desk (Streamlit)
├── src/
│   ├── data.py             data contract + synthetic generator
│   ├── features.py         leakage-proof preprocessing
│   ├── train.py            XGBoost training + metrics
│   └── predict.py          load model, score applicants
├── api/
│   ├── main.py             FastAPI app
│   └── schemas.py          Pydantic request/response models
├── tests/                  pytest suite
├── models/                 trained pipeline + metadata (generated)
├── Dockerfile · docker-compose.yml · Makefile
└── requirements.txt · requirements-dev.txt
```

## What I'd add next

- Swap the synthetic dataset for a real one (LendingClub or Home Credit) and re-tune.
- Add probability calibration (Platt or isotonic) and re-check the Brier score.
- A drift monitor comparing live feature distributions against training.

---

<div align="center">
<sub>Built by <b>Tadaishe Maumbe</b> · <a href="https://github.com/nanettetada">@nanettetada</a></sub>
</div>
