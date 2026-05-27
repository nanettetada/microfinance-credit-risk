"""Streamlit dashboard for the credit risk MLOps project.

Run with:
    streamlit run dashboard.py

Pairs nicely with the FastAPI service in `api/main.py` — the dashboard talks to
the saved model directly via `src.predict` rather than over HTTP, so it works
even when the API isn't running.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import (
    CATEGORICAL_FEATURES,
    HOME_OWNERSHIP,
    NUMERIC_FEATURES,
    PURPOSE,
    load_or_generate,
)
from src.predict import get_metadata, predict_many

st.set_page_config(page_title="Credit Risk Dashboard", page_icon=":bank:", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #2E86C1 0%, #1B4F72 100%);
        padding: 32px 28px;
        border-radius: 14px;
        color: white;
        margin: -10px 0 28px 0;
        box-shadow: 0 8px 24px rgba(46, 134, 193, 0.25);
    }
    .hero h1 { margin: 0 0 6px 0; font-size: 38px; font-weight: 700; letter-spacing: -0.5px; }
    .hero p  { margin: 0; font-size: 17px; opacity: 0.92; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        padding: 14px 18px;
        border-radius: 12px;
        border-left: 4px solid #2E86C1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        font-weight: 600;
    }
    </style>

    <div class="hero">
      <h1>:bank: Credit Risk — End-to-End Default Prediction</h1>
      <p>Built by Tadaishe Maumbe &middot; XGBoost &middot; FastAPI &middot; Docker &middot; Streamlit policy simulator</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading portfolio sample...")
def load_portfolio() -> pd.DataFrame:
    return load_or_generate(n=20_000)


@st.cache_data(show_spinner="Scoring portfolio...")
def score_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preds = predict_many(X.to_dict(orient="records"))
    df = df.copy()
    df["pd_score"] = [p["default_probability"] for p in preds]
    df["risk_band"] = [p["risk_band"] for p in preds]
    return df


# --------------------------------------------------------------------------- #
try:
    meta = get_metadata()
    model_loaded = True
except FileNotFoundError:
    meta = {"model_version": "n/a", "metrics": {}, "risk_bands": {}}
    model_loaded = False


if not model_loaded:
    st.error(
        "No trained model found at `models/model.joblib`. "
        "Run `python -m src.train` from the project root, then refresh."
    )
    st.stop()

df = load_portfolio()
df = score_portfolio(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Applications", f"{len(df):,}")
c2.metric("Observed default rate", f"{df['default'].mean()*100:.1f}%")
c3.metric("Model AUC (training)", f"{meta.get('metrics', {}).get('test_auc', 0):.3f}")
c4.metric("Model version", meta.get("model_version", "n/a"))


tab_score, tab_book, tab_policy = st.tabs(
    [":dart: Score an applicant", ":books: Book overview", ":scales: Policy simulator"]
)

# --------------------------------------------------------------------------- #
with tab_score:
    st.subheader("Score a new application")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 80, 35)
        income = st.number_input("Annual income ($)", 12_000, 500_000, 65_000, step=1000)
        employment = st.slider("Employment (years)", 0, 50, 5)
    with col2:
        loan_amount = st.number_input("Loan amount ($)", 1_000, 100_000, 15_000, step=500)
        loan_term = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60], index=2)
        credit_history = st.slider("Credit history (months)", 0, 600, 84)
    with col3:
        existing_loans = st.slider("Existing loans", 0, 10, 1)
        home = st.selectbox("Home ownership", HOME_OWNERSHIP)
        purpose = st.selectbox("Loan purpose", PURPOSE)

    record = {
        "age": age, "income": income, "loan_amount": loan_amount,
        "loan_term_months": loan_term, "employment_years": employment,
        "credit_history_months": credit_history, "existing_loans": existing_loans,
        "home_ownership": home, "purpose": purpose,
    }
    result = predict_many([record])[0]
    prob = result["default_probability"]

    st.markdown("### Risk score")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1F2937"},
            "steps": [
                {"range": [0, 20], "color": "#27AE60"},
                {"range": [20, 50], "color": "#F39C12"},
                {"range": [50, 100], "color": "#E74C3C"},
            ],
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    if result["risk_band"] == "LOW":
        st.success(f":white_check_mark: **Low risk** ({prob*100:.1f}%). Recommend approve.")
    elif result["risk_band"] == "MEDIUM":
        st.warning(f":warning: **Medium risk** ({prob*100:.1f}%). Manual review.")
    else:
        st.error(f":x: **High risk** ({prob*100:.1f}%). Recommend decline.")

    dti = loan_amount / max(income, 1)
    st.caption(f"Debt-to-income ratio for this application: **{dti:.2f}**")


# --------------------------------------------------------------------------- #
with tab_book:
    st.subheader("Risk profile of the book")
    col_a, col_b = st.columns(2)
    with col_a:
        band_counts = df["risk_band"].value_counts().reset_index()
        band_counts.columns = ["band", "count"]
        fig = px.pie(
            band_counts, names="band", values="count", hole=0.55,
            color="band",
            color_discrete_map={"LOW": "#27AE60", "MEDIUM": "#F39C12", "HIGH": "#E74C3C"},
        )
        fig.update_layout(title="Applications by risk band")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.histogram(df, x="pd_score", nbins=30, color="risk_band",
                            color_discrete_map={"LOW": "#27AE60", "MEDIUM": "#F39C12", "HIGH": "#E74C3C"},
                            labels={"pd_score": "Predicted default probability"})
        fig.update_layout(title="Score distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Default rate by predicted score decile")
    df["decile"] = pd.qcut(df["pd_score"], 10, labels=range(1, 11))
    decile_df = df.groupby("decile", observed=True)["default"].agg(["mean", "count"]).reset_index()
    decile_df.columns = ["decile", "default_rate", "applications"]
    fig = px.bar(
        decile_df, x="decile", y="default_rate", text=decile_df["default_rate"].map(lambda x: f"{x*100:.1f}%"),
        color="default_rate", color_continuous_scale="Reds",
    )
    fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    top_decile = decile_df.iloc[-1]
    overall_rate = df["default"].mean()
    lift = top_decile["default_rate"] / max(overall_rate, 1e-6)
    st.info(
        f":bulb: **Insight:** the top decile of model scores contains "
        f"applications that default at **{top_decile['default_rate']*100:.1f}%** — "
        f"**{lift:.1f}×** the overall rate. That's the lift you need for a risk-based "
        f"pricing strategy to be worth running."
    )


# --------------------------------------------------------------------------- #
with tab_policy:
    st.subheader("Policy simulator")
    threshold = st.slider("Auto-decline threshold (PD)", 0.0, 1.0, 0.50, step=0.01)

    auto_decline = df[df["pd_score"] >= threshold]
    approved = df[df["pd_score"] < threshold]
    n_declined = len(auto_decline)
    bad_declined = int(auto_decline["default"].sum())
    n_approved = len(approved)
    bad_approved = int(approved["default"].sum())
    overall_default = df["default"].mean()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Auto-declined", f"{n_declined:,}")
    a2.metric("Defaulters caught", f"{bad_declined:,}",
              help="Applications declined that would have defaulted")
    a3.metric("Approved", f"{n_approved:,}")
    a4.metric("Default rate in approved book",
              f"{(bad_approved/max(n_approved,1))*100:.1f}%",
              delta=f"{((bad_approved/max(n_approved,1)) - overall_default)*100:.1f} pp vs base",
              delta_color="inverse")

    st.markdown("### Trade-off across thresholds")
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        approved_t = df[df["pd_score"] < t]
        declined_t = df[df["pd_score"] >= t]
        rows.append({
            "threshold": t,
            "approval_rate": len(approved_t) / len(df),
            "approved_default_rate": approved_t["default"].mean() if len(approved_t) else 0,
            "caught_defaulters": declined_t["default"].sum() / max(df["default"].sum(), 1),
        })
    trade = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trade["threshold"], y=trade["approval_rate"],
                              name="Approval rate", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=trade["threshold"], y=trade["approved_default_rate"],
                              name="Default rate in approved book", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=trade["threshold"], y=trade["caught_defaulters"],
                              name="Defaulters caught (recall)", mode="lines+markers"))
    fig.update_layout(yaxis_tickformat=".0%", height=420,
                       xaxis_title="Decline threshold (PD)", yaxis_title="Rate")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        ":bulb: **Insight:** lowering the threshold catches more defaulters but rejects more good applicants. "
        "A practical choice depends on the cost of a bad loan vs. the revenue from a marginal one — "
        "the right number is a business decision, not a modelling one."
    )
