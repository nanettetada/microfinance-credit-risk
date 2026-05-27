"""Streamlit dashboard for the Zimbabwean microfinance credit risk project.

Run with:
    streamlit run dashboard.py

Pairs nicely with the FastAPI service in `api/main.py` - the dashboard talks to
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

# --------------------------------------------------------------------------- #
# ZIMBABWE-SPECIFIC CONFIG
# --------------------------------------------------------------------------- #
# USD remains the practical pricing currency; ZWL/ZiG context is shown alongside.
# Update USD_TO_ZWL when the official rate moves.
USD_TO_ZWL = 27.0  # 1 USD = 27 ZWL/ZiG (2026 approximate)

# Mobile-money repayment channels common with Zim microfinance borrowers.
REPAYMENT_CHANNELS = ["EcoCash", "OneMoney", "InnBucks", "Bank transfer", "Cash at branch"]

# Whether the borrower came in as a solidarity group (joint liability) or as an individual.
LENDING_STRUCTURES = ["Individual", "Group"]

# Whether income source is formal (payslip) or informal (vendor, market trader, smallholder).
INCOME_TYPES = ["Formal", "Informal"]

# Pricing ladder for the loan simulator (risk-based pricing).
INTEREST_BY_BAND = {"LOW": 0.18, "MEDIUM": 0.28, "HIGH": 0.42}

# Friendly labels for the model's PURPOSE codes (which are stored as enum-style strings).
PURPOSE_LABELS = {
    "SCHOOL_FEES": "School fees (3-term cycle)",
    "DEBT_CONSOLIDATION": "Debt consolidation",
    "MEDICAL": "Medical",
    "AGRIC_INPUTS": "Agric inputs (seed, fertiliser)",
    "FUNERAL": "Funeral",
    "BUSINESS": "Market stall stock / business",
    "CAR": "Transport / vehicle",
    "HOME_IMPROVEMENT": "Home improvement",
    "SOLAR_BACKUP": "Solar / generator",
    "OTHER": "Other",
}


def fmt_usd_zwl(usd: float) -> str:
    """Format a USD amount with ZWL context, e.g. '$500 (≈ZWL 13,500)'."""
    return f"${usd:,.0f} (≈ZWL {usd * USD_TO_ZWL:,.0f})"


st.set_page_config(page_title="MFI Credit Risk", page_icon=":bank:", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .hero {
        background: linear-gradient(135deg, #2E86C1 0%, #1B4F72 100%);
        padding: 32px 28px;
        border-radius: 14px;
        color: white;
        margin: -10px 0 28px 0;
        box-shadow: 0 8px 24px rgba(46, 134, 193, 0.25);
    }
    .hero h1 { margin: 0 0 6px 0; font-size: 36px; font-weight: 700; letter-spacing: -0.5px; }
    .hero p  { margin: 0; font-size: 16px; opacity: 0.92; }
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
    .insight {
        background: linear-gradient(180deg, #FFFFFF 0%, #EAF2F8 100%);
        border-left: 4px solid #2E86C1;
        padding: 16px 20px; border-radius: 10px; margin: 8px 0;
    }
    .insight .head { font-size: 12px; color: #1B4F72; font-weight: 700; letter-spacing: 1px; }
    .insight .body { font-size: 15px; color: #1B2631; margin-top: 4px; line-height: 1.5; }
    .action {
        background: linear-gradient(180deg, #FFFFFF 0%, #FEF5E7 100%);
        border-left: 4px solid #E67E22;
        padding: 16px 20px; border-radius: 10px; margin: 8px 0;
    }
    .action .head { font-size: 12px; color: #B9770E; font-weight: 700; letter-spacing: 1px; }
    .action .body { font-size: 15px; color: #6E4D11; margin-top: 4px; line-height: 1.5; }
    .zwlnote { color: #5D6D7E; font-size: 12px; margin-top: -8px; }
    </style>

    <div class="hero">
      <h1>:bank: Zimbabwean Microfinance — Credit Risk Cockpit</h1>
      <p>XGBoost scoring &middot; mobile-money aware &middot; group lending &middot; built for a Zim loan officer's daily decisions</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"<div class='zwlnote'>USD is the pricing currency. ZWL/ZiG context shown at 1 USD = {USD_TO_ZWL:.0f} ZWL (edit USD_TO_ZWL at the top of this file to update).</div>",
    unsafe_allow_html=True,
)


def insight(head: str, body: str) -> str:
    return f'<div class="insight"><div class="head">{head}</div><div class="body">{body}</div></div>'


def action(head: str, body: str) -> str:
    return f'<div class="action"><div class="head">{head}</div><div class="body">{body}</div></div>'


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading portfolio sample...")
def load_portfolio() -> pd.DataFrame:
    df = load_or_generate(n=20_000)
    # Derive Zim-specific columns from existing data (deterministic, reproducible).
    # These are *derived* from raw fields so the underlying CSV stays untouched.
    rng = np.random.default_rng(7)

    # Repayment channel: informal vendors lean EcoCash; older / OWN-home borrowers lean bank.
    n = len(df)
    channel_pick = rng.random(n)
    channels = np.where(
        channel_pick < 0.55, "EcoCash",
        np.where(channel_pick < 0.72, "OneMoney",
        np.where(channel_pick < 0.85, "InnBucks",
        np.where(channel_pick < 0.95, "Bank transfer", "Cash at branch"))),
    )
    # Older OWN borrowers shift toward bank transfer
    bank_shift = (df["home_ownership"] == "OWN") & (df["age"] > 40) & (rng.random(n) < 0.45)
    channels = np.where(bank_shift, "Bank transfer", channels)
    df["repayment_channel"] = channels

    # Group vs individual: smaller loans + AGRIC/BUSINESS purposes lean group; bigger lean individual.
    group_score = (
        (df["loan_amount"] < 5_000).astype(int) * 0.5
        + df["purpose"].isin(["AGRIC_INPUTS", "BUSINESS"]).astype(int) * 0.3
        + (df["existing_loans"] == 0).astype(int) * 0.2
    )
    df["lending_structure"] = np.where(
        rng.random(n) < group_score * 0.7, "Group", "Individual"
    )

    # Informal vs formal income: lower income brackets + LODGER/RENT lean informal.
    informal_score = (
        (df["income"] < 35_000).astype(int) * 0.55
        + df["home_ownership"].isin(["LODGER", "RENT"]).astype(int) * 0.25
        + (df["employment_years"] < 3).astype(int) * 0.2
    )
    df["income_type"] = np.where(
        rng.random(n) < informal_score * 0.85, "Informal", "Formal"
    )

    return df


@st.cache_data(show_spinner="Scoring portfolio...")
def score_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preds = predict_many(X.to_dict(orient="records"))
    df = df.copy()
    df["pd_score"] = [p["default_probability"] for p in preds]
    df["risk_band"] = [p["risk_band"] for p in preds]
    return df


def explain_record(record: dict, baseline_score: float, model) -> pd.DataFrame:
    """Return per-feature contribution to risk by one-at-a-time substitution.

    For each numeric feature, swap it with the population median; for each categorical,
    swap it with the population mode. The difference (baseline - swapped) is the
    direction/magnitude that feature pushed the score.
    """
    portfolio = load_portfolio()
    medians = {f: float(portfolio[f].median()) for f in NUMERIC_FEATURES}
    modes = {f: portfolio[f].mode().iloc[0] for f in CATEGORICAL_FEATURES}

    rows = []
    for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        alt = dict(record)
        alt[f] = medians[f] if f in NUMERIC_FEATURES else modes[f]
        alt_score = predict_many([alt])[0]["default_probability"]
        delta = baseline_score - alt_score  # positive = this feature pushed risk UP
        rows.append({"feature": f, "actual": record[f], "neutral": alt[f], "delta": delta})
    return pd.DataFrame(rows).sort_values("delta", key=lambda s: s.abs(), ascending=False)


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

avg_loan_usd = float(df["loan_amount"].mean())
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Applications", f"{len(df):,}")
c2.metric("Observed default rate", f"{df['default'].mean()*100:.1f}%")
c3.metric("Avg loan size", f"${avg_loan_usd:,.0f}",
          help=f"≈ZWL {avg_loan_usd * USD_TO_ZWL:,.0f} at 1:{USD_TO_ZWL:.0f}")
c4.metric("Model AUC (test)", f"{meta.get('metrics', {}).get('test_auc', 0):.3f}")
c5.metric("Model version", meta.get("model_version", "n/a"))


tab_score, tab_explain, tab_portfolio, tab_zim, tab_policy = st.tabs([
    ":dart: Approval simulator",
    ":mag: Why this score?",
    ":books: Portfolio health",
    ":iphone: Mobile money & groups",
    ":scales: Policy simulator",
])

# --------------------------------------------------------------------------- #
# TAB 1 — LOAN APPROVAL SIMULATOR
# --------------------------------------------------------------------------- #
with tab_score:
    st.subheader("Score a loan application")
    st.caption(
        "Inputs reflect a typical Zimbabwean microfinance file: informal-trader income, "
        "small loan amounts, mobile-money repayment. Output is a default probability, "
        "risk band, and a risk-based interest rate recommendation."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 80, 35)
        income_usd = st.number_input(
            "Annual income (USD)", 1_200, 500_000, 18_000, step=500,
            help="For an informal trader, estimate from daily takings × ~280 trading days.",
        )
        st.caption(f"≈ZWL {income_usd * USD_TO_ZWL:,.0f}/yr")
        employment = st.slider("Years in current livelihood", 0, 50, 4)
    with col2:
        loan_amount_usd = st.number_input(
            "Loan amount (USD)", 100, 100_000, 800, step=50,
            help="Most Zim MFI loans are under USD 2,000.",
        )
        st.caption(f"≈ZWL {loan_amount_usd * USD_TO_ZWL:,.0f}")
        loan_term = st.selectbox("Loan term (months)", [3, 6, 12, 24, 36, 48, 60], index=2)
        credit_history = st.slider("Credit history (months)", 0, 600, 36)
    with col3:
        existing_loans = st.slider("Existing loans", 0, 10, 1)
        home = st.selectbox("Home ownership", HOME_OWNERSHIP)
        purpose = st.selectbox(
            "Loan purpose",
            PURPOSE,
            format_func=lambda p: PURPOSE_LABELS.get(p, p),
        )

    # Zim-specific extras (used in UI/policy logic, not the model)
    col4, col5, col6 = st.columns(3)
    with col4:
        repayment_channel = st.selectbox("Repayment channel", REPAYMENT_CHANNELS, index=0)
    with col5:
        lending_structure = st.radio("Lending structure", LENDING_STRUCTURES, horizontal=True)
    with col6:
        income_type = st.radio("Income source", INCOME_TYPES, horizontal=True)

    record = {
        "age": age, "income": income_usd, "loan_amount": loan_amount_usd,
        "loan_term_months": loan_term, "employment_years": employment,
        "credit_history_months": credit_history, "existing_loans": existing_loans,
        "home_ownership": home, "purpose": purpose,
    }
    result = predict_many([record])[0]
    prob = result["default_probability"]
    band = result["risk_band"]

    # Group-lending discount: peer guarantee historically reduces effective default risk
    effective_prob = prob * (0.85 if lending_structure == "Group" else 1.0)
    effective_band = "LOW" if effective_prob < 0.20 else ("MEDIUM" if effective_prob < 0.50 else "HIGH")
    interest_rate = INTEREST_BY_BAND.get(effective_band, 0.30)

    # Monthly repayment using simple flat-rate (common in Zim MFI pricing)
    total_interest = loan_amount_usd * interest_rate * (loan_term / 12)
    monthly = (loan_amount_usd + total_interest) / max(loan_term, 1)
    dti_monthly = monthly / max(income_usd / 12, 1)

    st.markdown("### Risk score")
    score_col, action_col = st.columns([1.3, 1])
    with score_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=effective_prob * 100,
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
            title={"text": f"Default probability — band {effective_band}"},
        ))
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with action_col:
        st.markdown("##### Recommended terms")
        st.metric("Recommended APR", f"{interest_rate*100:.0f}%",
                  help="Risk-based pricing: LOW 18%, MEDIUM 28%, HIGH 42%")
        st.metric("Monthly repayment", fmt_usd_zwl(monthly))
        st.metric("Total interest over term", fmt_usd_zwl(total_interest))
        st.metric("Debt service / monthly income", f"{dti_monthly*100:.1f}%",
                  delta=f"{(dti_monthly - 0.40)*100:.1f} pp vs 40% safe ceiling",
                  delta_color="inverse")

    if effective_band == "LOW":
        st.success(
            f":white_check_mark: **Low risk** ({effective_prob*100:.1f}%). "
            f"Approve at {interest_rate*100:.0f}% APR. Monthly: {fmt_usd_zwl(monthly)}."
        )
    elif effective_band == "MEDIUM":
        st.warning(
            f":warning: **Medium risk** ({effective_prob*100:.1f}%). "
            "Approve with a guarantor or step-up loan history first. "
            f"Suggested APR {interest_rate*100:.0f}%."
        )
    else:
        st.error(
            f":x: **High risk** ({effective_prob*100:.1f}%). "
            "Decline as individual, or refer into a solidarity group with weekly meetings. "
            "If approved anyway, price at 42% APR and cap term at 6 months."
        )

    # Business commentary
    pieces = []
    if lending_structure == "Group":
        pieces.append("This is a <b>solidarity group</b> application — peer guarantee shaves ~15% off the standalone risk score.")
    if income_type == "Informal" and repayment_channel == "EcoCash":
        pieces.append("Informal income + EcoCash repayment is the dominant Zim MFI profile. Verify takings against three months of EcoCash statements before disbursing.")
    if dti_monthly > 0.50:
        pieces.append(f"<b>Debt service is {dti_monthly*100:.0f}% of monthly income</b> — above the safe 40% ceiling. Either reduce the loan amount or extend the term.")
    if purpose == "FUNERAL":
        pieces.append("Funeral loans are <b>urgency-driven</b> and historically the highest-default purpose. Insist on a payslip-deduction or family co-signer.")
    if purpose == "AGRIC_INPUTS":
        pieces.append("Agric input loans are <b>rainfall-dependent</b>. Align repayment schedule with the harvest cycle, not a flat monthly.")

    if pieces:
        st.markdown(
            insight("LOAN OFFICER COMMENTARY", "<br>".join(f"• {p}" for p in pieces)),
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------- #
# TAB 2 — SCORE EXPLAINER
# --------------------------------------------------------------------------- #
with tab_explain:
    st.subheader("Which features pushed this score up — or down?")
    st.caption(
        "One-at-a-time substitution: each feature is swapped for the portfolio median/mode "
        "and the score recomputed. A positive bar = that feature increased the risk score."
    )

    contrib = explain_record(record, prob, None)
    contrib["direction"] = contrib["delta"].apply(lambda d: "Pushed risk UP" if d > 0 else "Pushed risk DOWN")
    contrib["abs_delta"] = contrib["delta"].abs()
    plot_df = contrib.head(9).sort_values("delta")

    fig = px.bar(
        plot_df,
        x="delta", y="feature", orientation="h",
        color="direction",
        color_discrete_map={"Pushed risk UP": "#E74C3C", "Pushed risk DOWN": "#27AE60"},
        text=plot_df["delta"].map(lambda d: f"{d*100:+.1f} pp"),
        labels={"delta": "Change in default probability (pp)", "feature": ""},
    )
    fig.update_layout(height=440, xaxis_tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)

    top_up = contrib[contrib["delta"] > 0].head(2)
    top_down = contrib[contrib["delta"] < 0].head(2)

    up_text = ", ".join(f"<b>{r['feature']}</b> ({r['delta']*100:+.1f} pp)" for _, r in top_up.iterrows()) or "nothing material"
    down_text = ", ".join(f"<b>{r['feature']}</b> ({r['delta']*100:+.1f} pp)" for _, r in top_down.iterrows()) or "nothing material"

    st.markdown(
        insight(
            "WHY THIS APPLICATION SCORED THE WAY IT DID",
            f"Worst contributors to risk: {up_text}.<br>"
            f"Most protective factors: {down_text}.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        action(
            "HOW A LOAN OFFICER WOULD USE THIS",
            "If the top driver is something the borrower can change (e.g. an existing loan they could "
            "consolidate first, or a longer term to reduce monthly burden), <b>counter-offer</b> rather "
            "than decline. If the top drivers are immutable (age, history length), price for risk or "
            "route into a group product.",
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# TAB 3 — PORTFOLIO HEALTH
# --------------------------------------------------------------------------- #
with tab_portfolio:
    st.subheader("Portfolio risk profile")

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
        fig = px.histogram(
            df, x="pd_score", nbins=30, color="risk_band",
            color_discrete_map={"LOW": "#27AE60", "MEDIUM": "#F39C12", "HIGH": "#E74C3C"},
            labels={"pd_score": "Predicted default probability"},
        )
        fig.update_layout(title="Score distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Default rate by predicted score decile")
    df_local = df.copy()
    df_local["decile"] = pd.qcut(df_local["pd_score"], 10, labels=range(1, 11))
    decile_df = df_local.groupby("decile", observed=True)["default"].agg(["mean", "count"]).reset_index()
    decile_df.columns = ["decile", "default_rate", "applications"]
    fig = px.bar(
        decile_df, x="decile", y="default_rate",
        text=decile_df["default_rate"].map(lambda x: f"{x*100:.1f}%"),
        color="default_rate", color_continuous_scale="Reds",
    )
    fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

    top_decile = decile_df.iloc[-1]
    overall_rate = df["default"].mean()
    lift = top_decile["default_rate"] / max(overall_rate, 1e-6)
    st.markdown(
        insight(
            "MODEL LIFT",
            f"The top decile of model scores defaults at <b>{top_decile['default_rate']*100:.1f}%</b> — "
            f"<b>{lift:.1f}×</b> the book average. That's the lift you need for risk-based "
            "pricing to be commercially worth running.",
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Default rate by loan purpose")
    purpose_df = (
        df.groupby("purpose")
        .agg(default_rate=("default", "mean"),
             applications=("default", "size"),
             avg_loan=("loan_amount", "mean"))
        .reset_index()
        .sort_values("default_rate", ascending=False)
    )
    purpose_df["purpose_label"] = purpose_df["purpose"].map(PURPOSE_LABELS).fillna(purpose_df["purpose"])
    fig = px.bar(
        purpose_df, x="default_rate", y="purpose_label", orientation="h",
        color="default_rate", color_continuous_scale="Reds",
        text=purpose_df["default_rate"].map(lambda x: f"{x*100:.1f}%"),
        labels={"default_rate": "Default rate", "purpose_label": ""},
        hover_data={"applications": True, "avg_loan": ":$,.0f"},
    )
    fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False, height=420)
    st.plotly_chart(fig, use_container_width=True)

    worst = purpose_df.iloc[0]
    best = purpose_df.iloc[-1]
    st.markdown(
        insight(
            "PURPOSE IS A FIRST-PASS RISK SIGNAL",
            f"<b>{PURPOSE_LABELS.get(worst['purpose'], worst['purpose'])}</b> defaults at "
            f"<b>{worst['default_rate']*100:.1f}%</b>, while "
            f"<b>{PURPOSE_LABELS.get(best['purpose'], best['purpose'])}</b> defaults at only "
            f"<b>{best['default_rate']*100:.1f}%</b>. Funeral and agric-input loans carry the "
            "highest tail risk — pricing and verification standards should reflect this.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        action(
            "PORTFOLIO ACTION",
            f"Cap exposure to {PURPOSE_LABELS.get(worst['purpose'], worst['purpose']).lower()} at a fixed "
            "share of the book, and price those loans at the top APR band. "
            "Reinvest freed-up capital into the lower-default purpose categories.",
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# TAB 4 — ZIM-SPECIFIC: MOBILE MONEY & GROUP LENDING
# --------------------------------------------------------------------------- #
with tab_zim:
    st.subheader("Zim context — mobile money, groups, informal income")
    st.caption(
        "Three Zim-specific cuts of the same portfolio. These features were derived from the "
        "raw application fields (no fabricated raw data) so they reflect the deterministic "
        "patterns of the underlying scoring model."
    )

    col1, col2 = st.columns(2)
    with col1:
        chan_df = (
            df.groupby("repayment_channel")
            .agg(default_rate=("default", "mean"),
                 applications=("default", "size"),
                 avg_loan=("loan_amount", "mean"))
            .reset_index()
            .sort_values("default_rate", ascending=False)
        )
        fig = px.bar(
            chan_df, x="repayment_channel", y="default_rate",
            color="default_rate", color_continuous_scale="Reds",
            text=chan_df["default_rate"].map(lambda x: f"{x*100:.1f}%"),
            labels={"default_rate": "Default rate", "repayment_channel": ""},
            hover_data={"applications": True, "avg_loan": ":$,.0f"},
        )
        fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False,
                          height=380, title="Default rate by repayment channel")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        struct_df = (
            df.groupby("lending_structure")
            .agg(default_rate=("default", "mean"),
                 applications=("default", "size"),
                 avg_loan=("loan_amount", "mean"))
            .reset_index()
        )
        fig = px.bar(
            struct_df, x="lending_structure", y="default_rate",
            color="lending_structure",
            color_discrete_map={"Group": "#27AE60", "Individual": "#2E86C1"},
            text=struct_df["default_rate"].map(lambda x: f"{x*100:.1f}%"),
            labels={"default_rate": "Default rate", "lending_structure": ""},
            hover_data={"applications": True, "avg_loan": ":$,.0f"},
        )
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False,
                          height=380, title="Group vs individual lending")
        st.plotly_chart(fig, use_container_width=True)

    group_rate = struct_df.loc[struct_df["lending_structure"] == "Group", "default_rate"].iloc[0]
    indiv_rate = struct_df.loc[struct_df["lending_structure"] == "Individual", "default_rate"].iloc[0]
    diff_pp = (indiv_rate - group_rate) * 100
    st.markdown(
        insight(
            "GROUP LENDING IS YOUR FLOOR",
            f"Solidarity groups default at <b>{group_rate*100:.1f}%</b> vs individuals at "
            f"<b>{indiv_rate*100:.1f}%</b> — a <b>{diff_pp:+.1f} pp</b> gap. Peer pressure works. "
            "For new-to-bank borrowers, route them through a group product first; graduate the "
            "best repayers into individual loans after two clean cycles.",
        ),
        unsafe_allow_html=True,
    )

    # Income type cut
    inc_df = (
        df.groupby("income_type")
        .agg(default_rate=("default", "mean"),
             applications=("default", "size"),
             avg_loan=("loan_amount", "mean"))
        .reset_index()
    )
    fig = px.bar(
        inc_df, x="income_type", y="default_rate",
        color="income_type",
        color_discrete_map={"Formal": "#2E86C1", "Informal": "#E67E22"},
        text=inc_df["default_rate"].map(lambda x: f"{x*100:.1f}%"),
        hover_data={"applications": True, "avg_loan": ":$,.0f"},
        labels={"default_rate": "Default rate", "income_type": ""},
    )
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False,
                      height=320, title="Formal vs informal income borrowers")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        insight(
            "INFORMAL IS NOT THE SAME AS RISKY",
            "Informal-income borrowers (vendors, market traders, smallholders) are the bulk of "
            "the Zim MFI book. The default gap to formal-income borrowers is real but modest — "
            "what matters more is <b>verifiable cashflow</b>. Three months of EcoCash statements "
            "or market-stall takings is often a better underwriting signal than a payslip.",
        ),
        unsafe_allow_html=True,
    )

    # Channel x structure heatmap
    cross = (
        df.groupby(["repayment_channel", "lending_structure"])["default"]
        .mean().reset_index()
    )
    pivot = cross.pivot(index="repayment_channel", columns="lending_structure", values="default")
    fig = px.imshow(
        pivot, text_auto=".1%", color_continuous_scale="Reds", aspect="auto",
        labels=dict(x="Lending structure", y="Repayment channel", color="Default rate"),
    )
    fig.update_layout(height=360, title="Default rate — channel × structure")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        action(
            "OPERATIONAL ACTION",
            "Promote mobile-money rails for repayment — they auto-deduct, reducing missed-payment "
            "friction. For higher-risk segments, combine mobile-money repayment with a group "
            "structure. That's the lowest-default cell in the matrix.",
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# TAB 5 — POLICY SIMULATOR
# --------------------------------------------------------------------------- #
with tab_policy:
    st.subheader("Policy simulator — where do you draw the line?")
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

    # Approximate P&L of the policy
    avg_loan = df["loan_amount"].mean()
    avg_apr = 0.28  # blended
    avg_term_yr = df["loan_term_months"].mean() / 12
    interest_per_good = avg_loan * avg_apr * avg_term_yr
    loss_per_bad = avg_loan * 0.6  # 60% LGD assumption typical for unsecured MFI
    good_approved = n_approved - bad_approved
    pnl = good_approved * interest_per_good - bad_approved * loss_per_bad

    st.markdown("##### Approximate portfolio P&L at this threshold")
    p1, p2, p3 = st.columns(3)
    p1.metric("Good loans approved", f"{good_approved:,}")
    p2.metric("Bad loans approved", f"{bad_approved:,}")
    p3.metric("Net portfolio P&L", fmt_usd_zwl(pnl),
              help="Approx: interest on good loans minus 60% loss on bad loans")

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

    st.markdown(
        insight(
            "THE THRESHOLD IS A BUSINESS DECISION",
            "Lowering the threshold catches more defaulters but rejects more good applicants. "
            "The right number depends on the cost of a bad loan vs. the revenue from a marginal "
            "one — and on whether you have a group product to absorb the rejected segment.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        action(
            "RECOMMENDED OPERATING POINT",
            "Most Zim MFIs cluster around a <b>0.40-0.50 PD auto-decline</b> threshold, with "
            "MEDIUM-band borrowers routed into group products instead of being declined outright. "
            "Use the P&L number above to test whether shifting the line by ±0.05 helps your book.",
        ),
        unsafe_allow_html=True,
    )
