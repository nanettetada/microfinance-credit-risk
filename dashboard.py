"""The Loan Desk — a credit decision tool for Zimbabwean microfinance.

Run with:
    streamlit run dashboard.py

The dashboard reads the trained model directly via `src.predict`, so it works
even when the FastAPI service isn't running.
"""
from __future__ import annotations

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
from src.predict import MODEL_PATH, get_metadata, predict_many

# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
USD_TO_ZWL = 27.0  # 1 USD ≈ 27 ZiG (2026 approx). Edit when the rate moves.

REPAYMENT_CHANNELS = ["EcoCash", "OneMoney", "InnBucks", "Bank transfer", "Cash at branch"]
LENDING_STRUCTURES = ["Individual", "Group"]
INCOME_TYPES = ["Formal", "Informal"]
INTEREST_BY_BAND = {"LOW": 0.18, "MEDIUM": 0.28, "HIGH": 0.42}

PURPOSE_LABELS = {
    "SCHOOL_FEES": "School fees",
    "DEBT_CONSOLIDATION": "Paying off other debt",
    "MEDICAL": "Medical bills",
    "AGRIC_INPUTS": "Seed & fertiliser",
    "FUNERAL": "Funeral",
    "BUSINESS": "Stock for a business",
    "CAR": "A vehicle",
    "HOME_IMPROVEMENT": "Fixing up the home",
    "SOLAR_BACKUP": "Solar / generator",
    "OTHER": "Something else",
}

# Plain-English names for the model's features (used in the reasons panel).
FEATURE_LABELS = {
    "age": "their age",
    "income": "their income",
    "loan_amount": "the loan size",
    "loan_term_months": "the loan term",
    "employment_years": "how long they've been earning",
    "credit_history_months": "the length of their credit history",
    "existing_loans": "their existing loans",
    "home_ownership": "their home situation",
    "purpose": "what the loan is for",
}

# Palette — warm editorial light theme.
INK = "#1A1A17"
MUTED = "#7A756A"
LINE = "#E7E3DA"
GREEN = "#16794C"
AMBER = "#B4690E"
RED = "#B3261E"
PAPER = "#FBFAF7"

BAND_COLOR = {"LOW": GREEN, "MEDIUM": AMBER, "HIGH": RED}
BAND_VERDICT = {"LOW": "Approve", "MEDIUM": "Take a closer look", "HIGH": "Decline"}


def usd_zwl(usd: float) -> str:
    return f"${usd:,.0f} <span class='zwl'>≈ ZiG {usd * USD_TO_ZWL:,.0f}</span>"


# --------------------------------------------------------------------------- #
st.set_page_config(page_title="The Loan Desk", page_icon="•", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, p, span, div, label, input, button { font-family: 'Inter', sans-serif; }
    .stApp { background: #FBFAF7; }
    #MainMenu, footer, header[data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 2.2rem; max-width: 1180px; }

    /* Wordmark header */
    .wordmark { font-family: 'Fraunces', serif; font-weight: 600; font-size: 34px;
                color: #1A1A17; letter-spacing: -0.5px; line-height: 1; margin: 0; }
    .wordmark .dot { color: #16794C; }
    .tagline { color: #7A756A; font-size: 15px; margin: 8px 0 0 0; max-width: 560px; line-height: 1.5; }
    .modelchip { display: inline-flex; align-items: center; gap: 8px; background: #fff;
                 border: 1px solid #E7E3DA; border-radius: 999px; padding: 7px 14px;
                 font-size: 12.5px; color: #4A463E; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
    .modelchip b { color: #16794C; font-weight: 600; }
    .rule { height: 1px; background: #E7E3DA; border: 0; margin: 22px 0 6px 0; }

    /* Section heading */
    .sec { font-family: 'Fraunces', serif; font-weight: 500; font-size: 23px; color: #1A1A17;
           letter-spacing: -0.3px; margin: 6px 0 2px 0; }
    .sec-sub { color: #7A756A; font-size: 14.5px; margin: 0 0 14px 0; line-height: 1.55; max-width: 720px; }

    /* Form group label */
    .grp { font-size: 12px; font-weight: 600; color: #16794C; letter-spacing: 0.4px;
           margin: 4px 0 2px 0; text-transform: uppercase; }

    /* Decision card */
    .card { background: #fff; border: 1px solid #E7E3DA; border-radius: 18px; padding: 26px 28px;
            box-shadow: 0 6px 24px rgba(26,26,23,0.05); }
    .verdict { display: inline-block; font-family: 'Fraunces', serif; font-weight: 600;
               font-size: 13px; letter-spacing: 0.5px; padding: 7px 15px; border-radius: 999px; }
    .headline { font-family: 'Fraunces', serif; font-size: 27px; font-weight: 500; color: #1A1A17;
                margin: 16px 0 2px 0; line-height: 1.25; letter-spacing: -0.3px; }
    .headline b { font-weight: 600; }
    .sub { color: #7A756A; font-size: 14px; margin: 0; }

    /* Risk bar */
    .bar-wrap { margin: 22px 0 8px 0; }
    .bar { position: relative; height: 12px; border-radius: 999px;
           background: linear-gradient(90deg, #2E9E6B 0%, #34A06E 20%, #E0A33A 20%, #E0A33A 50%, #D2685E 50%, #C44539 100%); }
    .bar .marker { position: absolute; top: -6px; width: 3px; height: 24px; background: #1A1A17;
                   border-radius: 2px; transform: translateX(-50%); box-shadow: 0 0 0 3px rgba(251,250,247,0.9); }
    .bar-ticks { display: flex; justify-content: space-between; color: #9A9488; font-size: 11px; margin-top: 7px; }

    /* Reasons */
    .reasons { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 6px; }
    .reasons h4 { font-size: 12.5px; font-weight: 600; color: #4A463E; margin: 0 0 8px 0; }
    .reasons ul { margin: 0; padding-left: 0; list-style: none; }
    .reasons li { font-size: 14px; color: #3A362F; margin-bottom: 7px; padding-left: 18px; position: relative; line-height: 1.45; }
    .reasons li.up::before { content: "▴"; position: absolute; left: 0; color: #B3261E; }
    .reasons li.down::before { content: "▾"; position: absolute; left: 0; color: #16794C; }

    /* Offer rows */
    .offer { margin-top: 4px; }
    .offer-row { display: flex; justify-content: space-between; align-items: baseline;
                 padding: 12px 0; border-bottom: 1px solid #F0EDE5; }
    .offer-row:last-child { border-bottom: 0; }
    .offer-row .k { color: #7A756A; font-size: 13.5px; }
    .offer-row .v { color: #1A1A17; font-size: 16px; font-weight: 600; font-feature-settings: "tnum"; }
    .zwl { color: #A39C8D; font-size: 12.5px; font-weight: 400; }

    .note { background: #F3F1EA; border-radius: 12px; padding: 14px 16px; color: #4A463E;
            font-size: 13.5px; line-height: 1.55; margin-top: 14px; }
    .takeaway { color: #3A362F; font-size: 14.5px; line-height: 1.6; margin: 10px 0 2px 0; max-width: 760px; }
    .takeaway b { color: #1A1A17; }

    /* Tabs */
    div[data-testid="stTabs"] button[data-baseweb="tab"] { font-weight: 500; font-size: 15px; color: #7A756A; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #16794C; }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: #16794C; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_portfolio() -> pd.DataFrame:
    df = load_or_generate(n=20_000)
    rng = np.random.default_rng(7)
    n = len(df)
    pick = rng.random(n)
    channels = np.where(
        pick < 0.55, "EcoCash",
        np.where(pick < 0.72, "OneMoney",
        np.where(pick < 0.85, "InnBucks",
        np.where(pick < 0.95, "Bank transfer", "Cash at branch"))),
    )
    bank_shift = (df["home_ownership"] == "OWN") & (df["age"] > 40) & (rng.random(n) < 0.45)
    df["repayment_channel"] = np.where(bank_shift, "Bank transfer", channels)

    group_score = (
        (df["loan_amount"] < 5_000).astype(int) * 0.5
        + df["purpose"].isin(["AGRIC_INPUTS", "BUSINESS"]).astype(int) * 0.3
        + (df["existing_loans"] == 0).astype(int) * 0.2
    )
    df["lending_structure"] = np.where(rng.random(n) < group_score * 0.7, "Group", "Individual")

    informal_score = (
        (df["income"] < 35_000).astype(int) * 0.55
        + df["home_ownership"].isin(["LODGER", "RENT"]).astype(int) * 0.25
        + (df["employment_years"] < 3).astype(int) * 0.2
    )
    df["income_type"] = np.where(rng.random(n) < informal_score * 0.85, "Informal", "Formal")
    return df


@st.cache_data(show_spinner=False)
def score_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preds = predict_many(X.to_dict(orient="records"))
    df = df.copy()
    df["pd_score"] = [p["default_probability"] for p in preds]
    df["risk_band"] = [p["risk_band"] for p in preds]
    return df


@st.cache_data(show_spinner=False)
def feature_drivers(record: dict, baseline: float) -> pd.DataFrame:
    """One-at-a-time substitution: swap each feature for the portfolio median/mode
    and see how the score moves. Positive delta = the feature pushed risk up."""
    portfolio = load_portfolio()
    medians = {f: float(portfolio[f].median()) for f in NUMERIC_FEATURES}
    modes = {f: portfolio[f].mode().iloc[0] for f in CATEGORICAL_FEATURES}
    rows = []
    for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        alt = dict(record)
        alt[f] = medians[f] if f in NUMERIC_FEATURES else modes[f]
        alt_score = predict_many([alt])[0]["default_probability"]
        rows.append({"feature": f, "delta": baseline - alt_score})
    return pd.DataFrame(rows).sort_values("delta", key=lambda s: s.abs(), ascending=False)


def style_fig(fig, height=380):
    # Coerce title text to "" so Plotly.js doesn't render a stray "undefined"
    # for charts that have no title (their heading is an HTML element above).
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=INK, size=13),
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text=fig.layout.title.text or "",
                   font=dict(family="Fraunces, serif", size=17, color=INK)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    return fig


# --------------------------------------------------------------------------- #
# On a fresh deploy (Streamlit Cloud, HF Spaces) the model artifact is
# gitignored, so train it once on first boot rather than hard-stopping.
if not MODEL_PATH.exists():
    with st.spinner("First run — training the credit model (~20 seconds). This only happens once."):
        from src.train import train
        train()

meta = get_metadata()
auc = meta.get("metrics", {}).get("test_auc", 0)
version = meta.get("model_version", "n/a")

# Header
h_left, h_right = st.columns([3, 1.5])
with h_left:
    st.markdown('<p class="wordmark">The Loan Desk<span class="dot">.</span></p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="tagline">A quick, honest read on whether a microfinance loan is likely to be '
        'repaid — built around how lending actually works in Zimbabwe.</p>',
        unsafe_allow_html=True,
    )
with h_right:
    st.markdown(
        f'<div style="text-align:right; padding-top:8px;">'
        f'<span class="modelchip">Tells good loans from bad <b>{auc:.0%}</b> of the time</span>'
        f'<div style="color:#A39C8D;font-size:11.5px;margin-top:8px;">Priced in USD · ZiG at 1:{USD_TO_ZWL:.0f} · model v{version}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
st.markdown('<hr class="rule"/>', unsafe_allow_html=True)

df = score_portfolio(load_portfolio())

tab_decision, tab_book, tab_rules = st.tabs(["Make a decision", "The whole book", "Set the rules"])

# =========================================================================== #
# DECISION
# =========================================================================== #
with tab_decision:
    left, right = st.columns([0.82, 1], gap="large")

    with left:
        st.markdown('<div class="sec">Tell me about the application</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">A typical file: an informal trader, a small loan, mobile-money repayment.</div>', unsafe_allow_html=True)

        st.markdown('<div class="grp">The borrower</div>', unsafe_allow_html=True)
        age = st.slider("Age", 18, 80, 35)
        income_usd = st.number_input("Income, per year (USD)", 1_200, 500_000, 18_000, step=500,
                                     help="For an informal trader, estimate daily takings × ~280 trading days.")
        employment = st.slider("Years earning a living this way", 0, 50, 4)
        home = st.selectbox("Home situation", HOME_OWNERSHIP,
                            format_func=lambda h: {"RENT": "Renting", "OWN": "Owns home",
                                                   "MORTGAGE": "On a mortgage", "LODGER": "Lodging (a room)",
                                                   "OTHER": "Other"}.get(h, h))

        st.markdown('<div class="grp">The loan</div>', unsafe_allow_html=True)
        loan_amount_usd = st.number_input("How much they want (USD)", 100, 100_000, 800, step=50,
                                          help="Most Zim microfinance loans are under USD 2,000.")
        loan_term = st.selectbox("Paid back over", [3, 6, 12, 24, 36, 48, 60], index=2,
                                 format_func=lambda m: f"{m} months")
        purpose = st.selectbox("What it's for", PURPOSE, format_func=lambda p: PURPOSE_LABELS.get(p, p))
        existing_loans = st.slider("Loans they already have", 0, 10, 1)
        credit_history = st.slider("Months of credit history", 0, 600, 36)

        st.markdown('<div class="grp">How they came in</div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            lending_structure = st.radio("Applying as", LENDING_STRUCTURES, horizontal=True)
        with cc2:
            repayment_channel = st.selectbox("Repays via", REPAYMENT_CHANNELS, index=0)
        income_type = st.radio("Income is", INCOME_TYPES, horizontal=True)

    # --- scoring ---
    record = {
        "age": age, "income": income_usd, "loan_amount": loan_amount_usd,
        "loan_term_months": loan_term, "employment_years": employment,
        "credit_history_months": credit_history, "existing_loans": existing_loans,
        "home_ownership": home, "purpose": purpose,
    }
    raw = predict_many([record])[0]
    prob = raw["default_probability"]
    # Group lending: peer guarantee historically lowers effective risk.
    eff = prob * (0.85 if lending_structure == "Group" else 1.0)
    band = "LOW" if eff < 0.20 else ("MEDIUM" if eff < 0.50 else "HIGH")
    apr = INTEREST_BY_BAND[band]
    total_interest = loan_amount_usd * apr * (loan_term / 12)
    monthly = (loan_amount_usd + total_interest) / max(loan_term, 1)
    dti = monthly / max(income_usd / 12, 1)

    color = BAND_COLOR[band]
    verdict = BAND_VERDICT[band]
    marker_pos = min(max(eff * 100, 1.5), 98.5)

    drivers = feature_drivers(record, prob)
    up = drivers[drivers["delta"] > 0].head(3)
    down = drivers[drivers["delta"] < 0].head(3)
    up_items = "".join(f'<li class="up">{FEATURE_LABELS[r.feature].capitalize()}</li>' for r in up.itertuples()) or '<li class="up" style="color:#A39C8D">Nothing major</li>'
    down_items = "".join(f'<li class="down">{FEATURE_LABELS[r.feature].capitalize()}</li>' for r in down.itertuples()) or '<li class="down" style="color:#A39C8D">Nothing major</li>'

    if band == "LOW":
        headline = f"There's about a <b>{eff:.0%}</b> chance this loan goes bad."
        takeaway = f"Comfortable to approve at <b>{apr:.0%} a year</b>. The repayment lands at {usd_zwl(monthly)} a month."
    elif band == "MEDIUM":
        headline = f"Roughly a <b>{eff:.0%}</b> chance of trouble — worth a second look."
        takeaway = f"Approvable with a guarantor, a smaller amount, or a clean repayment record first. Suggested rate <b>{apr:.0%} a year</b>."
    else:
        headline = f"High risk — around a <b>{eff:.0%}</b> chance this isn't repaid."
        takeaway = f"Decline as an individual loan, or route them into a solidarity group with weekly meetings. If approved anyway, cap the term at 6 months and price at <b>{apr:.0%}</b>."

    group_line = ""
    if lending_structure == "Group":
        group_line = '<div class="note">This is a <b>group application</b> — the peer guarantee shaves about 15% off the standalone risk, which is already reflected above.</div>'
    elif dti > 0.5:
        group_line = f'<div class="note">Heads-up: the monthly repayment is <b>{dti:.0%}</b> of their monthly income — above the 40% comfort line. Consider a smaller loan or a longer term.</div>'

    with right:
        st.markdown(
            f"""
            <div class="card">
              <span class="verdict" style="background:{color}1A; color:{color};">{verdict}</span>
              <div class="headline">{headline}</div>
              <p class="sub">For {PURPOSE_LABELS.get(purpose, purpose).lower()} · {usd_zwl(loan_amount_usd)} over {loan_term} months</p>

              <div class="bar-wrap">
                <div class="bar"><div class="marker" style="left:{marker_pos:.1f}%;"></div></div>
                <div class="bar-ticks"><span>Safe</span><span>Watch (20%)</span><span>Risky (50%)</span><span>100%</span></div>
              </div>

              <div class="reasons">
                <div><h4>Working against them</h4><ul>{up_items}</ul></div>
                <div><h4>In their favour</h4><ul>{down_items}</ul></div>
              </div>

              <div class="offer" style="margin-top:18px;">
                <div class="offer-row"><span class="k">Recommended rate</span><span class="v">{apr:.0%} a year</span></div>
                <div class="offer-row"><span class="k">Monthly repayment</span><span class="v">{usd_zwl(monthly)}</span></div>
                <div class="offer-row"><span class="k">Total interest over the term</span><span class="v">{usd_zwl(total_interest)}</span></div>
                <div class="offer-row"><span class="k">Repayment vs monthly income</span><span class="v" style="color:{'#B3261E' if dti>0.5 else INK}">{dti:.0%}</span></div>
              </div>

              <p class="takeaway">{takeaway}</p>
              {group_line}
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================================== #
# THE WHOLE BOOK
# =========================================================================== #
with tab_book:
    st.markdown('<div class="sec">How the whole book looks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Every chart here is the model scoring 20,000 applications. '
        'It\'s how you\'d sanity-check that the model is actually sorting good loans from bad ones.</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Applications scored", f"{len(df):,}")
    m2.metric("Actually defaulted", f"{df['default'].mean()*100:.1f}%")
    m3.metric("Average loan", f"${df['loan_amount'].mean():,.0f}")

    c1, c2 = st.columns(2)
    with c1:
        bands = df["risk_band"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).reset_index()
        bands.columns = ["band", "count"]
        fig = px.pie(bands, names="band", values="count", hole=0.6,
                     color="band", color_discrete_map=BAND_COLOR,
                     title="Where the applications land")
        fig.update_traces(textinfo="label+percent", textfont_size=13)
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with c2:
        fig = px.histogram(df, x="pd_score", nbins=30, color="risk_band",
                           color_discrete_map=BAND_COLOR,
                           title="The spread of risk scores",
                           labels={"pd_score": "Chance of default"})
        fig.update_layout(legend_title_text="", bargap=0.04)
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    st.markdown('<div class="sec" style="font-size:19px;margin-top:14px;">Does a higher score really mean a worse loan?</div>', unsafe_allow_html=True)
    d = df.copy()
    d["decile"] = pd.qcut(d["pd_score"], 10, labels=range(1, 11))
    dec = d.groupby("decile", observed=True)["default"].mean().reset_index()
    dec.columns = ["decile", "rate"]
    fig = px.bar(dec, x="decile", y="rate", color="rate", color_continuous_scale=["#CFE8DC", GREEN, AMBER, RED],
                 text=dec["rate"].map(lambda x: f"{x*100:.0f}%"),
                 labels={"decile": "Riskiest 10% of scores  →", "rate": "Actually defaulted"})
    fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    top = dec.iloc[-1]["rate"]; base = df["default"].mean()
    st.markdown(
        f'<p class="takeaway">Yes. The 10% of applications the model flags as riskiest default at '
        f'<b>{top*100:.0f}%</b> — about <b>{top/base:.1f}×</b> the book average of {base*100:.0f}%. '
        f'That gap is the whole point: it\'s what lets you price for risk instead of guessing.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec" style="font-size:19px;margin-top:18px;">What people borrow for — and how often it goes wrong</div>', unsafe_allow_html=True)
    pur = (df.groupby("purpose").agg(rate=("default", "mean"), n=("default", "size"),
                                     avg=("loan_amount", "mean")).reset_index()
           .sort_values("rate", ascending=True))
    pur["label"] = pur["purpose"].map(PURPOSE_LABELS).fillna(pur["purpose"])
    fig = px.bar(pur, x="rate", y="label", orientation="h", color="rate",
                 color_continuous_scale=["#CFE8DC", AMBER, RED],
                 text=pur["rate"].map(lambda x: f"{x*100:.0f}%"),
                 labels={"rate": "Default rate", "label": ""})
    fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    worst = pur.iloc[-1]; best = pur.iloc[0]
    st.markdown(
        f'<p class="takeaway"><b>{worst["label"]}</b> loans go bad most often ({worst["rate"]*100:.0f}%), '
        f'while <b>{best["label"].lower()}</b> loans are the safest ({best["rate"]*100:.0f}%). '
        f'Funeral and seed-and-fertiliser loans carry the most tail risk — one is urgency-driven, the other rides on the rains.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec" style="font-size:19px;margin-top:18px;">The Zimbabwe-specific cuts</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Group lending and mobile-money repayment are the levers that actually move default rates here.</div>', unsafe_allow_html=True)
    z1, z2 = st.columns(2)
    with z1:
        s = df.groupby("lending_structure")["default"].mean().reset_index()
        fig = px.bar(s, x="lending_structure", y="default", color="lending_structure",
                     color_discrete_map={"Group": GREEN, "Individual": "#5B8DB8"},
                     text=s["default"].map(lambda x: f"{x*100:.1f}%"),
                     title="Group vs individual", labels={"default": "Default rate", "lending_structure": ""})
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with z2:
        ch = (df.groupby("repayment_channel")["default"].mean().reset_index()
              .sort_values("default"))
        fig = px.bar(ch, x="repayment_channel", y="default", color="default",
                     color_continuous_scale=["#CFE8DC", AMBER, RED],
                     text=ch["default"].map(lambda x: f"{x*100:.1f}%"),
                     title="By repayment channel", labels={"default": "Default rate", "repayment_channel": ""})
        fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    g = df.groupby("lending_structure")["default"].mean()
    st.markdown(
        f'<p class="takeaway">Solidarity groups default at <b>{g.get("Group",0)*100:.1f}%</b> against '
        f'<b>{g.get("Individual",0)*100:.1f}%</b> for individuals — peer pressure works. The practical move: '
        f'put new borrowers through a group product first, then graduate your best repayers onto individual loans.</p>',
        unsafe_allow_html=True,
    )

# =========================================================================== #
# SET THE RULES
# =========================================================================== #
with tab_rules:
    st.markdown('<div class="sec">Where do you draw the line?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Pick the score above which you\'ll automatically decline. '
        'Stricter catches more bad loans but turns away more good customers — drag it and watch the trade-off.</div>',
        unsafe_allow_html=True,
    )

    threshold = st.slider("Decline anyone scoring above", 0.05, 0.95, 0.50, step=0.01, format="%.2f")
    approved = df[df["pd_score"] < threshold]
    declined = df[df["pd_score"] >= threshold]
    bad_approved = int(approved["default"].sum())
    good_approved = len(approved) - bad_approved
    base_rate = df["default"].mean()
    approved_rate = bad_approved / max(len(approved), 1)

    avg_loan = df["loan_amount"].mean()
    interest_per_good = avg_loan * 0.28 * (df["loan_term_months"].mean() / 12)
    loss_per_bad = avg_loan * 0.6  # 60% loss given default, typical unsecured MFI
    pnl = good_approved * interest_per_good - bad_approved * loss_per_bad
    pnl_str = f"${pnl/1e6:,.1f}M" if abs(pnl) >= 1e6 else f"${pnl:,.0f}"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("You'd approve", f"{len(approved):,}", f"{len(approved)/len(df)*100:.0f}% of applicants")
    k2.metric("Bad loans let through", f"{bad_approved:,}")
    k3.metric("Default rate in your book", f"{approved_rate*100:.1f}%",
              f"{(approved_rate-base_rate)*100:+.1f} pts vs no model", delta_color="inverse")
    k4.metric("Margin on this book", pnl_str,
              help="Interest earned over the life of the approved loans, minus a 60% loss on the bad ones that slip through. Not annualised.")

    rows = []
    for t in np.linspace(0.05, 0.95, 19):
        ap = df[df["pd_score"] < t]; de = df[df["pd_score"] >= t]
        rows.append({"t": t, "Approval rate": len(ap)/len(df),
                     "Default rate in approved book": ap["default"].mean() if len(ap) else 0,
                     "Share of bad loans caught": de["default"].sum()/max(df["default"].sum(), 1)})
    trade = pd.DataFrame(rows)
    fig = go.Figure()
    for col, c in [("Approval rate", "#5B8DB8"), ("Default rate in approved book", RED), ("Share of bad loans caught", GREEN)]:
        fig.add_trace(go.Scatter(x=trade["t"], y=trade[col], name=col, mode="lines",
                                 line=dict(color=c, width=3)))
    fig.add_vline(x=threshold, line=dict(color=INK, dash="dot", width=1.5))
    fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Decline threshold",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.28))
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    st.markdown(
        '<p class="takeaway">There\'s no single right answer — it depends on what a bad loan costs you versus '
        'what you earn on a marginal good one. In practice most Zimbabwean lenders settle around a '
        '<b>0.40–0.50</b> cut-off and route the middle band into group products rather than declining them outright.</p>',
        unsafe_allow_html=True,
    )
