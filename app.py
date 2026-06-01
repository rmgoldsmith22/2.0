from __future__ import annotations

import io
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from calculations import (
    build_positions,
    calc_alpha_r2_tracking_error,
    calc_annualized_volatility,
    calc_calmar_ratio,
    calc_capture_ratios,
    calc_correlation_matrix,
    calc_cvar,
    calc_hhi,
    calc_max_drawdown,
    calc_monte_carlo,
    calc_portfolio_beta,
    calc_portfolio_cumulative,
    calc_rolling_metrics,
    calc_sharpe_ratio,
    calc_sortino_ratio,
    calc_treynor_ratio,
    calc_var,
)
from data_fetcher import fetch_price_history, fetch_ticker_info

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Portfolio Risk Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif !important; }
.stApp { background-color: #0F172A !important; }

h1, h2, h3,
[data-testid="stHeadingWithActionElements"] h1,
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    letter-spacing: -0.02em !important;
    color: #F8FAFC !important;
}

[data-testid="stSidebar"] {
    background-color: #1E293B !important;
    border-right: 1px solid #334155 !important;
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif !important; }

[data-testid="metric-container"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    box-shadow: inset 0 2px 0 0 #F59E0B !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] > div {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #64748B !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid #334155 !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    color: #64748B !important;
    background-color: transparent !important;
    border: none !important;
    padding: 10px 18px !important;
}
.stTabs [aria-selected="true"] {
    color: #F59E0B !important;
    border-bottom: 2px solid #F59E0B !important;
    background-color: transparent !important;
}

[data-testid="stExpander"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
}

.stSelectbox label, .stSlider label, .stFileUploader label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #64748B !important;
}

.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    background-color: #F59E0B !important;
    color: #0F172A !important;
    border: none !important;
    border-radius: 6px !important;
}
.stButton > button:hover { background-color: #D97706 !important; }

[data-testid="stDataFrame"], [data-testid="stTable"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
}

p, li, span, div { font-family: 'DM Sans', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

HERE = pathlib.Path(__file__).parent

BENCHMARKS: dict[str, str] = {"S&P 500": "^GSPC", "Nasdaq 100": "QQQ", "MSCI World": "ACWI"}
PRIMARY_BENCH = "^GSPC"

# ── Plain-English metric definitions ─────────────────────────────────────────

HELP = {
    "beta":           "Sensitivity to the market. Beta 1.0 = moves with market; 1.5 = 50% more volatile; 0.5 = half as volatile.",
    "var_1d":         "Maximum expected 1-day loss at 95% confidence — should only be exceeded ~1 in 20 trading days based on history.",
    "sharpe":         "Return earned per unit of total risk: (Return − Risk-Free Rate) ÷ Volatility. Above 1.0 is good; above 2.0 is excellent.",
    "sortino":        "Like Sharpe, but only penalises downside moves. Preferred for retirement accounts — upside volatility is not a problem.",
    "max_dd":         "Largest peak-to-trough decline over the period. The loss an investor who bought at the worst moment would have experienced.",
    "hhi":            "Concentration score: sum of squared weights. Under 0.15 = diversified; over 0.25 = concentrated. 1.0 = all in one position.",
    "cvar":           "Expected Shortfall: the average loss on the worst 5% of days — more informative than VaR, which only shows the threshold.",
    "calmar":         "Annual return ÷ |Max Drawdown|. Above 1.0 means the annual gain exceeds the worst historical loss.",
    "treynor":        "Excess return per unit of market risk (beta). Useful when this portfolio is one sleeve of a larger diversified allocation.",
    "ann_vol":        "Annualised standard deviation of daily returns. 15% vol = portfolio can typically swing ±15% over a full year.",
    "alpha":          "Jensen's Alpha: return above what CAPM predicts given your beta vs the S&P 500. Positive = outperformed expectations.",
    "r2":             "% of portfolio movement explained by the S&P 500. High R² ≈ index fund. Low R² = idiosyncratic or truly diversified drivers.",
    "tracking_error": "Annualised std of (portfolio − benchmark) returns. Low = hugs the index. High = active, differentiated strategy.",
    "up_capture":     "In rising S&P 500 markets, what % of gains did the portfolio capture? 110% = outperformed during rallies.",
    "down_capture":   "In falling S&P 500 markets, what % of losses did the portfolio absorb? 80% = fell only 80% as much as the market. Lower is better.",
    "rolling":        "A static Sharpe or volatility number hides regime changes. Rolling charts reveal whether risk behaviour has been consistent over time.",
    "monte_carlo":    "Uses historical mean return and volatility to run thousands of simulated future paths. Not a prediction — a probabilistic planning range.",
}

SECTION_EXPLAINERS = {
    "overview": """
**Portfolio Beta** — How amplified your exposure to the market is. A beta of 1.2 means a 10% market rally would be expected to produce a 12% gain (and vice versa on the downside). Values below 1.0 reduce market sensitivity; values above 1.0 increase it.

**1-Day VaR (95%)** — The dollar loss you expect to exceed no more than once every 20 trading days. Computed using historical simulation over the past 252 trading days.

**Sharpe Ratio** — The most widely used risk-adjusted performance measure. Divides annualised excess return by annualised total volatility. A ratio above 1.0 is considered good for a diversified portfolio; most long-only equity portfolios land between 0.3 and 1.2.

**Sortino Ratio** — Identical to Sharpe except the denominator uses only *downside deviation* (returns below the risk-free rate). Since investors don't mind large gains, the Sortino is a more investor-friendly measure — particularly for retirement-focused portfolios. A Sortino above 1.5 is excellent.

**Max Drawdown** — The steepest valley in the portfolio's history. Behavioural finance research shows that many investors sell near the trough. Knowing the max drawdown sets realistic expectations and helps clients stick to their plan.

**HHI Concentration Index** — The Herfindahl-Hirschman Index quantifies portfolio concentration as the sum of squared weights. A 10-position equal-weight portfolio scores 0.10. The SEC and FINRA consider portfolios with HHI above 0.25 to be "concentrated."
""",
    "risk": """
**CVaR / Expected Shortfall** — VaR tells you the loss threshold you'll breach 5% of the time. CVaR tells you what the *average* loss is on those worst 5% of days. Required by Basel III bank regulations because it better captures tail risk.

**Calmar Ratio** — A favourite of alternative investment managers: annual return divided by absolute max drawdown. A Calmar above 1.0 means the strategy earns back its worst historical loss in under one year.

**Treynor Ratio** — Strips out diversifiable (idiosyncratic) risk and evaluates return only relative to systematic (market) beta. Especially useful when comparing portfolio sleeves within a broader asset allocation.

**Rolling Sharpe (90-day)** — A static Sharpe masks stress periods. This chart shows whether risk-adjusted performance has been consistent or whether it degraded during market dislocations.

**Rolling Volatility (90-day)** — Reveals volatility clustering: calm periods interrupted by spikes. Useful for timing rebalancing and for setting client expectations during volatile regimes.

**Drawdown Chart** — Every trough below the running peak, over the full lookback period. Useful for identifying recovery time and whether the portfolio tends to bounce back quickly or lingers underwater.

**Correlation Matrix** — Values near +1.0 mean two holdings move together — holding both provides limited diversification benefit. Values near −1.0 mean they offset each other. A well-diversified portfolio shows many values between −0.3 and +0.5.
""",
    "benchmarks": """
**Jensen's Alpha** — The intercept from regressing your portfolio's excess returns against the benchmark's excess returns. An alpha of +3% means the portfolio outperformed the CAPM prediction by 3% per year given its level of market risk. This is the measure of manager skill vs passive exposure.

**R-Squared** — If R² is 0.95, then 95% of your daily fluctuations are explained by the S&P 500's movements. A high R² with moderate Sharpe suggests you're taking a lot of market risk. Low R² means your returns are driven by factors beyond broad market beta.

**Tracking Error** — The annualised standard deviation of (portfolio return − benchmark return). An S&P 500 index fund has ~0.02% tracking error. An active manager typically runs 3–8%. High tracking error is only justified if it accompanies positive alpha.

**Up Capture Ratio** — During all days/periods the S&P 500 rose, what fraction of those gains did the portfolio earn? Calculated using annualised geometric returns. 100% = matched the market on up days; above 100% = beat it.

**Down Capture Ratio** — During all days/periods the S&P 500 fell, what fraction of those losses did the portfolio absorb? Below 100% is desirable. The ideal active manager has high up-capture and low down-capture.
""",
    "monte_carlo": """
**How the simulation works** — The model fits a normal distribution to the portfolio's historical daily mean return and volatility. It then draws from that distribution to generate the specified number of independent random paths over the chosen horizon.

**Reading the fan chart:**
- **Median line (50th pct)** — Half of all simulated paths end above this value. This is your central estimate.
- **Inner band (25th–75th pct)** — The "most likely" range. Half of all simulated outcomes fall within this zone.
- **Outer band (5th–95th pct)** — Covers 90% of simulated scenarios. Outcomes outside this band are statistically rare under the model's assumptions.

**Important caveats for clients:**
- Assumes *constant* weights and that historical return and volatility persist into the future.
- Does not model fat tails, correlation breakdowns during crises, or black swan events.
- Does not account for taxes, advisory fees, or inflation in real terms (unless your return inputs already reflect them).
- Use as a planning framework, not a forecast. Revise assumptions annually.
""",
}

# ── Cached data fetchers ──────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def cached_price_history(tickers_tuple: tuple, period: str = "1y"):
    return fetch_price_history(list(tickers_tuple), period)


@st.cache_data(ttl=300, show_spinner=False)
def cached_ticker_info(ticker: str) -> dict:
    return fetch_ticker_info(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_benchmark_history(period: str = "1y") -> pd.DataFrame:
    df, _ = fetch_price_history(list(BENCHMARKS.values()), period)
    return df


@st.cache_data(show_spinner=False)
def cached_monte_carlo(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    initial_value: float,
    years: int,
    simulations: int,
    monthly_contribution: float,
) -> pd.DataFrame | None:
    return calc_monte_carlo(price_df, positions_df, initial_value, years, simulations, monthly_contribution)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Portfolio Risk")
    st.markdown("---")
    st.subheader("Portfolio Input")
    uploaded = st.file_uploader("CSV: Ticker, Shares, Avg_Cost", type=["csv"], label_visibility="collapsed")

    st.caption("—— or ——")
    col_load, col_dl = st.columns(2)
    with col_load:
        load_sample = st.button("▶ Load Sample", use_container_width=True)
    with col_dl:
        st.download_button("⬇ Download", data=(HERE / "sample_portfolio.csv").read_bytes(),
                           file_name="sample_portfolio.csv", mime="text/csv", use_container_width=True)

    if load_sample:
        st.session_state["use_sample"] = True
    if uploaded is not None:
        st.session_state["use_sample"] = False

    st.markdown("---")
    st.subheader("Risk Settings")
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.25,
                                help="Annual rate used in Sharpe, Sortino, Treynor, and Alpha calculations.") / 100
    mc_sims = st.select_slider("Monte Carlo Paths", options=[500, 1000, 2000, 5000], value=1000,
                                help="More paths = smoother bands but slower calculation.")

    st.markdown("---")
    st.caption("Data: Yahoo Finance · Cache: 5 min")

# ── Resolve data source ───────────────────────────────────────────────────────

st.title("📊 Portfolio Risk Dashboard")

portfolio_df: pd.DataFrame | None = None
if uploaded is not None:
    try:
        portfolio_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not parse CSV: {exc}")
        st.stop()
elif st.session_state.get("use_sample"):
    portfolio_df = pd.read_csv(HERE / "sample_portfolio.csv")
else:
    st.info("Upload a portfolio CSV or click **▶ Load Sample** in the sidebar to get started.")
    st.stop()

missing = {"Ticker", "Shares", "Avg_Cost"} - set(portfolio_df.columns)
if missing:
    st.error(f"CSV missing columns: {', '.join(sorted(missing))}")
    st.stop()

portfolio_df["Ticker"] = portfolio_df["Ticker"].astype(str).str.strip().str.upper()
tickers = portfolio_df["Ticker"].tolist()

# ── Fetch market data ─────────────────────────────────────────────────────────

with st.spinner("Fetching market data from Yahoo Finance…"):
    price_df, failed = cached_price_history(tuple(tickers))
    ticker_info = {t: cached_ticker_info(t) for t in tickers if t not in failed}
    bench_df = cached_benchmark_history()

for t in failed:
    st.warning(f"No data for **{t}** — skipped.")

active_df = portfolio_df[~portfolio_df["Ticker"].isin(failed)].copy()
if active_df.empty:
    st.error("No valid tickers. Check your CSV.")
    st.stop()

# Patch current_price from last row of price history.
# Ticker.info is unreliable on cloud hosts (rate-limited); price history is more robust.
for t in active_df["Ticker"].tolist():
    if t in price_df.columns:
        last = price_df[t].dropna()
        if not last.empty:
            ticker_info.setdefault(t, {})["current_price"] = float(last.iloc[-1])

# ── Build positions ───────────────────────────────────────────────────────────

positions = build_positions(active_df, ticker_info)
if positions.empty:
    st.error("Could not build positions — verify tickers have valid prices.")
    st.stop()

valid = positions["Ticker"].tolist()
pdf = price_df[[t for t in valid if t in price_df.columns]]  # trimmed price DataFrame

# ── All calculations (done before tabs to avoid re-computation) ───────────────

port_beta   = calc_portfolio_beta(positions)
corr        = calc_correlation_matrix(pdf)
var_data    = calc_var(pdf, positions)
cvar_data   = calc_cvar(pdf, positions)
sharpe      = calc_sharpe_ratio(pdf, positions, risk_free_rate)
sortino     = calc_sortino_ratio(pdf, positions, risk_free_rate)
calmar      = calc_calmar_ratio(pdf, positions)
treynor     = calc_treynor_ratio(pdf, positions, port_beta, risk_free_rate)
max_dd, dd_series = calc_max_drawdown(pdf, positions)
ann_vol     = calc_annualized_volatility(pdf, positions)
hhi         = calc_hhi(positions)
port_cum    = calc_portfolio_cumulative(pdf, positions)
rolling     = calc_rolling_metrics(pdf, positions, window=63, risk_free_rate=risk_free_rate)

primary_bench_returns: pd.Series | None = None
if not bench_df.empty and PRIMARY_BENCH in bench_df.columns:
    primary_bench_returns = bench_df[PRIMARY_BENCH].pct_change().dropna()

bench_stats = {"alpha": None, "r2": None, "tracking_error": None}
up_cap = down_cap = None
if primary_bench_returns is not None:
    bench_stats = calc_alpha_r2_tracking_error(pdf, positions, primary_bench_returns, risk_free_rate)
    up_cap, down_cap = calc_capture_ratios(pdf, positions, primary_bench_returns)

total_value = positions["Value"].sum()
total_cost  = positions["Cost_Basis"].sum()
total_pnl   = positions["PnL"].sum()
total_pnl_pct = total_pnl / total_cost * 100 if total_cost else 0.0


# ── Helper: render a metric with help icon ────────────────────────────────────

def _fmt(val: float | None, fmt: str, prefix: str = "", suffix: str = "") -> str:
    return f"{prefix}{val:{fmt}}{suffix}" if val is not None else "N/A"


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_risk, tab_bench, tab_mc, tab_holdings = st.tabs([
    "📊 Overview", "⚠️ Risk Analysis", "📈 Benchmarks", "🎯 Projections", "📋 Holdings"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:

    # ── Summary cards row 1 ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Portfolio Value", f"${total_value:,.2f}")
    c2.metric("Unrealized P&L", f"${total_pnl:,.2f}", f"{total_pnl_pct:+.2f}%", delta_color="normal")
    c3.metric("Portfolio Beta", f"{port_beta:.2f}", help=HELP["beta"])
    if var_data["var_1d"] is not None:
        c4.metric("1-Day VaR (95%)", f"${var_data['var_1d']:,.2f}",
                  f"{var_data['var_1d_pct']:.2f}% of portfolio", delta_color="inverse",
                  help=HELP["var_1d"])
    else:
        c4.metric("1-Day VaR (95%)", "N/A", help=HELP["var_1d"])

    # ── Summary cards row 2 ───────────────────────────────────────────────────
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Sharpe Ratio (1Y)",  _fmt(sharpe,  ".2f"), help=HELP["sharpe"])
    c6.metric("Sortino Ratio (1Y)", _fmt(sortino, ".2f"), help=HELP["sortino"])
    c7.metric("Max Drawdown", f"{max_dd * 100:.2f}%" if max_dd is not None else "N/A",
              delta_color="inverse", help=HELP["max_dd"])
    c8.metric("HHI Concentration", _fmt(hhi, ".3f"), help=HELP["hhi"])

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown(SECTION_EXPLAINERS["overview"])

    st.divider()

    # ── Allocation + Sector + P&L ─────────────────────────────────────────────
    col_pie, col_sector, col_pnl = st.columns(3)

    with col_pie:
        st.subheader("Allocation")
        fig_pie = px.pie(positions, values="Value", names="Ticker", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Plotly)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10, l=0, r=0), height=340)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_sector:
        st.subheader("By Sector")
        sector_df = (positions.groupby("Sector")["Value"].sum()
                               .reset_index()
                               .sort_values("Value", ascending=True))
        sector_df["Weight %"] = sector_df["Value"] / total_value * 100
        fig_sec = go.Figure(go.Bar(
            x=sector_df["Weight %"], y=sector_df["Sector"],
            orientation="h",
            text=sector_df["Weight %"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            marker_color=px.colors.qualitative.Safe[:len(sector_df)],
        ))
        fig_sec.update_layout(xaxis_title="Weight (%)", yaxis_title=None,
                              margin=dict(t=10, b=10, l=0, r=10), height=340,
                              plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sec, use_container_width=True)

    with col_pnl:
        st.subheader("Unrealized P&L")
        pnl_s = positions.sort_values("PnL")
        fig_pnl = go.Figure(go.Bar(
            x=pnl_s["Ticker"], y=pnl_s["PnL"],
            marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in pnl_s["PnL"]],
            text=pnl_s["PnL"].apply(lambda v: f"${v:,.0f}"),
            textposition="outside",
        ))
        fig_pnl.update_layout(xaxis_title=None, yaxis_title="P&L ($)",
                              margin=dict(t=10, b=10, l=0, r=10), height=340,
                              plot_bgcolor="rgba(0,0,0,0)")
        fig_pnl.add_hline(y=0, line_color="gray", line_width=1)
        st.plotly_chart(fig_pnl, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab_risk:

    # ── Risk metric cards ─────────────────────────────────────────────────────
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("CVaR / Exp. Shortfall",
              f"${cvar_data['cvar_1d']:,.2f}" if cvar_data["cvar_1d"] else "N/A",
              f"{cvar_data['cvar_1d_pct']:.2f}% of portfolio" if cvar_data["cvar_1d_pct"] else None,
              delta_color="inverse", help=HELP["cvar"])
    r2.metric("Calmar Ratio",  _fmt(calmar,  ".2f"), help=HELP["calmar"])
    r3.metric("Treynor Ratio", _fmt(treynor, ".2f"), help=HELP["treynor"])
    r4.metric("Annualised Volatility",
              f"{ann_vol * 100:.2f}%" if ann_vol else "N/A", help=HELP["ann_vol"])

    # Beta gauge row
    st.divider()
    col_gauge, col_var = st.columns(2)

    with col_gauge:
        st.subheader("Portfolio Beta")
        gauge_max = max(3.0, round(port_beta * 1.6, 1))
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=port_beta,
            delta={"reference": 1.0, "suffix": " vs market"},
            number={"font": {"size": 44}},
            gauge={
                "axis": {"range": [0, gauge_max]},
                "bar": {"color": "#1f77b4", "thickness": 0.3},
                "steps": [
                    {"range": [0, 0.75],      "color": "#d4efdf"},
                    {"range": [0.75, 1.25],   "color": "#a9dfbf"},
                    {"range": [1.25, 2.0],    "color": "#f9e79f"},
                    {"range": [2.0, gauge_max], "color": "#f1948a"},
                ],
                "threshold": {"line": {"color": "black", "width": 3},
                              "thickness": 0.75, "value": 1.0},
            },
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=20, b=10, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_var:
        st.subheader("VaR vs CVaR")
        if var_data["var_1d"] and cvar_data["cvar_1d"]:
            var_compare = pd.DataFrame([
                {"Metric": "1-Day VaR (95%)",  "Loss ($)": var_data["var_1d"]},
                {"Metric": "1-Day CVaR (95%)", "Loss ($)": cvar_data["cvar_1d"]},
                {"Metric": "5-Day VaR (95%)",  "Loss ($)": var_data["var_5d"]},
            ])
            fig_vc = go.Figure(go.Bar(
                x=var_compare["Metric"], y=var_compare["Loss ($)"],
                marker_color=["#e74c3c", "#922b21", "#c0392b"],
                text=var_compare["Loss ($)"].apply(lambda v: f"${v:,.0f}"),
                textposition="outside",
            ))
            fig_vc.update_layout(xaxis_title=None, yaxis_title="Expected Loss ($)",
                                 height=260, margin=dict(t=10, b=10, l=10, r=10),
                                 showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_vc, use_container_width=True)
        else:
            st.info("Insufficient data to compute VaR/CVaR.")

    st.divider()

    # ── Rolling charts ────────────────────────────────────────────────────────
    if not rolling.empty:
        col_rs, col_rv = st.columns(2)

        with col_rs:
            st.subheader("Rolling Sharpe Ratio (90-day)")
            fig_rs = go.Figure(go.Scatter(
                x=rolling.index, y=rolling["Rolling Sharpe"],
                line=dict(color="#1f77b4", width=1.8), fill="tozeroy",
                fillcolor="rgba(31, 119, 180, 0.12)",
                hovertemplate="%{y:.2f}<extra></extra>",
            ))
            fig_rs.add_hline(y=1.0, line_dash="dash", line_color="#2ca02c",
                             annotation_text="1.0 threshold", annotation_position="top right")
            fig_rs.add_hline(y=0.0, line_color="gray", line_width=0.5)
            fig_rs.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10),
                                 yaxis_title="Sharpe", hovermode="x",
                                 plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rs, use_container_width=True)

        with col_rv:
            st.subheader("Rolling Volatility (90-day, ann.)")
            fig_rv = go.Figure(go.Scatter(
                x=rolling.index, y=rolling["Rolling Volatility (%)"],
                line=dict(color="#ff7f0e", width=1.8), fill="tozeroy",
                fillcolor="rgba(255, 127, 14, 0.12)",
                hovertemplate="%{y:.1f}%<extra></extra>",
            ))
            fig_rv.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10),
                                 yaxis_title="Volatility (%)", hovermode="x",
                                 plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rv, use_container_width=True)
    else:
        st.info("Not enough history for rolling metrics.")

    st.divider()

    # ── Drawdown chart ────────────────────────────────────────────────────────
    st.subheader("Drawdown from Rolling Peak")
    if dd_series is not None and not dd_series.empty:
        fig_dd = go.Figure(go.Scatter(
            x=dd_series.index, y=dd_series * 100,
            fill="tozeroy", fillcolor="rgba(214, 39, 40, 0.18)",
            line=dict(color="#d62728", width=1.5),
            hovertemplate="%{y:.2f}%<extra></extra>",
        ))
        fig_dd.add_hline(y=0, line_color="gray", line_width=0.5)
        fig_dd.update_layout(height=220, yaxis_title="Drawdown (%)", hovermode="x",
                             margin=dict(t=10, b=10, l=10, r=10),
                             showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.info("Insufficient data for drawdown chart.")

    st.divider()

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.subheader("Return Correlation Matrix")
    if not corr.empty and corr.shape[0] > 1:
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu_r", zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text:.2f}",
            colorbar=dict(title="ρ", tickvals=[-1, -0.5, 0, 0.5, 1]),
        ))
        fig_corr.update_layout(height=400, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Need at least 2 tickers for a correlation matrix.")

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown(SECTION_EXPLAINERS["risk"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

with tab_bench:

    # ── Benchmark metric cards ────────────────────────────────────────────────
    b1, b2, b3, b4, b5 = st.columns(5)
    alpha = bench_stats.get("alpha")
    r2    = bench_stats.get("r2")
    te    = bench_stats.get("tracking_error")

    b1.metric("Jensen's Alpha", f"{alpha * 100:+.2f}%" if alpha is not None else "N/A",
              delta_color="normal", help=HELP["alpha"])
    b2.metric("R² vs S&P 500", f"{r2 * 100:.1f}%" if r2 is not None else "N/A",
              help=HELP["r2"])
    b3.metric("Tracking Error", f"{te * 100:.2f}%" if te is not None else "N/A",
              help=HELP["tracking_error"])
    b4.metric("Up Capture", f"{up_cap:.1f}%" if up_cap is not None else "N/A",
              help=HELP["up_capture"])
    b5.metric("Down Capture", f"{down_cap:.1f}%" if down_cap is not None else "N/A",
              delta_color="inverse", help=HELP["down_capture"])

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown(SECTION_EXPLAINERS["benchmarks"])

    st.divider()

    # ── Performance comparison line chart ─────────────────────────────────────
    st.subheader("Performance vs Benchmarks")
    st.caption("Constant-weight backtest: shows what today's allocation would have returned over the past year.")

    if port_cum is not None and not port_cum.empty:
        start = port_cum.index[0]
        comparison = pd.DataFrame({"Your Portfolio": port_cum / port_cum.iloc[0] * 100})

        bench_label_map = {v: k for k, v in BENCHMARKS.items()}
        for ticker, label in bench_label_map.items():
            if ticker in bench_df.columns:
                s = bench_df[ticker].dropna()
                s = s[s.index >= start]
                if not s.empty:
                    comparison[label] = s / s.iloc[0] * 100

        compare_reset = comparison.reset_index()
        date_col = compare_reset.columns[0]
        fig_cmp = px.line(
            compare_reset, x=date_col, y=comparison.columns.tolist(),
            labels={"value": "Growth of $100", "variable": ""},
            color_discrete_map={
                "Your Portfolio": "#1f77b4",
                "S&P 500":       "#ff7f0e",
                "Nasdaq 100":    "#2ca02c",
                "MSCI World":    "#9467bd",
            },
        )
        fig_cmp.update_traces(line_width=2)
        fig_cmp.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=420, margin=dict(t=40, b=10, l=10, r=10),
            yaxis_title="Growth of $100", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Alpha/capture summary table ───────────────────────────────────────
        if alpha is not None:
            st.markdown("#### Relative Performance Summary (vs S&P 500)")
            summary_data = [
                {"Metric": "Jensen's Alpha (annualised)", "Value": f"{alpha * 100:+.2f}%",
                 "Interpretation": "Above market expectation" if alpha > 0 else "Below market expectation"},
                {"Metric": "R-Squared", "Value": f"{r2 * 100:.1f}%",
                 "Interpretation": "Highly index-like" if r2 > 0.9 else ("Moderately correlated" if r2 > 0.6 else "Low benchmark correlation")},
                {"Metric": "Tracking Error", "Value": f"{te * 100:.2f}%",
                 "Interpretation": "Tight index tracking" if te < 0.03 else ("Active range" if te < 0.10 else "Highly active")},
                {"Metric": "Up Capture", "Value": f"{up_cap:.1f}%" if up_cap else "N/A",
                 "Interpretation": "Captures upside well" if (up_cap and up_cap > 100) else "Underperforms in rallies"},
                {"Metric": "Down Capture", "Value": f"{down_cap:.1f}%" if down_cap else "N/A",
                 "Interpretation": "Good downside protection" if (down_cap and down_cap < 90) else "Limited downside protection"},
            ]
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data to build performance comparison.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · MONTE CARLO PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab_mc:

    st.subheader("Monte Carlo Portfolio Projection")
    st.caption("Simulates thousands of future portfolio paths using historical mean return and volatility.")

    mc_col1, mc_col2, mc_col3 = st.columns(3)
    mc_years  = mc_col1.slider("Time Horizon (years)", 5, 40, 20, 5)
    mc_contrib = mc_col2.number_input("Monthly Contribution ($)", min_value=0, max_value=50_000,
                                      value=0, step=100,
                                      help="Additional cash added each month (approximately every 21 trading days).")
    mc_col3.metric("Starting Value", f"${total_value:,.2f}")

    mc_df = cached_monte_carlo(pdf, positions, total_value, mc_years, mc_sims, float(mc_contrib))

    if mc_df is not None:
        # ── Fan chart ─────────────────────────────────────────────────────────
        fig_mc = go.Figure()

        # Outer band p5–p95
        fig_mc.add_trace(go.Scatter(x=mc_df["year"], y=mc_df["p95"], mode="lines",
                                    line=dict(color="rgba(0,0,0,0)"), showlegend=False))
        fig_mc.add_trace(go.Scatter(x=mc_df["year"], y=mc_df["p5"], mode="lines",
                                    fill="tonexty", fillcolor="rgba(31,119,180,0.10)",
                                    line=dict(color="rgba(0,0,0,0)"), name="5th–95th pct"))

        # Inner band p25–p75
        fig_mc.add_trace(go.Scatter(x=mc_df["year"], y=mc_df["p75"], mode="lines",
                                    line=dict(color="rgba(0,0,0,0)"), showlegend=False))
        fig_mc.add_trace(go.Scatter(x=mc_df["year"], y=mc_df["p25"], mode="lines",
                                    fill="tonexty", fillcolor="rgba(31,119,180,0.25)",
                                    line=dict(color="rgba(0,0,0,0)"), name="25th–75th pct"))

        # Median
        fig_mc.add_trace(go.Scatter(x=mc_df["year"], y=mc_df["p50"], mode="lines",
                                    line=dict(color="#1f77b4", width=2.5), name="Median (50th pct)"))

        # Starting value reference line
        fig_mc.add_hline(y=total_value, line_dash="dot", line_color="gray",
                         annotation_text=f"Starting value ${total_value:,.0f}",
                         annotation_position="top right")

        fig_mc.update_layout(
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Years from Today",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            height=440, margin=dict(t=40, b=10, l=10, r=10),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_mc.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig_mc, use_container_width=True)

        # ── Outcome table ─────────────────────────────────────────────────────
        st.markdown("#### Projected Outcomes by Horizon")
        horizon_rows = []
        for y in [1, 3, 5, 10, 20, 30]:
            if y > mc_years:
                continue
            idx = min(int(y * 252), len(mc_df) - 1)
            row = mc_df.iloc[idx]
            horizon_rows.append({
                "Horizon":              f"{y} year{'s' if y > 1 else ''}",
                "Bear (5th pct)":       f"${row['p5']:,.0f}",
                "Conservative (25th)":  f"${row['p25']:,.0f}",
                "Median (50th)":        f"${row['p50']:,.0f}",
                "Optimistic (75th)":    f"${row['p75']:,.0f}",
                "Bull (95th pct)":      f"${row['p95']:,.0f}",
            })
        st.dataframe(pd.DataFrame(horizon_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Not enough price history to run Monte Carlo simulation.")

    with st.expander("ℹ️ How to read this simulation"):
        st.markdown(SECTION_EXPLAINERS["monte_carlo"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · HOLDINGS
# ══════════════════════════════════════════════════════════════════════════════

with tab_holdings:

    st.subheader("Position Details")

    display_cols = ["Ticker", "Sector", "Shares", "Avg_Cost", "Current_Price",
                    "Value", "Cost_Basis", "PnL", "PnL_Pct", "Weight", "Beta"]
    display_df = positions[display_cols].rename(columns={
        "Avg_Cost": "Avg Cost", "Current_Price": "Price",
        "Cost_Basis": "Cost Basis", "PnL": "P&L ($)",
        "PnL_Pct": "P&L (%)", "Weight": "Weight (%)",
    })

    def _pnl_color(val):
        if not isinstance(val, (int, float, np.floating)):
            return ""
        return "color: #2ca02c" if val > 0 else ("color: #d62728" if val < 0 else "")

    styled = (
        display_df.style
        .format({"Avg Cost": "${:.2f}", "Price": "${:.2f}", "Value": "${:,.2f}",
                 "Cost Basis": "${:,.2f}", "P&L ($)": "${:,.2f}",
                 "P&L (%)": "{:.2f}%", "Weight (%)": "{:.2f}%",
                 "Beta": "{:.2f}", "Shares": "{:.4f}"}, na_rep="N/A")
        .map(_pnl_color, subset=["P&L ($)", "P&L (%)"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Beta Contribution by Position")
    beta_df = positions[["Ticker", "Beta", "Weight", "Beta_Contribution"]].sort_values(
        "Beta_Contribution", ascending=False)
    fig_beta = go.Figure(go.Bar(
        x=beta_df["Ticker"], y=beta_df["Beta_Contribution"],
        marker=dict(color=beta_df["Beta"], colorscale="RdYlGn_r",
                    showscale=True, colorbar=dict(title="Raw β")),
        text=beta_df["Beta_Contribution"].apply(lambda v: f"{v:.3f}"),
        textposition="outside",
    ))
    fig_beta.add_hline(y=port_beta, line_dash="dash",
                       annotation_text=f"Portfolio β = {port_beta:.2f}",
                       annotation_position="top right")
    fig_beta.update_layout(xaxis_title=None, yaxis_title="Beta Contribution",
                           height=300, margin=dict(t=20, b=10, l=10, r=10),
                           plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_beta, use_container_width=True)

    st.divider()

    # ── Download report ───────────────────────────────────────────────────────
    st.subheader("Export Risk Report")

    def build_report() -> bytes:
        buf = io.StringIO()
        buf.write("PORTFOLIO RISK REPORT\n")
        buf.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        buf.write("SUMMARY\n")
        summary_rows = {
            "Total Portfolio Value":    f"${total_value:,.2f}",
            "Total Unrealized P&L":     f"${total_pnl:,.2f}",
            "Total P&L %":              f"{total_pnl_pct:.2f}%",
            "Portfolio Beta":           f"{port_beta:.2f}",
            "Sharpe Ratio (1Y)":        _fmt(sharpe,  ".2f"),
            "Sortino Ratio (1Y)":       _fmt(sortino, ".2f"),
            "Calmar Ratio":             _fmt(calmar,  ".2f"),
            "Treynor Ratio":            _fmt(treynor, ".2f"),
            "Max Drawdown":             f"{max_dd * 100:.2f}%" if max_dd else "N/A",
            "Annualised Volatility":    f"{ann_vol * 100:.2f}%" if ann_vol else "N/A",
            "HHI Concentration":        f"{hhi:.3f}",
            "1-Day VaR (95%)":          f"${var_data['var_1d']:,.2f}" if var_data["var_1d"] else "N/A",
            "1-Day CVaR (95%)":         f"${cvar_data['cvar_1d']:,.2f}" if cvar_data["cvar_1d"] else "N/A",
            "5-Day VaR (95%)":          f"${var_data['var_5d']:,.2f}" if var_data["var_5d"] else "N/A",
            "Jensen's Alpha":           f"{bench_stats['alpha'] * 100:+.2f}%" if bench_stats["alpha"] else "N/A",
            "R² vs S&P 500":            f"{bench_stats['r2'] * 100:.1f}%" if bench_stats["r2"] else "N/A",
            "Tracking Error":           f"{bench_stats['tracking_error'] * 100:.2f}%" if bench_stats["tracking_error"] else "N/A",
            "Up Capture Ratio":         f"{up_cap:.1f}%" if up_cap else "N/A",
            "Down Capture Ratio":       f"{down_cap:.1f}%" if down_cap else "N/A",
            "Risk-Free Rate Used":      f"{risk_free_rate * 100:.2f}%",
        }
        pd.DataFrame.from_dict(summary_rows, orient="index", columns=["Value"]).to_csv(buf)

        buf.write("\nPOSITION DETAILS\n")
        positions.to_csv(buf, index=False)

        buf.write("\nCORRELATION MATRIX\n")
        if not corr.empty:
            corr.round(4).to_csv(buf)

        return buf.getvalue().encode()

    st.download_button(
        "⬇ Download Full Risk Report (CSV)",
        data=build_report(),
        file_name=f"portfolio_risk_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
