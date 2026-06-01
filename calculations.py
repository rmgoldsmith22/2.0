from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ── Internal helper ───────────────────────────────────────────────────────────

def _portfolio_daily_returns(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
) -> pd.Series | None:
    """Value-weighted daily return series. Shared by all risk metric functions."""
    if price_df.empty or positions_df.empty:
        return None
    returns = price_df.pct_change().dropna()
    tickers = [t for t in positions_df["Ticker"].tolist() if t in returns.columns]
    if not tickers:
        return None
    weights = positions_df.set_index("Ticker").loc[tickers, "Weight"] / 100
    return returns[tickers].dot(weights)


# ── Sector mapping for ETFs (yfinance does not return sector for funds) ───────

ETF_SECTOR_MAP: dict[str, str] = {
    "VOO": "Broad US Equity",   "VTI": "Broad US Equity",
    "SPY": "Broad US Equity",   "IVV": "Broad US Equity",
    "QQQM": "Technology",       "QQQ": "Technology",
    "XLK": "Technology",        "VGT": "Technology",
    "VDC": "Consumer Staples",  "XLP": "Consumer Staples",
    "XLF": "Financials",        "VFH": "Financials",
    "XLE": "Energy",            "VDE": "Energy",
    "XLV": "Health Care",       "VHT": "Health Care",
    "XLI": "Industrials",       "VIS": "Industrials",
    "VFMO": "US Momentum Factor",
    "MTUM": "US Momentum Factor",
    "BTC": "Digital Assets",    "GBTC": "Digital Assets",
    "ETH": "Digital Assets",    "ETHE": "Digital Assets",
    "ACWI": "International Equity", "URTH": "International Equity",
    "VEA": "International Equity",  "EFA": "International Equity",
    "EEM": "Emerging Markets",      "VWO": "Emerging Markets",
    "TLT": "Fixed Income",      "AGG": "Fixed Income",
    "BND": "Fixed Income",      "SHY": "Fixed Income",
    "GLD": "Commodities",       "IAU": "Commodities",
    "BPTRX": "Multi-Cap Growth",
    "ARKK": "Innovation",
}


# ── Position building ─────────────────────────────────────────────────────────

def build_positions(portfolio_df: pd.DataFrame, ticker_info: dict) -> pd.DataFrame:
    rows = []
    for _, row in portfolio_df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        try:
            shares = float(row["Shares"])
            avg_cost = float(row["Avg_Cost"])
        except (ValueError, TypeError):
            continue

        info = ticker_info.get(ticker, {})
        current_price = info.get("current_price")
        if current_price is None:
            continue

        value = shares * current_price
        cost_basis = shares * avg_cost
        pnl = value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0.0
        beta = info.get("beta") if info.get("beta") is not None else 1.0
        sector = ETF_SECTOR_MAP.get(ticker) or info.get("sector") or "Other"

        rows.append({
            "Ticker": ticker,
            "Shares": shares,
            "Avg_Cost": avg_cost,
            "Current_Price": current_price,
            "Value": value,
            "Cost_Basis": cost_basis,
            "PnL": pnl,
            "PnL_Pct": pnl_pct,
            "Beta": float(beta),
            "Market_Cap": info.get("market_cap"),
            "Sector": sector,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    total_value = df["Value"].sum()
    df["Weight"] = (df["Value"] / total_value * 100).round(4)
    df["Beta_Contribution"] = (df["Beta"] * df["Weight"] / 100).round(4)
    return df.reset_index(drop=True)


# ── Core portfolio metrics ────────────────────────────────────────────────────

def calc_portfolio_beta(positions_df: pd.DataFrame) -> float:
    return float(positions_df["Beta_Contribution"].sum())


def calc_correlation_matrix(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty or price_df.shape[1] < 2:
        return pd.DataFrame()
    return price_df.pct_change().dropna().corr()


def calc_portfolio_cumulative(
    price_df: pd.DataFrame, positions_df: pd.DataFrame
) -> pd.Series | None:
    """Cumulative return series for constant-weight backtest, starting at 1.0."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None:
        return None
    return (1 + port_returns).cumprod()


def calc_hhi(positions_df: pd.DataFrame) -> float:
    """Herfindahl-Hirschman Index. 1/n = equal weight; 1.0 = single position."""
    weights = positions_df["Weight"] / 100
    return float((weights ** 2).sum())


# ── Return distribution / tail risk ──────────────────────────────────────────

def calc_var(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    confidence: float = 0.95,
    lookback: int = 252,
) -> dict:
    empty = {"var_1d": None, "var_5d": None, "var_1d_pct": None, "var_5d_pct": None}
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None:
        return empty
    port_returns = port_returns.tail(lookback)
    if len(port_returns) < 30:
        return empty
    var_1d_pct = float(np.percentile(port_returns, (1 - confidence) * 100))
    var_5d_pct = var_1d_pct * np.sqrt(5)
    total_value = float(positions_df["Value"].sum())
    return {
        "var_1d": abs(var_1d_pct) * total_value,
        "var_5d": abs(var_5d_pct) * total_value,
        "var_1d_pct": abs(var_1d_pct) * 100,
        "var_5d_pct": abs(var_5d_pct) * 100,
    }


def calc_cvar(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    confidence: float = 0.95,
    lookback: int = 252,
) -> dict:
    """Expected Shortfall: average loss beyond the VaR threshold."""
    empty = {"cvar_1d": None, "cvar_1d_pct": None}
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None:
        return empty
    port_returns = port_returns.tail(lookback)
    if len(port_returns) < 30:
        return empty
    threshold = np.percentile(port_returns, (1 - confidence) * 100)
    tail = port_returns[port_returns <= threshold]
    if tail.empty:
        return empty
    cvar_pct = float(tail.mean())
    total_value = float(positions_df["Value"].sum())
    return {
        "cvar_1d": abs(cvar_pct) * total_value,
        "cvar_1d_pct": abs(cvar_pct) * 100,
    }


def calc_max_drawdown(
    price_df: pd.DataFrame, positions_df: pd.DataFrame
) -> tuple[float | None, pd.Series | None]:
    """Returns (max drawdown fraction, full drawdown time series)."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 2:
        return None, None
    cum = (1 + port_returns).cumprod()
    drawdown = cum / cum.expanding().max() - 1
    return float(drawdown.min()), drawdown


def calc_annualized_volatility(
    price_df: pd.DataFrame, positions_df: pd.DataFrame
) -> float | None:
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 2:
        return None
    return float(port_returns.std() * np.sqrt(252))


# ── Risk-adjusted return ratios ───────────────────────────────────────────────

def calc_sharpe_ratio(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> float | None:
    """(Ann. excess return) / (ann. total volatility)."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return None
    excess = port_returns - risk_free_rate / 252
    std = excess.std()
    return float(excess.mean() / std * np.sqrt(252)) if std else None


def calc_sortino_ratio(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> float | None:
    """(Ann. excess return) / (ann. downside deviation). Only penalises losses."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return None
    daily_mar = risk_free_rate / 252
    ann_return = (1 + port_returns.mean()) ** 252 - 1
    downside = port_returns[port_returns < daily_mar] - daily_mar
    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
    return float((ann_return - risk_free_rate) / downside_dev) if downside_dev else None


def calc_calmar_ratio(
    price_df: pd.DataFrame, positions_df: pd.DataFrame
) -> float | None:
    """CAGR / |max drawdown|. How much return earned per unit of worst historical loss."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return None
    n_years = len(port_returns) / 252
    total_return = (1 + port_returns).prod() - 1
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    cum = (1 + port_returns).cumprod()
    max_dd = float((cum / cum.expanding().max() - 1).min())
    return float(cagr / abs(max_dd)) if max_dd != 0 else None


def calc_treynor_ratio(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    port_beta: float,
    risk_free_rate: float = 0.05,
) -> float | None:
    """(Ann. excess return) / beta. Focuses on systematic risk only."""
    if port_beta == 0:
        return None
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return None
    ann_return = (1 + port_returns.mean()) ** 252 - 1
    return float((ann_return - risk_free_rate) / port_beta)


# ── Benchmark-relative metrics ────────────────────────────────────────────────

def calc_alpha_r2_tracking_error(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> dict:
    """Jensen's alpha (annualised), R², and annualised tracking error vs benchmark."""
    empty = {"alpha": None, "r2": None, "tracking_error": None}
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None:
        return empty
    aligned = pd.DataFrame({"port": port_returns, "bench": benchmark_returns}).dropna()
    if len(aligned) < 30:
        return empty
    daily_rf = risk_free_rate / 252
    _, intercept, r_value, _, _ = stats.linregress(
        aligned["bench"] - daily_rf, aligned["port"] - daily_rf
    )
    alpha_ann = (1 + intercept) ** 252 - 1
    tracking_error = (aligned["port"] - aligned["bench"]).std() * np.sqrt(252)
    return {
        "alpha": float(alpha_ann),
        "r2": float(r_value ** 2),
        "tracking_error": float(tracking_error),
    }


def calc_capture_ratios(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> tuple[float | None, float | None]:
    """Up and down capture ratios vs benchmark (annualised geometric method)."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None:
        return None, None
    aligned = pd.DataFrame({"port": port_returns, "bench": benchmark_returns}).dropna()
    up = aligned[aligned["bench"] > 0]
    down = aligned[aligned["bench"] < 0]

    def geo_ann(series: pd.Series, n: int) -> float:
        return (1 + series).prod() ** (252 / n) - 1

    up_capture = None
    if len(up) >= 10:
        bench_up = geo_ann(up["bench"], len(up))
        if bench_up != 0:
            up_capture = geo_ann(up["port"], len(up)) / bench_up * 100

    down_capture = None
    if len(down) >= 10:
        bench_down = geo_ann(down["bench"], len(down))
        if bench_down != 0:
            down_capture = geo_ann(down["port"], len(down)) / bench_down * 100

    return up_capture, down_capture


# ── Rolling metrics ───────────────────────────────────────────────────────────

def calc_rolling_metrics(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    window: int = 63,
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """Rolling Sharpe ratio and annualised volatility over a sliding window (~63 days = 1 quarter)."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < window + 5:
        return pd.DataFrame()
    daily_rf = risk_free_rate / 252
    roll_std = port_returns.rolling(window).std()
    roll_sharpe = (port_returns.rolling(window).mean() - daily_rf) / roll_std * np.sqrt(252)
    roll_vol = roll_std * np.sqrt(252) * 100
    return pd.DataFrame({
        "Rolling Sharpe": roll_sharpe,
        "Rolling Volatility (%)": roll_vol,
    }).dropna()


# ── Monte Carlo simulation ────────────────────────────────────────────────────

def calc_monte_carlo(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    initial_value: float,
    years: int = 20,
    simulations: int = 1000,
    monthly_contribution: float = 0.0,
) -> pd.DataFrame | None:
    """
    Simulate portfolio value paths using historical mean/vol.
    Returns DataFrame with percentile bands and a 'year' column.
    """
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return None

    daily_mean = float(port_returns.mean())
    daily_std = float(port_returns.std())
    n_days = years * 252

    rng = np.random.default_rng(42)
    rand_rets = rng.normal(daily_mean, daily_std, (n_days, simulations))

    paths = np.empty((n_days + 1, simulations))
    paths[0] = initial_value
    monthly_days = 21
    for t in range(1, n_days + 1):
        paths[t] = paths[t - 1] * (1 + rand_rets[t - 1])
        if monthly_contribution > 0 and t % monthly_days == 0:
            paths[t] += monthly_contribution

    pcts = [5, 25, 50, 75, 95]
    data = np.percentile(paths, pcts, axis=1).T
    df = pd.DataFrame(data, columns=[f"p{p}" for p in pcts])
    df["year"] = np.linspace(0, years, n_days + 1)
    return df
