"""
Schrödinger's Macro Trend — Synthetic Data Generator
=====================================================
Generates realistic-looking ETF prices and macro state variables
for testing the full backtest pipeline when live data is unavailable.

The synthetic data exhibits:
- Correlated asset returns within asset classes
- Regime switching (bull/bear/crisis)
- Mean-reverting macro variables
- Realistic volatility levels per asset class
"""

import numpy as np
import pandas as pd
from pathlib import Path

import config as cfg

np.random.seed(42)


def generate_synthetic_data(
    start: str = "1993-01-04",
    end: str = "2025-09-26",
    freq: str = "W-FRI",
) -> dict:
    """
    Generate synthetic weekly data matching the structure expected by the backtest.

    Returns dict with:
        "etf_weekly"   : pd.DataFrame of ETF prices
        "etf_daily"    : pd.DataFrame of ETF prices (daily, derived from weekly)
        "macro_weekly"  : pd.DataFrame of macro state variables
        "macro_daily"   : pd.DataFrame of macro (daily, derived from weekly)
    """
    dates = pd.date_range(start=start, end=end, freq=freq)
    n = len(dates)
    print(f"[Synthetic] Generating {n} weekly observations from {start} to {end}")

    # ── Regime switching (hidden Markov-like) ──
    # States: 0=expansion, 1=slowdown, 2=crisis
    regime = np.zeros(n, dtype=int)
    transition = np.array([
        [0.96, 0.03, 0.01],  # expansion → expansion/slowdown/crisis
        [0.08, 0.85, 0.07],  # slowdown → expansion/slowdown/crisis
        [0.15, 0.20, 0.65],  # crisis → expansion/slowdown/crisis
    ])
    for i in range(1, n):
        regime[i] = np.random.choice(3, p=transition[regime[i - 1]])

    regime_series = pd.Series(regime, index=dates, name="regime")

    # ── Generate macro state variables ──
    macro = pd.DataFrame(index=dates)

    # S&P 500 (market) — level, not return
    sp_drift = np.where(regime == 0, 0.002, np.where(regime == 1, 0.0, -0.008))
    sp_vol = np.where(regime == 0, 0.02, np.where(regime == 1, 0.03, 0.06))
    sp_returns = sp_drift + sp_vol * np.random.randn(n)
    sp_price = 450 * np.exp(np.cumsum(sp_returns))
    macro["market"] = sp_price

    # Yield curve (10y - 2y spread, in percentage points)
    # Positive in expansion, flattening in slowdown, inverted in crisis
    yc_target = np.where(regime == 0, 1.5, np.where(regime == 1, 0.3, -0.5))
    yc = np.zeros(n)
    yc[0] = 1.0
    for i in range(1, n):
        yc[i] = yc[i - 1] + 0.05 * (yc_target[i] - yc[i - 1]) + 0.1 * np.random.randn()
    macro["yield_curve"] = yc

    # WTI oil price
    oil_target = np.where(regime == 0, 70, np.where(regime == 1, 55, 40))
    oil = np.zeros(n)
    oil[0] = 60
    for i in range(1, n):
        oil[i] = oil[i - 1] * np.exp(0.01 * (np.log(oil_target[i]) - np.log(oil[i - 1])) + 0.04 * np.random.randn())
    oil = np.maximum(oil, 10)
    macro["oil"] = oil

    # Copper price
    cu_target = np.where(regime == 0, 4.0, np.where(regime == 1, 3.2, 2.5))
    cu = np.zeros(n)
    cu[0] = 3.5
    for i in range(1, n):
        cu[i] = cu[i - 1] * np.exp(0.01 * (np.log(cu_target[i]) - np.log(cu[i - 1])) + 0.03 * np.random.randn())
    cu = np.maximum(cu, 1.0)
    macro["copper"] = cu

    # 3-month T-bill yield (monetary policy)
    tb_target = np.where(regime == 0, 3.5, np.where(regime == 1, 2.0, 0.5))
    tb = np.zeros(n)
    tb[0] = 2.0
    for i in range(1, n):
        tb[i] = tb[i - 1] + 0.02 * (tb_target[i] - tb[i - 1]) + 0.08 * np.random.randn()
    tb = np.clip(tb, 0.0, 8.0)
    macro["monetary"] = tb

    # VIX
    vix_base = np.where(regime == 0, 14, np.where(regime == 1, 22, 35))
    vix = np.zeros(n)
    vix[0] = 16
    for i in range(1, n):
        vix[i] = vix[i - 1] + 0.1 * (vix_base[i] - vix[i - 1]) + 2.0 * np.random.randn()
    vix = np.clip(vix, 9, 80)
    macro["volatility"] = vix

    # Stock-bond correlation (3-year rolling, already computed)
    # Positive in inflationary regimes, negative in deflationary
    sbc_target = np.where(regime == 0, -0.2, np.where(regime == 1, 0.0, -0.3))
    sbc = np.zeros(n)
    sbc[0] = -0.15
    for i in range(1, n):
        sbc[i] = sbc[i - 1] + 0.03 * (sbc_target[i] - sbc[i - 1]) + 0.02 * np.random.randn()
    sbc = np.clip(sbc, -0.6, 0.5)
    macro["stock_bond"] = sbc

    # ── Generate ETF prices ──
    # Return characteristics per ETF
    etf_params = {
        # ticker: (annual_drift, weekly_vol, beta_to_sp, regime_sensitivity)
        "SPY":  (0.10, 0.020, 1.0, 1.0),
        "IWM":  (0.09, 0.025, 1.2, 1.3),
        "QQQ":  (0.12, 0.025, 1.1, 1.2),
        "EFA":  (0.07, 0.022, 0.8, 0.9),
        "EEM":  (0.08, 0.030, 0.9, 1.4),
        "MTUM": (0.11, 0.022, 0.9, 0.8),
        "TLT":  (0.04, 0.018, -0.3, -0.5),
        "IEF":  (0.03, 0.008, -0.1, -0.2),
        "LQD":  (0.04, 0.010, 0.1, 0.3),
        "HYG":  (0.05, 0.012, 0.5, 0.8),
        "XLE":  (0.08, 0.030, 0.8, 1.5),
        "XLF":  (0.09, 0.025, 1.1, 1.4),
        "VNQ":  (0.07, 0.025, 0.7, 1.1),
        "XLK":  (0.12, 0.025, 1.0, 1.0),
        "DBC":  (0.03, 0.025, 0.3, 0.8),
        "GLD":  (0.06, 0.018, -0.1, -0.6),
        "USO":  (-0.02, 0.040, 0.3, 1.2),
        "UUP":  (0.01, 0.008, -0.1, -0.3),
    }

    # Different inception dates to mimic real ETFs
    inception_offsets = {
        "SPY": 0, "IWM": 0, "QQQ": 0, "EFA": 0, "EEM": 100,
        "MTUM": 500, "TLT": 50, "IEF": 50, "LQD": 50, "HYG": 300,
        "XLE": 0, "XLF": 0, "VNQ": 200, "XLK": 0,
        "DBC": 400, "GLD": 350, "USO": 400, "UUP": 450,
    }

    etf_prices = pd.DataFrame(index=dates)

    # Common market factor
    market_shocks = np.random.randn(n)

    for ticker, (drift, vol, beta, regime_sens) in etf_params.items():
        inception = inception_offsets.get(ticker, 0)

        # Regime-dependent drift adjustment
        drift_adj = np.where(regime == 0, drift / 52,
                    np.where(regime == 1, drift / 52 * 0.3,
                             -abs(drift) / 52 * regime_sens))

        # Returns = drift + beta * market + idiosyncratic
        idio_vol = vol * np.sqrt(1 - min(beta ** 2, 0.95))
        weekly_ret = drift_adj + beta * 0.02 * market_shocks + idio_vol * np.random.randn(n)

        price = 100 * np.exp(np.cumsum(weekly_ret))
        price_series = pd.Series(price, index=dates)

        # Set pre-inception to NaN
        if inception > 0:
            price_series.iloc[:inception] = np.nan

        etf_prices[ticker] = price_series

    # ── Create daily data (interpolated from weekly for vol calculation) ──
    daily_dates = pd.date_range(start=start, end=end, freq="B")  # business days
    etf_daily = etf_prices.reindex(daily_dates).interpolate(method="time")
    macro_daily = macro.reindex(daily_dates).interpolate(method="time")

    print(f"[Synthetic] ETF prices: {etf_prices.shape}")
    print(f"[Synthetic] Macro vars: {macro.shape}")
    print(f"[Synthetic] Regimes: expansion={np.mean(regime==0):.0%}, "
          f"slowdown={np.mean(regime==1):.0%}, crisis={np.mean(regime==2):.0%}")

    return {
        "etf_weekly": etf_prices,
        "etf_daily": etf_daily,
        "macro_weekly": macro,
        "macro_daily": macro_daily,
        "_regime": regime_series,  # for diagnostics
    }
