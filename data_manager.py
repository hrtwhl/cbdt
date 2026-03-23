"""
Schrödinger's Macro Trend — Data Manager
=========================================
Fetches ETF prices from Yahoo Finance and macro data from FRED.
Caches locally to avoid repeated downloads.
"""

import os
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────
# Yahoo Finance
# ─────────────────────────────────────────────────
def fetch_yahoo(tickers: list, start: str = "1990-01-01", end: str = None) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers from Yahoo Finance.
    Returns a DataFrame with DatetimeIndex (daily) and one column per ticker.
    """
    import yfinance as yf

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    print(f"[Yahoo] Downloading {len(tickers)} tickers from {start} to {end} ...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yf.download returns multi-level columns when >1 ticker
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    for t in tickers:
        if t in prices.columns:
            pct_na = prices[t].isna().mean()
            if pct_na > 0.5:
                warnings.warn(f"[Yahoo] {t}: {pct_na:.0%} missing — check ticker validity")

    print(f"[Yahoo] Got {len(prices)} daily observations, {prices.shape[1]} tickers")
    return prices


def fetch_yahoo_macro(start: str = "1990-01-01", end: str = None) -> pd.DataFrame:
    """Download macro-related series from Yahoo Finance."""
    macro_yahoo_tickers = []
    ticker_map = {}
    for var_name, spec in cfg.MACRO_VARIABLES.items():
        if spec["source"] == "yahoo":
            t = spec["ticker"]
            macro_yahoo_tickers.append(t)
            ticker_map[t] = var_name

    if not macro_yahoo_tickers:
        return pd.DataFrame()

    raw = fetch_yahoo(macro_yahoo_tickers, start=start, end=end)
    raw = raw.rename(columns=ticker_map)
    return raw


# ─────────────────────────────────────────────────
# FRED
# ─────────────────────────────────────────────────
def fetch_fred(series_ids: list, start: str = "1990-01-01", end: str = None) -> pd.DataFrame:
    """
    Download series from FRED using the fredapi package.
    Falls back to Yahoo if FRED API key is not set.
    """
    if cfg.FRED_API_KEY == "YOUR_FRED_API_KEY_HERE" or not cfg.FRED_API_KEY:
        print("[FRED] No API key set — falling back to Yahoo proxies.")
        return _fetch_fred_via_yahoo(series_ids, start, end)

    from fredapi import Fred
    fred = Fred(api_key=cfg.FRED_API_KEY)

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    frames = {}
    for sid in series_ids:
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            frames[sid] = s
            print(f"[FRED] {sid}: {len(s)} observations")
        except Exception as e:
            warnings.warn(f"[FRED] Failed to fetch {sid}: {e}")
            frames[sid] = pd.Series(dtype=float)

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def _fetch_fred_via_yahoo(series_ids: list, start: str, end: str) -> pd.DataFrame:
    """Approximate FRED series using Yahoo Finance tickers."""
    import yfinance as yf

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    frames = {}
    for sid in series_ids:
        if sid == "T10Y2Y":
            # Compute from 10y (^TNX) and 2y (^IRX is 13-week, not ideal)
            # Better: use DGS10 - DGS2 if available, or approximate
            tnx = yf.download("^TNX", start=start, end=end, auto_adjust=True, progress=False)
            # Yahoo ^TNX is 10-year yield × 10, so TNX/10 = yield in %
            # ^IRX is 13-week T-bill yield
            irx = yf.download("^IRX", start=start, end=end, auto_adjust=True, progress=False)
            if not tnx.empty and not irx.empty:
                y10 = tnx["Close"].squeeze() / 10 if "Close" in tnx.columns else tnx.iloc[:, 0] / 10
                y3m = irx["Close"].squeeze() / 10 if "Close" in irx.columns else irx.iloc[:, 0] / 10
                # NOTE: This is 10y - 3m, not 10y - 2y. Close enough as fallback.
                spread = y10 - y3m
                frames[sid] = spread
                print(f"[Yahoo fallback] {sid} approximated as 10y - 13w spread")
            else:
                warnings.warn(f"[Yahoo fallback] Could not approximate {sid}")
                frames[sid] = pd.Series(dtype=float)

        elif sid in cfg.FRED_YAHOO_FALLBACKS and cfg.FRED_YAHOO_FALLBACKS[sid]:
            yf_ticker = cfg.FRED_YAHOO_FALLBACKS[sid]
            raw = yf.download(yf_ticker, start=start, end=end, auto_adjust=True, progress=False)
            if not raw.empty:
                s = raw["Close"].squeeze() if "Close" in raw.columns else raw.iloc[:, 0]
                if sid == "DTB3":
                    # ^IRX returns yield in percent; FRED DTB3 is also in percent
                    pass
                frames[sid] = s
                print(f"[Yahoo fallback] {sid} → {yf_ticker}: {len(s)} obs")
            else:
                frames[sid] = pd.Series(dtype=float)
        else:
            warnings.warn(f"[Yahoo fallback] No fallback for {sid}")
            frames[sid] = pd.Series(dtype=float)

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def fetch_fred_macro(start: str = "1990-01-01", end: str = None) -> pd.DataFrame:
    """Download macro series needed from FRED."""
    fred_series = []
    series_map = {}
    for var_name, spec in cfg.MACRO_VARIABLES.items():
        if spec["source"] == "fred":
            sid = spec["series"]
            fred_series.append(sid)
            series_map[sid] = var_name

    if not fred_series:
        return pd.DataFrame()

    raw = fetch_fred(fred_series, start=start, end=end)
    raw = raw.rename(columns=series_map)
    return raw


# ─────────────────────────────────────────────────
# Resample to weekly (Friday close)
# ─────────────────────────────────────────────────
def to_weekly(df: pd.DataFrame, method: str = "last") -> pd.DataFrame:
    """Resample daily data to weekly frequency ending on Friday."""
    if method == "last":
        return df.resample("W-FRI").last()
    elif method == "mean":
        return df.resample("W-FRI").mean()
    else:
        raise ValueError(f"Unknown method: {method}")


# ─────────────────────────────────────────────────
# Compute stock-bond correlation
# ─────────────────────────────────────────────────
def compute_stock_bond_correlation(
    sp500_daily: pd.Series,
    bond_yield_daily: pd.Series,
    lookback_weeks: int = 156,
) -> pd.Series:
    """
    Rolling correlation between S&P 500 daily returns and 10-year bond returns.
    Bond returns approximated as: -duration × Δyield (duration ≈ 8 for 10y).

    Per Mulliner: 3-year rolling window on daily data, then map to weekly.
    """
    sp_ret = sp500_daily.pct_change()
    # Approximate bond return: -duration * change_in_yield / 100
    # Yield is in percentage points, so a 0.01 change = 1bp
    duration = 8.0
    bond_ret = -duration * bond_yield_daily.diff() / 100

    aligned = pd.DataFrame({"stock": sp_ret, "bond": bond_ret}).dropna()

    lookback_days = lookback_weeks * 5  # approximate
    corr = aligned["stock"].rolling(lookback_days, min_periods=lookback_days // 2).corr(
        aligned["bond"]
    )
    return corr


# ─────────────────────────────────────────────────
# Master data loader
# ─────────────────────────────────────────────────
def load_all_data(
    start: str = "1990-01-01",
    end: str = None,
    use_cache: bool = True,
) -> dict:
    """
    Load and assemble all data needed for the backtest.

    Returns a dict with:
        "etf_weekly"    : pd.DataFrame  — weekly adjusted close prices for ETFs
        "macro_weekly"  : pd.DataFrame  — weekly macro state variables (transformed)
        "etf_daily"     : pd.DataFrame  — daily prices (for vol calculation)
    """
    cache_file = CACHE_DIR / "all_data.pkl"
    if use_cache and cache_file.exists():
        print(f"[Cache] Loading from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # --- ETF prices ---
    etf_daily = fetch_yahoo(cfg.ETF_TICKERS, start=start, end=end)
    etf_weekly = to_weekly(etf_daily, method="last")

    # --- Macro: Yahoo-sourced (S&P 500, Copper, VIX) ---
    macro_yahoo_daily = fetch_yahoo_macro(start=start, end=end)

    # --- Macro: FRED-sourced (yield curve, oil, T-bill) ---
    macro_fred_daily = fetch_fred_macro(start=start, end=end)

    # Combine Yahoo + FRED macro into one daily frame
    macro_daily = pd.concat([macro_yahoo_daily, macro_fred_daily], axis=1)
    macro_daily = macro_daily.sort_index()

    # Forward-fill FRED data (often has gaps on weekends/holidays)
    macro_daily = macro_daily.ffill()

    # --- Stock-bond correlation ---
    sp500_daily = macro_daily["market"] if "market" in macro_daily.columns else None
    # For bond yield, we need the 10y yield daily. Try FRED first.
    bond_yield_daily = None
    if cfg.FRED_API_KEY and cfg.FRED_API_KEY != "YOUR_FRED_API_KEY_HERE":
        try:
            from fredapi import Fred
            fred = Fred(api_key=cfg.FRED_API_KEY)
            bond_yield_daily = fred.get_series("DGS10", observation_start=start)
            bond_yield_daily.index = pd.to_datetime(bond_yield_daily.index)
            bond_yield_daily = bond_yield_daily.astype(float)
            print(f"[FRED] DGS10 (10y yield): {len(bond_yield_daily)} obs")
        except Exception:
            pass

    if bond_yield_daily is None:
        # Fallback: Yahoo ^TNX
        import yfinance as yf
        tnx = yf.download("^TNX", start=start, end=end, auto_adjust=True, progress=False)
        if not tnx.empty:
            bond_yield_daily = (tnx["Close"].squeeze() if "Close" in tnx.columns else tnx.iloc[:, 0]) / 10
            print(f"[Yahoo fallback] 10y yield from ^TNX: {len(bond_yield_daily)} obs")

    if sp500_daily is not None and bond_yield_daily is not None:
        sb_corr = compute_stock_bond_correlation(
            sp500_daily, bond_yield_daily, lookback_weeks=cfg.STOCK_BOND_CORR_WINDOW
        )
        macro_daily["stock_bond"] = sb_corr
        print(f"[Computed] Stock-bond correlation: {sb_corr.notna().sum()} obs")

    # Resample macro to weekly
    macro_weekly = to_weekly(macro_daily, method="last")

    result = {
        "etf_weekly": etf_weekly,
        "etf_daily": etf_daily,
        "macro_weekly": macro_weekly,
        "macro_daily": macro_daily,
    }

    # Cache
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    print(f"[Cache] Saved to {cache_file}")

    return result
