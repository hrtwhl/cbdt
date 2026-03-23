"""
Schrödinger's Macro Trend — Core Model
=======================================
Feature engineering, analogue matching, Alive/Dead classification,
and portfolio construction.

Architecture mirrors:
- Mulliner et al. "Regimes" (2025) for macro state construction
- Heiden "Schrödinger's Macro Lens" for asset-level logic
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import config as cfg


# ══════════════════════════════════════════════════
# 1. FEATURE ENGINEERING
# ══════════════════════════════════════════════════

def compute_price_features(prices_weekly: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    For each ETF, compute per-asset price features (weekly frequency).

    Returns dict: { ticker: DataFrame with columns for each feature }
    """
    all_features = {}

    for ticker in prices_weekly.columns:
        p = prices_weekly[ticker].dropna()
        if len(p) < 52:
            continue

        feats = pd.DataFrame(index=p.index)

        # Momentum: log returns over various windows
        for w in cfg.PRICE_MOMENTUM_WINDOWS:
            feats[f"mom_{w}w"] = np.log(p / p.shift(w))

        # Realized volatility: annualized std of weekly log returns
        weekly_ret = np.log(p / p.shift(1))
        for w in cfg.PRICE_VOLATILITY_WINDOWS:
            feats[f"vol_{w}w"] = weekly_ret.rolling(w).std() * np.sqrt(52)

        # Drawdown from rolling high
        rolling_max = p.rolling(cfg.PRICE_DRAWDOWN_WINDOW, min_periods=1).max()
        feats["drawdown"] = (p - rolling_max) / rolling_max

        all_features[ticker] = feats

    return all_features


def compute_macro_features(macro_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Transform macro state variables following Mulliner et al.:
    1. 52-week (≈12-month) change
    2. Z-score over rolling 520 weeks (≈10 years)
    3. Winsorize at ±3

    Returns DataFrame with one column per macro variable, weekly frequency.
    """
    change_window = cfg.MACRO_CHANGE_WINDOW
    zscore_window = cfg.MACRO_ZSCORE_WINDOW
    winsorize = cfg.MACRO_WINSORIZE_LIMIT

    # Determine which macro columns are available
    expected_vars = [name for name, spec in cfg.MACRO_VARIABLES.items()
                     if name in macro_weekly.columns]

    if not expected_vars:
        warnings.warn("No macro variables found in data!")
        return pd.DataFrame()

    macro_feats = pd.DataFrame(index=macro_weekly.index)

    for var in expected_vars:
        series = macro_weekly[var].astype(float)

        # Step 1: 52-week difference (≈12-month change)
        # For the yield curve / rates, this is a level difference
        # For prices (market, oil, copper), use log difference for scale invariance
        if var in ("market", "oil", "copper"):
            # Use log for price series
            diff = np.log(series / series.shift(change_window))
        else:
            # Level difference for rates, spreads, VIX, correlation
            diff = series - series.shift(change_window)

        # Step 2: Rolling z-score
        roll_mean = diff.rolling(zscore_window, min_periods=zscore_window // 2).mean()
        roll_std = diff.rolling(zscore_window, min_periods=zscore_window // 2).std()
        z = (diff - roll_mean) / roll_std.replace(0, np.nan)

        # Step 3: Winsorize at ±3
        z = z.clip(-winsorize, winsorize)

        macro_feats[f"macro_{var}"] = z

    return macro_feats


def build_feature_matrix(
    ticker: str,
    price_features: Dict[str, pd.DataFrame],
    macro_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine price features for a specific asset with global macro features.
    Standardize all features to zero mean / unit variance before combining.

    Returns a DataFrame where each row is a week and each column is a feature.
    """
    if ticker not in price_features:
        return pd.DataFrame()

    pf = price_features[ticker].copy()
    mf = macro_features.copy()

    # Align on common dates
    common_idx = pf.index.intersection(mf.index)
    if len(common_idx) < cfg.MIN_ANALOGUES * 2:
        return pd.DataFrame()

    pf = pf.loc[common_idx]
    mf = mf.loc[common_idx]

    # Standardize price features (rolling z-score to avoid look-ahead)
    pf_std = _rolling_standardize(pf, window=cfg.MACRO_ZSCORE_WINDOW)

    # Macro features are already z-scored & winsorized by compute_macro_features
    # (following Mulliner et al.) — do NOT re-standardize.
    mf_std = mf

    # Weight: 60% price, 40% macro (Heiden)
    n_price = pf_std.shape[1]
    n_macro = mf_std.shape[1]

    if n_price > 0 and n_macro > 0:
        # Scale so that combined squared distances reflect the 60/40 split.
        # Each price feature contributes PRICE_WEIGHT / n_price to total variance;
        # each macro feature contributes MACRO_WEIGHT / n_macro.
        price_scale = np.sqrt(cfg.PRICE_WEIGHT / n_price)
        macro_scale = np.sqrt(cfg.MACRO_WEIGHT / n_macro)
        pf_std = pf_std * price_scale
        mf_std = mf_std * macro_scale

    combined = pd.concat([pf_std, mf_std], axis=1)
    return combined.dropna()


def _rolling_standardize(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling z-score standardization (no look-ahead)."""
    roll_mean = df.rolling(window, min_periods=window // 4).mean()
    roll_std = df.rolling(window, min_periods=window // 4).std()
    standardized = (df - roll_mean) / roll_std.replace(0, np.nan)
    return standardized.clip(-cfg.MACRO_WINSORIZE_LIMIT, cfg.MACRO_WINSORIZE_LIMIT)


# ══════════════════════════════════════════════════
# 2. ANALOGUE ENGINE
# ══════════════════════════════════════════════════

def find_analogues(
    feature_matrix: pd.DataFrame,
    target_idx: int,
    exclusion_window: int = None,
    top_pct: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the most similar historical weeks to the target week.

    Parameters:
    - feature_matrix: DataFrame (weeks × features)
    - target_idx: integer index of the target week in the DataFrame
    - exclusion_window: exclude this many recent weeks before target
    - top_pct: fraction of history to keep as analogues

    Returns:
    - analogue_indices: integer indices into feature_matrix of similar weeks
    - distances: corresponding distances
    """
    if exclusion_window is None:
        exclusion_window = cfg.EXCLUSION_WINDOW
    if top_pct is None:
        top_pct = cfg.ANALOGUE_PCT

    values = feature_matrix.values
    target_vec = values[target_idx].reshape(1, -1)

    # If target week has any NaN features, we can't compute distances
    if np.isnan(target_vec).any():
        return np.array([]), np.array([])

    # Candidate pool: everything before (target - exclusion_window)
    cutoff = target_idx - exclusion_window
    if cutoff < 1:
        return np.array([]), np.array([])

    candidates = values[:cutoff]

    # Only keep candidates with no NaN features
    valid_mask = ~np.isnan(candidates).any(axis=1)
    if valid_mask.sum() < cfg.MIN_ANALOGUES:
        return np.array([]), np.array([])

    valid_indices = np.where(valid_mask)[0]
    valid_candidates = candidates[valid_mask]

    # Euclidean distance
    dists = cdist(target_vec, valid_candidates, metric="euclidean").flatten()

    # Select top_pct most similar (smallest distance)
    n_select = max(cfg.MIN_ANALOGUES, int(len(dists) * top_pct))
    n_select = min(n_select, len(dists))

    sort_order = np.argsort(dists)[:n_select]

    analogue_indices = valid_indices[sort_order]
    analogue_dists = dists[sort_order]

    return analogue_indices, analogue_dists


def get_forward_returns(
    prices_weekly: pd.Series,
    feature_dates: pd.DatetimeIndex,
    analogue_indices: np.ndarray,
    horizons: List[int] = None,
) -> Dict[int, np.ndarray]:
    """
    For each analogue date, compute forward returns at each horizon.

    Returns: { horizon_weeks: array of forward returns }
    """
    if horizons is None:
        horizons = cfg.FORWARD_HORIZONS

    prices = prices_weekly.dropna()
    results = {}

    for h in horizons:
        fwd_rets = []
        for idx in analogue_indices:
            if idx >= len(feature_dates):
                continue
            date = feature_dates[idx]
            # Find the price on this date and h weeks later
            date_loc = prices.index.searchsorted(date)
            future_loc = date_loc + h

            if future_loc < len(prices) and date_loc < len(prices):
                p_now = prices.iloc[date_loc]
                p_future = prices.iloc[future_loc]
                if p_now > 0 and not np.isnan(p_now) and not np.isnan(p_future):
                    fwd_rets.append(p_future / p_now - 1)

        results[h] = np.array(fwd_rets) if fwd_rets else np.array([])

    return results


# ══════════════════════════════════════════════════
# 3. ALIVE / DEAD CLASSIFICATION
# ══════════════════════════════════════════════════

def classify_asset(
    forward_returns: Dict[int, np.ndarray],
    horizon_weights: List[float] = None,
) -> Dict:
    """
    Classify an asset as Alive, Dead, or Ambiguous based on analogue evidence.

    Returns dict with:
        "status": "alive" | "dead" | "ambiguous"
        "blended_return": float (weighted expected return)
        "confidence": float (blended t-stat)
        "hit_rate": float (fraction of positive analogues)
        "n_analogues": int
        "detail": dict per horizon
    """
    if horizon_weights is None:
        horizon_weights = cfg.HORIZON_WEIGHTS

    horizons = cfg.FORWARD_HORIZONS
    detail = {}
    blended_return = 0.0
    blended_tstat = 0.0
    blended_hit = 0.0
    total_weight = 0.0
    total_analogues = 0

    for h, w in zip(horizons, horizon_weights):
        rets = forward_returns.get(h, np.array([]))
        if len(rets) < cfg.ALIVE_MIN_ANALOGUES:
            detail[h] = {"n": len(rets), "mean": np.nan, "tstat": np.nan, "hit": np.nan}
            continue

        mean_ret = np.mean(rets)
        std_ret = np.std(rets, ddof=1)
        tstat = mean_ret / (std_ret / np.sqrt(len(rets))) if std_ret > 0 else 0
        hit_rate = np.mean(rets > 0)

        detail[h] = {
            "n": len(rets),
            "mean": mean_ret,
            "std": std_ret,
            "tstat": tstat,
            "hit": hit_rate,
            "p20": np.percentile(rets, 20) if len(rets) > 0 else np.nan,
            "p80": np.percentile(rets, 80) if len(rets) > 0 else np.nan,
        }

        blended_return += w * mean_ret
        blended_tstat += w * tstat
        blended_hit += w * hit_rate
        total_weight += w
        total_analogues = max(total_analogues, len(rets))

    # Normalize by total weight used (in case some horizons were skipped)
    if total_weight > 0:
        blended_return /= total_weight
        blended_tstat /= total_weight
        blended_hit /= total_weight
    else:
        return {
            "status": "dead",
            "blended_return": 0.0,
            "confidence": 0.0,
            "hit_rate": 0.0,
            "n_analogues": 0,
            "detail": detail,
        }

    # Classification logic
    alive_conditions = (
        total_analogues >= cfg.ALIVE_MIN_ANALOGUES
        and blended_hit >= cfg.ALIVE_HIT_RATE
        and blended_return >= cfg.ALIVE_MIN_RETURN
        and blended_tstat >= cfg.ALIVE_MIN_CONFIDENCE
    )

    # Check for ambiguity: one horizon bullish, another bearish
    horizon_signs = []
    for h in horizons:
        if h in detail and not np.isnan(detail[h].get("mean", np.nan)):
            horizon_signs.append(np.sign(detail[h]["mean"]))

    ambiguous = len(set(horizon_signs)) > 1  # mixed signs across horizons

    if alive_conditions:
        status = "alive"
    elif ambiguous and total_analogues >= cfg.ALIVE_MIN_ANALOGUES:
        status = "ambiguous"
    else:
        status = "dead"

    return {
        "status": status,
        "blended_return": blended_return,
        "confidence": blended_tstat,
        "hit_rate": blended_hit,
        "n_analogues": total_analogues,
        "detail": detail,
    }


# ══════════════════════════════════════════════════
# 4. PORTFOLIO CONSTRUCTION
# ══════════════════════════════════════════════════

def construct_portfolio(
    classifications: Dict[str, Dict],
    etf_weekly_prices: pd.DataFrame,
    current_date: pd.Timestamp,
    vol_lookback: int = None,
) -> Dict[str, float]:
    """
    Build the portfolio for a given week based on Alive/Dead classifications.

    Steps:
    1. Filter to Alive assets
    2. Rank by signal strength (blended_return / recent_vol)
    3. Apply logit-power transform for sizing
    4. Cap positions, normalize to 100%
    5. Apply volatility targeting

    Returns: dict { ticker: weight }
    """
    if vol_lookback is None:
        vol_lookback = cfg.VOL_LOOKBACK

    # Step 1: Alive assets
    alive_assets = {
        ticker: info for ticker, info in classifications.items()
        if info["status"] == "alive" and info["blended_return"] > 0
    }

    if not alive_assets:
        return {}  # All cash

    # Step 2: Signal strength = blended_return
    # (Heiden: "Alive assets are ranked by signal strength")
    signals = {}
    for ticker, info in alive_assets.items():
        sig = info["blended_return"]
        # Optionally scale by inverse of recent volatility for risk-parity flavor
        # But Heiden seems to rank purely by expected return, so we use that
        signals[ticker] = max(sig, 0.0)

    if not signals or sum(signals.values()) == 0:
        return {}

    # Step 3: Logit-power transform to emphasize strong views
    # Transform: w_i = signal_i^p / sum(signal_j^p)
    power = cfg.LOGIT_POWER
    raw_weights = {t: s ** power for t, s in signals.items()}
    total = sum(raw_weights.values())

    if total == 0:
        return {}

    weights = {t: w / total for t, w in raw_weights.items()}

    # Step 4: Position caps
    capped = False
    for _ in range(10):  # iterative capping
        excess = 0.0
        n_uncapped = 0
        for t in weights:
            if weights[t] > cfg.MAX_POSITION_WEIGHT:
                excess += weights[t] - cfg.MAX_POSITION_WEIGHT
                weights[t] = cfg.MAX_POSITION_WEIGHT
                capped = True
            else:
                n_uncapped += 1

        if excess > 0 and n_uncapped > 0:
            uncapped_tickers = [t for t in weights if weights[t] < cfg.MAX_POSITION_WEIGHT]
            uncapped_total = sum(weights[t] for t in uncapped_tickers)
            if uncapped_total > 0:
                for t in uncapped_tickers:
                    weights[t] += excess * (weights[t] / uncapped_total)
        else:
            break

    # Remove tiny positions
    weights = {t: w for t, w in weights.items() if w >= cfg.MIN_POSITION_WEIGHT}

    # Renormalize
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}

    # Step 5: Volatility targeting
    # Estimate portfolio volatility from recent data
    vol_scale = _compute_vol_scale(weights, etf_weekly_prices, current_date, vol_lookback)
    # Scale weights by vol_scale (capped for safety)
    vol_scale = min(vol_scale, cfg.VOL_SCALE_CAP)

    weights = {t: w * vol_scale for t, w in weights.items()}

    # If total > 1, we'd be levered; for long-only ETF, cap at 1.0
    total = sum(weights.values())
    if total > 1.0:
        weights = {t: w / total for t, w in weights.items()}

    return weights


def _compute_vol_scale(
    weights: Dict[str, float],
    prices_weekly: pd.DataFrame,
    current_date: pd.Timestamp,
    lookback: int,
) -> float:
    """
    Compute volatility scaling factor to target cfg.TARGET_VOL.
    Uses recent portfolio returns to estimate annualized vol.
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])

    # Get recent weekly returns
    available = [t for t in tickers if t in prices_weekly.columns]
    if len(available) < 2:
        return 1.0

    loc = prices_weekly.index.searchsorted(current_date)
    start = max(0, loc - lookback)
    recent = prices_weekly[available].iloc[start:loc]

    if len(recent) < 12:
        return 1.0

    rets = recent.pct_change().dropna()
    if len(rets) < 8:
        return 1.0

    # Portfolio returns with current weights
    w_vec = np.array([weights.get(t, 0) for t in available])
    w_vec = w_vec / w_vec.sum() if w_vec.sum() > 0 else w_vec

    port_rets = rets.values @ w_vec
    port_vol_annual = np.std(port_rets) * np.sqrt(52)

    if port_vol_annual > 0.01:
        return cfg.TARGET_VOL / port_vol_annual
    else:
        return 1.0


# ══════════════════════════════════════════════════
# 5. ANALOGUE PATH EXTRACTION (for charts)
# ══════════════════════════════════════════════════

def extract_analogue_paths(
    prices_weekly: pd.Series,
    feature_dates: pd.DatetimeIndex,
    analogue_indices: np.ndarray,
    lookback_weeks: int = 26,
    forward_weeks: int = 26,
) -> pd.DataFrame:
    """
    Extract standardized price paths for analogue visualization.
    Each path is rebased to 0 at the analogue date.

    Returns DataFrame: columns are analogue dates, rows are relative weeks.
    """
    prices = prices_weekly.dropna()
    paths = {}

    for idx in analogue_indices:
        if idx >= len(feature_dates):
            continue
        date = feature_dates[idx]
        date_loc = prices.index.searchsorted(date)

        start_loc = date_loc - lookback_weeks
        end_loc = date_loc + forward_weeks

        if start_loc < 0 or end_loc >= len(prices):
            continue

        segment = prices.iloc[start_loc : end_loc + 1]
        # Rebase to analogue date = 0
        base_price = prices.iloc[date_loc]
        if base_price > 0:
            rebased = segment / base_price - 1
            # Create relative week index
            rel_weeks = np.arange(-lookback_weeks, forward_weeks + 1)
            if len(rebased) == len(rel_weeks):
                paths[date] = pd.Series(rebased.values, index=rel_weeks)

    return pd.DataFrame(paths)
