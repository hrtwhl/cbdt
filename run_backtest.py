"""
Schrödinger's Macro Trend — Backtest Runner
=============================================
Runs the full backtest, computes portfolio metrics, and generates
publication-quality charts matching Heiden's Macro Lens style.

Usage:
    python run_backtest.py

Prerequisites:
    pip install yfinance fredapi numpy pandas scipy matplotlib seaborn
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import config as cfg
import data_manager as dm
import schrodinger as sch

warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════

def run_backtest() -> dict:
    """
    Run the full weekly backtest of Schrödinger's ETF strategy.

    Returns a dict with:
        "equity_curve": pd.Series
        "weights_history": pd.DataFrame
        "classifications_history": list of dicts
        "benchmark": pd.Series (SPY buy-and-hold)
    """
    print("=" * 60)
    print("  Schrödinger's Macro Trend — ETF Backtest")
    print("=" * 60)

    # ── 1. Load data ──
    print("\n[1/5] Loading data...")
    try:
        data = dm.load_all_data(start="1993-01-01", end=cfg.BACKTEST_END, use_cache=True)
        print("  Using LIVE market data.")
    except Exception as e:
        print(f"  Live data unavailable ({e}). Using synthetic data.")
        import synthetic_data as sd
        data = sd.generate_synthetic_data()

    etf_weekly = data["etf_weekly"]
    macro_weekly = data["macro_weekly"]

    print(f"  ETF weekly: {etf_weekly.shape[0]} weeks × {etf_weekly.shape[1]} tickers")
    print(f"  Macro weekly: {macro_weekly.shape[0]} weeks × {macro_weekly.shape[1]} variables")

    # Show data availability per ticker
    print("\n  Data availability:")
    for t in cfg.ETF_TICKERS:
        if t in etf_weekly.columns:
            first = etf_weekly[t].dropna().index[0].strftime("%Y-%m-%d")
            last = etf_weekly[t].dropna().index[-1].strftime("%Y-%m-%d")
            n = etf_weekly[t].dropna().shape[0]
            print(f"    {t:6s}: {first} → {last}  ({n} weeks)")
        else:
            print(f"    {t:6s}: NOT AVAILABLE")

    # ── 2. Compute features ──
    print("\n[2/5] Computing features...")
    price_features = sch.compute_price_features(etf_weekly)
    macro_features = sch.compute_macro_features(macro_weekly)
    print(f"  Price features computed for {len(price_features)} tickers")
    print(f"  Macro features: {macro_features.shape[1]} variables, {macro_features.dropna().shape[0]} valid weeks")

    # ── 3. Determine backtest range ──
    # Find earliest date where all critical features are available
    # Need: macro features + enough history for analogues
    macro_valid_start = macro_features.dropna().index[0] if not macro_features.empty else etf_weekly.index[0]

    bt_start = pd.Timestamp(cfg.BACKTEST_START)
    # Ensure we have enough warmup
    actual_start = max(bt_start, macro_valid_start)

    bt_end = pd.Timestamp(cfg.BACKTEST_END) if cfg.BACKTEST_END else etf_weekly.index[-1]

    # Get the weekly dates in the backtest range
    bt_dates = etf_weekly.index[(etf_weekly.index >= actual_start) & (etf_weekly.index <= bt_end)]
    print(f"\n[3/5] Backtest range: {bt_dates[0].strftime('%Y-%m-%d')} → {bt_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  {len(bt_dates)} rebalancing periods")

    # ── 4. Run weekly backtest loop ──
    print("\n[4/5] Running backtest...")
    equity = [cfg.INITIAL_CAPITAL]
    equity_dates = [bt_dates[0] - pd.Timedelta(weeks=1)]
    weights_history = []
    classifications_history = []
    current_weights = {}

    for i, date in enumerate(bt_dates):
        if i % 52 == 0:
            print(f"  Processing {date.strftime('%Y-%m-%d')} ({i}/{len(bt_dates)})...")

        # ── Compute returns for the past week ──
        if current_weights:
            week_return = _compute_portfolio_return(
                current_weights, etf_weekly, equity_dates[-1], date
            )
            # Deduct transaction costs for rebalancing
            # (costs applied when weights change, computed below)
        else:
            week_return = 0.0

        new_equity = equity[-1] * (1 + week_return)

        # ── Classify each asset ──
        week_classifications = {}
        for ticker in cfg.ETF_TICKERS:
            if ticker not in price_features:
                continue

            # Build feature matrix for this asset
            feat_matrix = sch.build_feature_matrix(ticker, price_features, macro_features)
            if feat_matrix.empty:
                continue

            # Find where current date falls in feature matrix
            if date not in feat_matrix.index:
                # Find closest date before current date
                valid_dates = feat_matrix.index[feat_matrix.index <= date]
                if len(valid_dates) == 0:
                    continue
                target_date = valid_dates[-1]
            else:
                target_date = date

            target_idx = feat_matrix.index.get_loc(target_date)

            # Need enough history for analogues
            if target_idx < cfg.EXCLUSION_WINDOW + cfg.MIN_ANALOGUES:
                continue

            # Find analogues
            analogue_idx, distances = sch.find_analogues(feat_matrix, target_idx)
            if len(analogue_idx) < cfg.MIN_ANALOGUES:
                continue

            # Get forward returns for analogues
            fwd_rets = sch.get_forward_returns(
                etf_weekly[ticker], feat_matrix.index, analogue_idx
            )

            # Classify
            classification = sch.classify_asset(fwd_rets)
            week_classifications[ticker] = classification

        # ── Construct portfolio ──
        new_weights = sch.construct_portfolio(
            week_classifications, etf_weekly, date
        )

        # ── Transaction costs ──
        turnover = _compute_turnover(current_weights, new_weights)
        tc = turnover * cfg.TRANSACTION_COST_BPS / 10000
        new_equity *= (1 - tc)

        current_weights = new_weights
        equity.append(new_equity)
        equity_dates.append(date)

        weights_history.append({"date": date, **new_weights})
        classifications_history.append({
            "date": date,
            "classifications": week_classifications,
            "weights": new_weights.copy(),
        })

    # ── Build result objects ──
    equity_curve = pd.Series(equity, index=equity_dates, name="Schrodinger")

    weights_df = pd.DataFrame(weights_history).set_index("date").fillna(0)

    # Benchmark: SPY buy-and-hold
    spy_prices = etf_weekly["SPY"].dropna()
    spy_start_loc = spy_prices.index.searchsorted(equity_dates[0])
    spy_segment = spy_prices.iloc[spy_start_loc:]
    spy_bh = cfg.INITIAL_CAPITAL * spy_segment / spy_segment.iloc[0]
    spy_bh.name = "SPY Buy & Hold"

    print(f"\n[5/5] Backtest complete.")
    return {
        "equity_curve": equity_curve,
        "weights_history": weights_df,
        "classifications_history": classifications_history,
        "benchmark": spy_bh,
    }


def _compute_portfolio_return(
    weights: dict,
    prices_weekly: pd.DataFrame,
    prev_date: pd.Timestamp,
    curr_date: pd.Timestamp,
) -> float:
    """Compute the weighted return of the portfolio over one week."""
    total_return = 0.0
    total_weight = sum(weights.values())
    cash_weight = max(0, 1.0 - total_weight)

    for ticker, w in weights.items():
        if ticker not in prices_weekly.columns:
            continue
        p = prices_weekly[ticker]
        # Find the prices at or before each date
        prev_loc = p.index.searchsorted(prev_date, side="right") - 1
        curr_loc = p.index.searchsorted(curr_date, side="right") - 1

        if prev_loc < 0 or curr_loc < 0 or prev_loc >= len(p) or curr_loc >= len(p):
            continue

        p0 = p.iloc[prev_loc]
        p1 = p.iloc[curr_loc]

        if p0 > 0 and not np.isnan(p0) and not np.isnan(p1):
            total_return += w * (p1 / p0 - 1)

    # Cash earns nothing (simplification; could add T-bill rate)
    return total_return


def _compute_turnover(old_weights: dict, new_weights: dict) -> float:
    """Compute one-way turnover between two weight vectors."""
    all_tickers = set(list(old_weights.keys()) + list(new_weights.keys()))
    turnover = 0.0
    for t in all_tickers:
        old_w = old_weights.get(t, 0.0)
        new_w = new_weights.get(t, 0.0)
        turnover += abs(new_w - old_w)
    return turnover / 2  # one-way


# ══════════════════════════════════════════════════
# PERFORMANCE METRICS
# ══════════════════════════════════════════════════

def compute_metrics(equity_curve: pd.Series, benchmark: pd.Series = None) -> dict:
    """Compute comprehensive performance metrics."""
    rets = equity_curve.pct_change().dropna()
    annual_factor = 52  # weekly data

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_years = len(rets) / annual_factor
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_vol = rets.std() * np.sqrt(annual_factor)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0

    # Drawdowns
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Sortino
    downside = rets[rets < 0]
    downside_vol = downside.std() * np.sqrt(annual_factor)
    sortino = cagr / downside_vol if downside_vol > 0 else 0

    # Win rate (weekly)
    win_rate = (rets > 0).mean()

    # Best / worst week
    best_week = rets.max()
    worst_week = rets.min()

    # Skewness / kurtosis
    skew = rets.skew()
    kurt = rets.kurtosis()

    metrics = {
        "Total Return": f"{total_return:.1%}",
        "CAGR": f"{cagr:.1%}",
        "Annualized Vol": f"{ann_vol:.1%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Max Drawdown": f"{max_dd:.1%}",
        "Calmar Ratio": f"{calmar:.2f}",
        "Weekly Win Rate": f"{win_rate:.1%}",
        "Best Week": f"{best_week:.2%}",
        "Worst Week": f"{worst_week:.2%}",
        "Skewness": f"{skew:.2f}",
        "Kurtosis": f"{kurt:.2f}",
        "# Years": f"{n_years:.1f}",
    }

    if benchmark is not None:
        bm_rets = benchmark.pct_change().dropna()
        # Align
        common_dates = rets.index.intersection(bm_rets.index)
        if len(common_dates) > 10:
            corr = rets.loc[common_dates].corr(bm_rets.loc[common_dates])
            metrics["Correlation to SPY"] = f"{corr:.2f}"

            # Beta / Alpha
            beta = np.cov(rets.loc[common_dates], bm_rets.loc[common_dates])[0, 1] / np.var(bm_rets.loc[common_dates])
            bm_cagr = (1 + (benchmark.iloc[-1] / benchmark.iloc[0] - 1)) ** (1 / n_years) - 1
            alpha = cagr - beta * bm_cagr
            metrics["Beta to SPY"] = f"{beta:.2f}"
            metrics["Alpha (ann.)"] = f"{alpha:.1%}"

    return metrics


# ══════════════════════════════════════════════════
# CHART GENERATION
# ══════════════════════════════════════════════════

def plot_equity_curve(equity: pd.Series, benchmark: pd.Series, output_path: str):
    """Plot equity curve vs benchmark (log scale)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.semilogy(equity.index, equity.values, linewidth=2, label="Schrödinger ETF Strategy", color="#1a73e8")
    # Align benchmark
    bm_aligned = benchmark.reindex(equity.index, method="ffill")
    if bm_aligned.notna().sum() > 10:
        ax.semilogy(bm_aligned.index, bm_aligned.values, linewidth=1.5, label="SPY Buy & Hold",
                     color="#ea4335", alpha=0.8)

    ax.set_title("Schrödinger's Macro Trend — ETF Strategy Backtest", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Portfolio Value (log scale)")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_drawdowns(equity: pd.Series, benchmark: pd.Series, output_path: str):
    """Plot underwater chart (drawdowns)."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for series, label, color in [
        (equity, "Schrödinger", "#1a73e8"),
        (benchmark, "SPY", "#ea4335"),
    ]:
        cummax = series.cummax()
        dd = (series - cummax) / cummax
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=color)
        ax.plot(dd.index, dd.values, linewidth=1, label=label, color=color)

    ax.set_title("Drawdowns", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_rolling_sharpe(equity: pd.Series, benchmark: pd.Series, output_path: str, window=52):
    """Plot rolling 1-year Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for series, label, color in [
        (equity, "Schrödinger", "#1a73e8"),
        (benchmark, "SPY", "#ea4335"),
    ]:
        rets = series.pct_change().dropna()
        rolling_mean = rets.rolling(window).mean() * 52
        rolling_std = rets.rolling(window).std() * np.sqrt(52)
        rolling_sr = rolling_mean / rolling_std
        ax.plot(rolling_sr.index, rolling_sr.values, linewidth=1.2, label=label, color=color)

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title(f"Rolling {window}-Week Sharpe Ratio", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_allocation_over_time(weights_df: pd.DataFrame, output_path: str):
    """Stacked area chart of asset class allocation over time."""
    if weights_df.empty:
        return

    # Group by asset class
    class_weights = pd.DataFrame(index=weights_df.index)
    for asset_class in ["equity", "fixed_income", "sector", "commodity", "currency"]:
        tickers = [t for t, info in cfg.ETF_UNIVERSE.items() if info["class"] == asset_class]
        cols = [t for t in tickers if t in weights_df.columns]
        if cols:
            class_weights[asset_class.replace("_", " ").title()] = weights_df[cols].sum(axis=1)

    # Add cash
    class_weights["Cash"] = 1.0 - class_weights.sum(axis=1)
    class_weights["Cash"] = class_weights["Cash"].clip(lower=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#1a73e8", "#34a853", "#fbbc04", "#ea4335", "#9c27b0", "#cccccc"]

    ax.stackplot(class_weights.index, class_weights.T.values,
                 labels=class_weights.columns, colors=colors[:len(class_weights.columns)],
                 alpha=0.8)

    ax.set_title("Portfolio Allocation by Asset Class", fontsize=14, fontweight="bold")
    ax.set_ylabel("Weight")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_yearly_returns(equity: pd.Series, benchmark: pd.Series, output_path: str):
    """Bar chart of yearly returns (like Exhibit 1 in Mulliner et al.)."""
    # Resample to yearly
    yearly_eq = equity.resample("YE").last().pct_change().dropna()
    yearly_bm = benchmark.resample("YE").last().pct_change().dropna()

    fig, ax = plt.subplots(figsize=(14, 6))

    years = yearly_eq.index.year
    x = np.arange(len(years))
    width = 0.35

    bars1 = ax.bar(x - width / 2, yearly_eq.values, width, label="Schrödinger",
                   color="#1a73e8", alpha=0.85)
    # Align benchmark years
    bm_vals = []
    for yr in years:
        if yr in yearly_bm.index.year:
            bm_vals.append(yearly_bm[yearly_bm.index.year == yr].values[0])
        else:
            bm_vals.append(0)
    ax.bar(x + width / 2, bm_vals, width, label="SPY", color="#ea4335", alpha=0.65)

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Calendar Year Returns", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_latest_allocations(classifications_history: list, output_path: str):
    """
    Horizontal bar chart of the most recent week's allocations,
    similar to Heiden's Alive/Dead roster display.
    """
    if not classifications_history:
        return

    latest = classifications_history[-1]
    weights = latest["weights"]
    classifs = latest["classifications"]
    date = latest["date"]

    if not weights:
        print("  No positions in final week — skipping allocation chart.")
        return

    # Sort by weight
    sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    tickers = [t for t, w in sorted_w]
    vals = [w for t, w in sorted_w]

    # Color by asset class
    class_colors = {
        "equity": "#1a73e8",
        "fixed_income": "#34a853",
        "sector": "#fbbc04",
        "commodity": "#ea4335",
        "currency": "#9c27b0",
    }
    colors = [class_colors.get(cfg.ETF_UNIVERSE.get(t, {}).get("class", ""), "#999999") for t in tickers]

    fig, ax = plt.subplots(figsize=(10, max(4, len(tickers) * 0.5)))
    bars = ax.barh(range(len(tickers)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(tickers)))
    labels = [f"{t} ({cfg.ETF_UNIVERSE.get(t, {}).get('label', '')})" for t in tickers]
    ax.set_yticklabels(labels, fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(f"Schrödinger's Alive Roster — {date.strftime('%Y-%m-%d')}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Portfolio Weight")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Add weight labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_analogue_fan_chart(
    ticker: str,
    data: dict,
    price_features: dict,
    macro_features: pd.DataFrame,
    etf_weekly: pd.DataFrame,
    output_path: str,
):
    """
    Generate Heiden-style analogue fan chart for a specific asset:
    - Top: recent price path + forward projection with confidence bands
    - Bottom: individual analogue paths
    """
    if ticker not in price_features or ticker not in etf_weekly.columns:
        return

    feat_matrix = sch.build_feature_matrix(ticker, price_features, macro_features)
    if feat_matrix.empty or len(feat_matrix) < cfg.EXCLUSION_WINDOW + cfg.MIN_ANALOGUES + 1:
        return

    target_idx = len(feat_matrix) - 1
    analogue_idx, distances = sch.find_analogues(feat_matrix, target_idx)

    if len(analogue_idx) < cfg.MIN_ANALOGUES:
        return

    # Extract paths
    paths = sch.extract_analogue_paths(
        etf_weekly[ticker], feat_matrix.index, analogue_idx,
        lookback_weeks=26, forward_weeks=26,
    )

    if paths.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

    # ── Top panel: median + confidence bands ──
    rel_weeks = paths.index
    fwd_mask = rel_weeks >= 0
    hist_mask = rel_weeks <= 0

    median_path = paths.median(axis=1)
    p20 = paths.quantile(0.2, axis=1)
    p80 = paths.quantile(0.8, axis=1)

    # Historical part (solid blue)
    ax1.plot(rel_weeks[hist_mask], median_path[hist_mask] * 100, color="#1a73e8", linewidth=2)
    # Forward part (dashed grey)
    ax1.plot(rel_weeks[fwd_mask], median_path[fwd_mask] * 100, color="grey", linewidth=2, linestyle="--")
    ax1.fill_between(rel_weeks[fwd_mask], p20[fwd_mask] * 100, p80[fwd_mask] * 100,
                      color="grey", alpha=0.2, label="20%–80% confidence")
    ax1.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax1.axhline(0, color="black", linewidth=0.3)
    ax1.set_title(f"{ticker} — Analogue Fan Chart ({cfg.ETF_UNIVERSE.get(ticker, {}).get('label', '')})",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Return (%)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Bottom panel: individual analogue paths ──
    for col in paths.columns[:15]:  # Show up to 15 analogues
        ax2.plot(rel_weeks, paths[col] * 100, alpha=0.4, linewidth=0.8)

    # Current price path
    prices = etf_weekly[ticker].dropna()
    current_date = feat_matrix.index[target_idx]
    current_loc = prices.index.searchsorted(current_date)
    lookback = 26
    start_loc = max(0, current_loc - lookback)
    if start_loc < current_loc and current_loc < len(prices):
        segment = prices.iloc[start_loc:current_loc + 1]
        base = prices.iloc[current_loc]
        rebased = (segment / base - 1) * 100
        rel_w = np.arange(-len(rebased) + 1, 1)
        ax2.plot(rel_w, rebased.values, color="#1a73e8", linewidth=2.5, label="Current")

    ax2.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax2.axhline(0, color="black", linewidth=0.3)
    ax2.set_xlabel("Weeks (0 = analogue date)")
    ax2.set_ylabel("Return (%)")
    ax2.set_title("Most Similar Historical Analogues", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_metrics_table(metrics: dict, output_path: str):
    """Render metrics as a clean table image."""
    fig, ax = plt.subplots(figsize=(6, max(4, len(metrics) * 0.35)))
    ax.axis("off")

    rows = list(metrics.items())
    table = ax.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Style header
    for j in range(2):
        cell = table[0, j]
        cell.set_facecolor("#1a73e8")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(1, len(rows) + 1):
        for j in range(2):
            cell = table[i, j]
            cell.set_facecolor("#f8f9fa" if i % 2 == 0 else "white")

    ax.set_title("Performance Summary", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════

def main():
    """Run the full backtest and generate all outputs."""

    # ── Run backtest ──
    results = run_backtest()

    equity = results["equity_curve"]
    benchmark = results["benchmark"]
    weights_df = results["weights_history"]
    classif_hist = results["classifications_history"]

    # ── Compute metrics ──
    print("\n" + "=" * 60)
    print("  PERFORMANCE METRICS")
    print("=" * 60)
    metrics = compute_metrics(equity, benchmark)
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    # SPY metrics for comparison
    print("\n  --- SPY Buy & Hold ---")
    spy_metrics = compute_metrics(benchmark)
    for k, v in spy_metrics.items():
        print(f"  {k:25s}: {v}")

    # ── Generate charts ──
    print("\n" + "=" * 60)
    print("  GENERATING CHARTS")
    print("=" * 60)

    plot_equity_curve(equity, benchmark, str(OUTPUT_DIR / "01_equity_curve.png"))
    plot_drawdowns(equity, benchmark, str(OUTPUT_DIR / "02_drawdowns.png"))
    plot_rolling_sharpe(equity, benchmark, str(OUTPUT_DIR / "03_rolling_sharpe.png"))
    plot_yearly_returns(equity, benchmark, str(OUTPUT_DIR / "04_yearly_returns.png"))
    plot_allocation_over_time(weights_df, str(OUTPUT_DIR / "05_allocation_over_time.png"))
    plot_latest_allocations(classif_hist, str(OUTPUT_DIR / "06_latest_allocations.png"))
    plot_metrics_table(metrics, str(OUTPUT_DIR / "07_metrics_table.png"))

    # ── Analogue fan charts for key assets ──
    print("\n  Generating analogue fan charts for key assets...")

    # Recompute features for fan charts using same data source as backtest
    try:
        data = dm.load_all_data(use_cache=True)
    except Exception:
        import synthetic_data as sd
        data = sd.generate_synthetic_data()

    price_features = sch.compute_price_features(data["etf_weekly"])
    macro_features = sch.compute_macro_features(data["macro_weekly"])

    for ticker in ["SPY", "GLD", "TLT", "XLE", "EEM"]:
        plot_analogue_fan_chart(
            ticker, data, price_features, macro_features, data["etf_weekly"],
            str(OUTPUT_DIR / f"08_fan_chart_{ticker}.png"),
        )

    # ── Save results to CSV ──
    equity.to_csv(OUTPUT_DIR / "equity_curve.csv", header=True)
    weights_df.to_csv(OUTPUT_DIR / "weights_history.csv")

    print("\n" + "=" * 60)
    print(f"  All outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
