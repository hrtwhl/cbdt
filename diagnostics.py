"""
Schrödinger's Macro Trend — Diagnostics
=========================================
Analyzes WHY the backtest underperforms.
Run this after run_backtest.py to understand the bottlenecks.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import config as cfg
import data_manager as dm
import schrodinger as sch

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_diagnostics():
    print("=" * 60)
    print("  DIAGNOSTICS")
    print("=" * 60)

    # Load cached data
    try:
        data = dm.load_all_data(use_cache=True)
    except:
        print("No cached data found. Run run_backtest.py first.")
        return

    etf_weekly = data["etf_weekly"]
    macro_weekly = data["macro_weekly"]

    # ── 1. Macro feature quality ──
    print("\n[1] MACRO FEATURE QUALITY")
    macro_features = sch.compute_macro_features(macro_weekly)
    print(f"  Shape: {macro_features.shape}")
    print(f"  First valid week: {macro_features.dropna().index[0]}")
    print(f"  Last valid week:  {macro_features.dropna().index[-1]}")
    print(f"  Valid weeks:      {macro_features.dropna().shape[0]}")
    print(f"\n  Per-variable stats (should be ~mean=0, ~std=1, range [-3,3]):")
    for col in macro_features.columns:
        s = macro_features[col].dropna()
        print(f"    {col:30s}: mean={s.mean():+.2f}  std={s.std():.2f}  "
              f"min={s.min():+.2f}  max={s.max():+.2f}  n={len(s)}")

    # ── 2. Price feature quality ──
    print("\n[2] PRICE FEATURE QUALITY")
    price_features = sch.compute_price_features(etf_weekly)
    for ticker in ["SPY", "TLT", "GLD", "XLE"]:
        if ticker in price_features:
            pf = price_features[ticker].dropna()
            print(f"  {ticker}: {pf.shape[0]} valid weeks, {pf.shape[1]} features")

    # ── 3. Feature matrix quality (combined) ──
    print("\n[3] COMBINED FEATURE MATRIX QUALITY")
    for ticker in cfg.ETF_TICKERS:
        fm = sch.build_feature_matrix(ticker, price_features, macro_features)
        if not fm.empty:
            print(f"  {ticker:6s}: {fm.shape[0]:5d} weeks × {fm.shape[1]:2d} features  "
                  f"(first: {fm.index[0].strftime('%Y-%m-%d')}, "
                  f"last: {fm.index[-1].strftime('%Y-%m-%d')})")
        else:
            print(f"  {ticker:6s}: EMPTY")

    # ── 4. Analogue matching test (for SPY at several dates) ──
    print("\n[4] ANALOGUE MATCHING DIAGNOSTICS (SPY)")
    fm_spy = sch.build_feature_matrix("SPY", price_features, macro_features)
    if not fm_spy.empty:
        test_dates = ["2008-10-01", "2015-06-01", "2020-03-20", "2023-01-01", "2025-01-01"]
        for td in test_dates:
            ts = pd.Timestamp(td)
            valid = fm_spy.index[fm_spy.index <= ts]
            if len(valid) == 0:
                continue
            target_date = valid[-1]
            target_idx = fm_spy.index.get_loc(target_date)

            if target_idx < cfg.EXCLUSION_WINDOW + cfg.MIN_ANALOGUES:
                print(f"  {td}: Not enough history (idx={target_idx}, need {cfg.EXCLUSION_WINDOW + cfg.MIN_ANALOGUES})")
                continue

            analogue_idx, distances = sch.find_analogues(fm_spy, target_idx)
            if len(analogue_idx) == 0:
                print(f"  {td}: No analogues found!")
                continue

            # Get forward returns
            fwd = sch.get_forward_returns(etf_weekly["SPY"], fm_spy.index, analogue_idx)

            # Classify
            classif = sch.classify_asset(fwd)

            analogue_dates = fm_spy.index[analogue_idx]
            print(f"\n  Date: {td} (actual: {target_date.strftime('%Y-%m-%d')}, idx={target_idx})")
            print(f"    Analogues found: {len(analogue_idx)}")
            print(f"    Distance range:  {distances.min():.2f} — {distances.max():.2f} (mean={distances.mean():.2f})")
            print(f"    Analogue year spread: {analogue_dates.min().year} — {analogue_dates.max().year}")
            print(f"    Status: {classif['status'].upper()}")
            print(f"    Blended return: {classif['blended_return']:.4f}")
            print(f"    Hit rate: {classif['hit_rate']:.2%}")
            print(f"    Confidence (t-stat): {classif['confidence']:.2f}")
            for h, detail in classif["detail"].items():
                if "mean" in detail and not np.isnan(detail.get("mean", np.nan)):
                    print(f"    {h}w: mean={detail['mean']:.4f}  hit={detail['hit']:.2%}  "
                          f"t={detail['tstat']:.2f}  n={detail['n']}")

    # ── 5. Alive/Dead statistics over backtest period ──
    print("\n[5] ALIVE/DEAD STATISTICS OVER BACKTEST")
    bt_start = pd.Timestamp(cfg.BACKTEST_START)
    bt_dates = fm_spy.index[fm_spy.index >= bt_start] if not fm_spy.empty else []

    alive_counts = []
    total_weight = []
    alive_tickers_list = []

    sample_dates = bt_dates[::13]  # every ~quarter for speed
    for date in sample_dates:
        week_alive = 0
        week_weight = 0.0
        week_tickers = []

        classifications = {}
        for ticker in cfg.ETF_TICKERS:
            fm = sch.build_feature_matrix(ticker, price_features, macro_features)
            if fm.empty:
                continue
            valid = fm.index[fm.index <= date]
            if len(valid) == 0:
                continue
            target_date = valid[-1]
            target_idx = fm.index.get_loc(target_date)
            if target_idx < cfg.EXCLUSION_WINDOW + cfg.MIN_ANALOGUES:
                continue

            analogue_idx, distances = sch.find_analogues(fm, target_idx)
            if len(analogue_idx) < cfg.MIN_ANALOGUES:
                continue

            fwd = sch.get_forward_returns(etf_weekly[ticker], fm.index, analogue_idx)
            classif = sch.classify_asset(fwd)
            classifications[ticker] = classif

            if classif["status"] == "alive":
                week_alive += 1
                week_tickers.append(ticker)

        # Build portfolio
        weights = sch.construct_portfolio(classifications, etf_weekly, date)
        week_weight = sum(weights.values())

        alive_counts.append(week_alive)
        total_weight.append(week_weight)
        alive_tickers_list.append(week_tickers)

    alive_counts = np.array(alive_counts)
    total_weight = np.array(total_weight)

    print(f"  Sample periods: {len(sample_dates)}")
    print(f"  Avg alive assets per week:   {alive_counts.mean():.1f}")
    print(f"  Median alive:                {np.median(alive_counts):.0f}")
    print(f"  Weeks with 0 alive:          {(alive_counts == 0).sum()} ({(alive_counts == 0).mean():.0%})")
    print(f"  Weeks with ≤2 alive:         {(alive_counts <= 2).sum()} ({(alive_counts <= 2).mean():.0%})")
    print(f"  Avg invested weight:         {total_weight.mean():.1%}")
    print(f"  Median invested weight:      {np.median(total_weight):.1%}")
    print(f"  Weeks fully in cash:         {(total_weight == 0).sum()} ({(total_weight == 0).mean():.0%})")

    # Most common alive tickers
    from collections import Counter
    all_alive = [t for sublist in alive_tickers_list for t in sublist]
    if all_alive:
        print(f"\n  Most frequently alive:")
        for ticker, count in Counter(all_alive).most_common(10):
            print(f"    {ticker:6s}: {count}/{len(sample_dates)} ({count/len(sample_dates):.0%})")

    # ── 6. Threshold sensitivity ──
    print("\n[6] THRESHOLD SENSITIVITY (SPY, latest date)")
    if not fm_spy.empty:
        target_idx = len(fm_spy) - 1
        analogue_idx, distances = sch.find_analogues(fm_spy, target_idx)
        if len(analogue_idx) > 0:
            fwd = sch.get_forward_returns(etf_weekly["SPY"], fm_spy.index, analogue_idx,
                                          analogue_distances=distances)

            print(f"  Forward return stats at each horizon:")
            for h in cfg.FORWARD_HORIZONS:
                raw = fwd.get(h, {})
                rets = raw["returns"] if isinstance(raw, dict) else raw
                if len(rets) > 0:
                    print(f"    {h}w: mean={np.mean(rets):.4f}  median={np.median(rets):.4f}  "
                          f"std={np.std(rets):.4f}  hit={np.mean(rets>0):.2%}  n={len(rets)}")

            # Test different thresholds
            print(f"\n  Classification under different thresholds:")
            for hit_thresh in [0.50, 0.52, 0.55, 0.60]:
                for ret_thresh in [0.0, 0.002, 0.005, 0.01]:
                    old_hit = cfg.ALIVE_HIT_RATE
                    old_ret = cfg.ALIVE_MIN_RETURN
                    cfg.ALIVE_HIT_RATE = hit_thresh
                    cfg.ALIVE_MIN_RETURN = ret_thresh
                    c = sch.classify_asset(fwd)
                    cfg.ALIVE_HIT_RATE = old_hit
                    cfg.ALIVE_MIN_RETURN = old_ret
                    print(f"    hit≥{hit_thresh:.0%} ret≥{ret_thresh:.3f}: "
                          f"{c['status']:10s} (blended={c['blended_return']:.4f}, "
                          f"hit={c['hit_rate']:.2%}, t={c['confidence']:.2f})")

    print("\n" + "=" * 60)
    print("  DIAGNOSIS SUMMARY")
    print("=" * 60)
    if alive_counts.mean() < 4:
        print("  ⚠ Too few alive assets → excessive cash drag")
    if (total_weight == 0).mean() > 0.1:
        print("  ⚠ Spending >10% of time fully in cash")
    if total_weight.mean() < 0.5:
        print("  ⚠ Average invested weight < 50% → severe cash drag")
    print(f"  Avg {alive_counts.mean():.1f} alive, avg {total_weight.mean():.0%} invested")


if __name__ == "__main__":
    run_diagnostics()
