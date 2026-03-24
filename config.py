"""
Schrödinger's Macro Trend — Configuration
==========================================
All tuneable parameters in one place.

Based on:
- Moritz Heiden, "Schrödinger's Macro Lens" (Sep/Oct 2025)
- Mulliner, Harvey, Xia, Fang, Van Hemert, "Regimes" (Mar 2025)
"""

# ─────────────────────────────────────────────
# DATA SOURCES
# ─────────────────────────────────────────────
# Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = "263fff36e95136e168c2b9128597195d"

# ─────────────────────────────────────────────
# ETF UNIVERSE (from Heiden Macro Lens #1)
# ─────────────────────────────────────────────
ETF_UNIVERSE = {
    # Equities
    "SPY":  {"class": "equity",      "label": "S&P 500"},
    "IWM":  {"class": "equity",      "label": "Russell 2000"},
    "QQQ":  {"class": "equity",      "label": "Nasdaq 100"},
    "EFA":  {"class": "equity",      "label": "MSCI EAFE"},
    "EEM":  {"class": "equity",      "label": "MSCI EM"},
    "MTUM": {"class": "equity",      "label": "US Momentum Factor"},  # proxy for IWMO
    # Fixed Income
    "TLT":  {"class": "fixed_income", "label": "20+ Year Treasury"},
    "IEF":  {"class": "fixed_income", "label": "7-10 Year Treasury"},
    "LQD":  {"class": "fixed_income", "label": "IG Corporate Bonds"},
    "HYG":  {"class": "fixed_income", "label": "High Yield Bonds"},
    # Sectors & Real Assets
    "XLE":  {"class": "sector",      "label": "Energy Select"},
    "XLF":  {"class": "sector",      "label": "Financials Select"},
    "VNQ":  {"class": "sector",      "label": "Real Estate"},
    "XLK":  {"class": "sector",      "label": "Technology Select"},
    # Commodities
    "DBC":  {"class": "commodity",   "label": "Commodity Tracking"},
    "GLD":  {"class": "commodity",   "label": "Gold"},
    "USO":  {"class": "commodity",   "label": "Oil Fund"},
    # Currency
    "UUP":  {"class": "currency",    "label": "USD Index"},
}
# NOTE: DBMF excluded (launched May 2019, too short for meaningful backtest).
# NOTE: IWMO replaced with MTUM (longer history, same factor).

ETF_TICKERS = list(ETF_UNIVERSE.keys())

# ─────────────────────────────────────────────
# MACRO STATE VARIABLES (from Mulliner et al.)
# ─────────────────────────────────────────────
# These are the 7 variables from the Regimes paper.
# Each is transformed: 12-month change → z-score (rolling 10y) → winsorize ±3.
#
# For weekly data we use 52-week changes and 520-week rolling windows.
MACRO_VARIABLES = {
    "market":       {"source": "yahoo", "ticker": "^GSPC",       "label": "S&P 500"},
    "yield_curve":  {"source": "fred",  "series": "T10Y2Y",      "label": "10y-2y Spread"},
    "oil":          {"source": "fred",  "series": "DCOILWTICO",  "label": "WTI Crude Oil"},
    "copper":       {"source": "yahoo", "ticker": "HG=F",        "label": "Copper Futures"},
    "monetary":     {"source": "fred",  "series": "DTB3",        "label": "3-Month T-Bill"},
    "volatility":   {"source": "yahoo", "ticker": "^VIX",        "label": "VIX Index"},
    # stock_bond_corr is computed from S&P 500 and 10y Treasury returns
    "stock_bond":   {"source": "computed", "label": "Stock-Bond Correlation"},
}

# Fallback Yahoo tickers for FRED series (if FRED key unavailable)
FRED_YAHOO_FALLBACKS = {
    "T10Y2Y":      None,         # No direct Yahoo equivalent; compute from ^TNX - ^IRX
    "DCOILWTICO":  "CL=F",      # WTI crude futures
    "DTB3":        "^IRX",       # 13-week T-bill yield (÷100 for Yahoo)
}

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
# Price features per asset (weekly)
PRICE_MOMENTUM_WINDOWS  = [4, 12, 26]        # weeks (dropped 8w — too correlated with 4/12)
PRICE_VOLATILITY_WINDOWS = [12]              # weeks (one vol measure is enough)
PRICE_DRAWDOWN_WINDOW    = 52                # weeks (high-water mark lookback)

# Macro feature transformation (following Mulliner)
MACRO_CHANGE_WINDOW     = 52    # weeks (~12 months)
MACRO_ZSCORE_WINDOW     = 520   # weeks (~10 years)
MACRO_WINSORIZE_LIMIT   = 3.0   # ±3 standard deviations

# Stock-bond correlation lookback
STOCK_BOND_CORR_WINDOW  = 156   # weeks (~3 years, per Mulliner)

# Relative importance: Heiden says ~60% price, ~40% macro
PRICE_WEIGHT = 0.6
MACRO_WEIGHT = 0.4

# Asset-class-specific macro feature relevance
# Each value is a weight multiplier for that macro variable when building features for that class.
# 0.0 = ignore, 1.0 = full weight, 0.5 = half weight
# This prevents bonds from matching on copper or commodities from matching on stock-bond corr.
MACRO_RELEVANCE = {
    "equity": {
        "macro_market": 1.0, "macro_yield_curve": 1.0, "macro_oil": 0.3,
        "macro_copper": 0.3, "macro_monetary": 0.8, "macro_volatility": 1.0,
        "macro_stock_bond": 0.5,
    },
    "fixed_income": {
        "macro_market": 0.3, "macro_yield_curve": 1.0, "macro_oil": 0.2,
        "macro_copper": 0.0, "macro_monetary": 1.0, "macro_volatility": 0.5,
        "macro_stock_bond": 1.0,
    },
    "sector": {
        "macro_market": 1.0, "macro_yield_curve": 0.8, "macro_oil": 0.8,
        "macro_copper": 0.5, "macro_monetary": 0.8, "macro_volatility": 1.0,
        "macro_stock_bond": 0.3,
    },
    "commodity": {
        "macro_market": 0.3, "macro_yield_curve": 0.5, "macro_oil": 1.0,
        "macro_copper": 1.0, "macro_monetary": 0.8, "macro_volatility": 0.5,
        "macro_stock_bond": 0.0,
    },
    "currency": {
        "macro_market": 0.5, "macro_yield_curve": 1.0, "macro_oil": 0.5,
        "macro_copper": 0.3, "macro_monetary": 1.0, "macro_volatility": 0.5,
        "macro_stock_bond": 0.3,
    },
}

# ─────────────────────────────────────────────
# ANALOGUE ENGINE
# ─────────────────────────────────────────────
# Distance metric
DISTANCE_METRIC = "euclidean"  # "euclidean" or "mahalanobis"

# Distance weighting: weight analogues by 1/distance when computing forward returns
DISTANCE_WEIGHTED = True  # True = closer analogues count more; False = uniform (original)

# Analogue selection
ANALOGUE_PCT       = 0.15      # Top 15% most similar (Mulliner default)
MIN_ANALOGUES      = 20        # Minimum analogues to form a view
EXCLUSION_WINDOW   = 52        # Exclude last 1 year (52 weeks) — compromise between Mulliner (3y) and no exclusion

# Forward horizons for conditional forecasts
FORWARD_HORIZONS   = [4, 8, 12]          # weeks
HORIZON_WEIGHTS    = [0.50, 0.30, 0.20]  # heavier on near-term (Heiden: "4 > 8 > 12")

# ─────────────────────────────────────────────
# ALIVE / DEAD CLASSIFICATION
# ─────────────────────────────────────────────
# An asset is "Alive" if ALL conditions are met:
ALIVE_MIN_ANALOGUES     = 20    # Enough historical matches
ALIVE_HIT_RATE          = 0.55  # >55% of analogue forward returns positive
ALIVE_MIN_RETURN        = 0.002 # Blended expected return > 0.2%
ALIVE_MIN_CONFIDENCE    = 0.5   # t-stat of mean return > 0.5 (relaxed from original 1.0)

# ─────────────────────────────────────────────
# PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────
# Position sizing
SIZING_METHOD       = "signal_strength"   # rank by blended expected return / vol
LOGIT_POWER         = 3.0                 # logit-power transform exponent (was 1.5 — concentrate in top picks)
MAX_POSITION_WEIGHT = 0.40                # 40% cap per ETF
MIN_POSITION_WEIGHT = 0.02                # 2% floor (below this → zero)

# Volatility targeting
TARGET_VOL          = 0.15   # 15% annualized (Heiden ETF version)
VOL_LOOKBACK        = 52     # weeks for portfolio vol estimation
VOL_SCALE_CAP       = 2.0    # max leverage from vol scaling (for safety)

# Rebalancing
REBALANCE_FREQ      = "W-FRI"  # weekly on Fridays

# Transaction costs
TRANSACTION_COST_BPS = 10  # 10 bps round-trip (conservative for liquid ETFs)

# ─────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────
BACKTEST_START = "2008-01-01"   # Earliest feasible given ETF inception dates
BACKTEST_END   = None           # None = latest available
INITIAL_CAPITAL = 100_000

# Minimum history required before first trade (features + analogues)
MIN_WARMUP_WEEKS = 520 + 52 + 12  # zscore window + change window + forward horizon
