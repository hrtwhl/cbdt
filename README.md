# Schrödinger's Macro Trend — ETF Backtest

A Python implementation of the analogue-based macro regime model described by Moritz Heiden ("Schrödinger's Macro Lens", 2025) with macro state variables from Mulliner, Harvey, Xia, Fang & Van Hemert ("Regimes", 2025).


## Architecture

```
schrodinger_backtest/
├── config.py           # All tuneable parameters
├── data_manager.py     # Data fetching (Yahoo Finance + FRED)
├── schrodinger.py      # Core model (features, analogues, alive/dead, portfolio)
├── run_backtest.py     # Main entry point (backtest + metrics + charts)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Methodology

### 1. State Construction ("Market Fingerprint")
Each Friday, every ETF is described by a feature vector combining:

**Price features** (~60% weight):
- Momentum: 4w, 8w, 12w, 26w log returns
- Realized volatility: 4w, 12w annualized
- Drawdown from 52-week high

**Macro state variables** (~40% weight, from Mulliner et al.):
1. S&P 500 (market level)
2. Yield curve (10y-2y spread)
3. WTI crude oil
4. Copper
5. 3-month T-bill (monetary policy)
6. VIX (volatility)
7. Stock-bond correlation

Each macro variable is transformed: 52-week change → rolling 520-week z-score → winsorized at ±3.

### 2. Analogue Matching
For each asset on each Friday:
- Compute Euclidean distance between today's feature vector and all historical weeks
- Exclude the most recent 156 weeks (3 years) to avoid momentum contamination
- Select the top 15% most similar weeks as analogues

### 3. Conditional Forecasts
For each analogue, look forward 4, 8, and 12 weeks and record what actually happened. This produces return distributions at each horizon.

### 4. Alive/Dead Classification
An asset is **Alive** (allocated to) if:
- Enough analogues (≥20)
- Consistent forward returns (>55% positive)
- Meaningful expected return (blended >0.5%)
- Sufficient confidence (blended t-stat >1.0)

Otherwise it's **Dead** (zero allocation).

### 5. Portfolio Construction
- Alive assets ranked by signal strength (blended expected return)
- Logit-power transform for conviction sizing
- Position cap: 40% max per ETF
- Volatility targeting: 15% annualized
- Long-only, weekly rebalancing (Friday close)
- Transaction costs: 10 bps round-trip

## ETF Universe

| Ticker | Class | Description |
|--------|-------|-------------|
| SPY | Equity | S&P 500 |
| IWM | Equity | Russell 2000 |
| QQQ | Equity | Nasdaq 100 |
| EFA | Equity | MSCI EAFE (Developed ex-US) |
| EEM | Equity | MSCI Emerging Markets |
| MTUM | Equity | US Momentum Factor (proxy for IWMO) |
| TLT | Fixed Income | 20+ Year Treasury |
| IEF | Fixed Income | 7-10 Year Treasury |
| LQD | Fixed Income | Investment Grade Corporate |
| HYG | Fixed Income | High Yield Corporate |
| XLE | Sector | Energy |
| XLF | Sector | Financials |
| VNQ | Sector | Real Estate |
| XLK | Sector | Technology |
| DBC | Commodity | Commodity Index |
| GLD | Commodity | Gold |
| USO | Commodity | Oil Fund |
| UUP | Currency | USD Index |


## References

- Heiden, M. (2025). "Opening the Box of Macro Trend." *Methods to the Madness*.
- Heiden, M. (2025). "Schrödinger's Macro Lens #1 & #2." *Methods to the Madness*.
- Mulliner, A., Harvey, C.R., Xia, C., Fang, E., Van Hemert, O. (2025). "Regimes." Man Group / Duke University.
- Gilboa, I. & Schmeidler, D. (1995). "Case-Based Decision Theory." *Quarterly Journal of Economics*.
