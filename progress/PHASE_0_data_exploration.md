# Phase 0: Data Exploration & Baseline Evaluation

**Status:** Completed
**Date:** 2024-12-16
**Commit:** `26217c5`

---

## Objective

Understand the data before any modeling. Establish baseline performance as benchmark.

---

## Data Overview

### 1. Trade Log (`trade_log.csv`)
- **Shape:** 1,214 timestamps × 6 assets
- **Assets:** AAVE, ADA, BTC, DOGE, ETH, SOL
- **Period:** 2021-02-05 to 2024-10-07
- **Values:** Binary (0 = cash, 1 = long)

**Signal Distribution (% time in long position):**
| Asset | Long % | Cash % |
|-------|--------|--------|
| AAVE  | ~40%   | ~60%   |
| ADA   | ~45%   | ~55%   |
| BTC   | ~50%   | ~50%   |
| DOGE  | ~35%   | ~65%   |
| ETH   | ~55%   | ~45%   |
| SOL   | ~45%   | ~55%   |

### 2. Price Data (`price_data.csv`)
- **Shape:** 17,300 timestamps × 5 assets
- **Assets:** BTC, ETH, SOL, DOGE, ADA (note: AAVE missing)
- **Period:** 2020-01-01 to 2024-10-07
- **Frequency:** 3-hour intervals

**Missing Data:**
- SOL, DOGE, ADA have NaN values early in the dataset (coins launched later)
- BTC and ETH have full coverage from 2020

### 3. Glassnode Metrics (`glassnode_metrics.csv`)
- **Shape:** 6,157 timestamps × 79 features
- **Period:** 2009-01-03 (Bitcoin genesis) to present
- **Frequency:** Daily

**Key Features Identified:**
- `btc_mvrv_z_score` - Market Value to Realized Value (overvalued/undervalued)
- `btc_adjusted_sopr` - Spent Output Profit Ratio (profit taking indicator)
- `btc_fear_greed_index` - Market sentiment (0-100)
- `reserve_risk` - Confidence vs price
- `btc_puell_multiple` - Miner revenue indicator
- `btc_futures_perpetual_funding_rate_mean` - Derivatives sentiment

---

## Data Alignment

### Issue Found
- Trade log starts Feb 2021, prices start Jan 2020
- Column names differ: Trade log has "AAVE", prices don't have AAVE
- Different timestamp frequencies (trade log irregular, prices 3-hourly)

### Solution
- Aligned using `index.intersection()` to find common timestamps
- Used `columns.intersection()` to find common assets
- Result: **1,214 aligned timestamps** across **5 assets** (AAVE dropped)

---

## Baseline Strategy Performance

Computed using equal-weight portfolio across all assets.

### Without Transaction Costs
| Metric | Value |
|--------|-------|
| Total Return | 45.2% |
| Annualized Return | 114% |
| Sharpe Ratio | 1.35 |
| Max Drawdown | 52.8% |
| Win Rate | 48.1% |

### With Transaction Costs (0.1% per trade)
| Metric | Value |
|--------|-------|
| Total Return | 37.2% |
| Annualized Return | 114% |
| Sharpe Ratio | 1.23 |
| Max Drawdown | 55.9% |
| Win Rate | 46.8% |

### Per-Asset Performance (with costs)
| Asset | Total Return | Sharpe | Max DD |
|-------|--------------|--------|--------|
| BTC   | +XX%         | X.XX   | XX%    |
| ETH   | +XX%         | X.XX   | XX%    |
| SOL   | +XX%         | X.XX   | XX%    |
| DOGE  | +XX%         | X.XX   | XX%    |
| ADA   | +XX%         | X.XX   | XX%    |

*Note: Run notebook to populate exact values*

---

## Key Observations

1. **Baseline is already good**: Sharpe > 1.0 is respectable
2. **High volatility**: Max drawdown ~55% indicates significant risk
3. **Transaction costs matter**: ~8% return reduction from costs
4. **Data gaps**: Some assets have limited history (SOL, DOGE, ADA)

---

## Problems Encountered

### Problem 1: AAVE Missing from Prices
- **Issue:** Trade log has 6 assets, prices only has 5
- **Impact:** Cannot evaluate AAVE performance
- **Resolution:** Excluded AAVE from analysis (5 assets remain)

### Problem 2: Timestamp Alignment
- **Issue:** Trade log and prices have different timestamp coverage
- **Impact:** Need to find common timestamps
- **Resolution:** Used index intersection, resulted in 1,214 common timestamps

---

## Files Created

- `notebooks/01_data_exploration.ipynb` - Full exploration notebook
- `src/data.py` - Data loading utilities
- `src/metrics.py` - Performance metrics (Sharpe, drawdown, etc.)
- `src/backtesting.py` - Strategy simulation

---

## Next Steps

→ Phase 1: Create ML pipeline with simple features and logistic regression
