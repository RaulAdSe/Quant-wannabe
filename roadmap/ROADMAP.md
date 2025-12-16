# Iqana Quant Challenge - Comprehensive Roadmap

> **Timeline:** 4 days
> **Objective:** Build an ML meta-strategy that filters a baseline trading strategy to improve risk-adjusted performance

---

## The Big Picture (Physics Analogy)

Think of this as a **signal filtering problem**:

- **Baseline strategy** = A noisy detector that says "long" or "cash"
- **Your ML model** = A filter that decides "trust this signal" or "ignore it"
- **Goal** = Improve signal-to-noise ratio (risk-adjusted returns)

The formula:
```
signal_ml[t,a] = signal_base[t,a] × 1[p̂(t,a) > τ]
```

This is a **gate function** - your model provides `p̂` (probability the trade is good), and you only follow the baseline when confident enough.

---

## Data Overview

| File | Rows | Description |
|------|------|-------------|
| `trade_log.csv` | 1,215 | Binary signals for 6 assets (AAVE, ADA, BTC, DOGE, ETH, SOL) |
| `price_data.csv` | 17,301 | 3-hour price data, starting Jan 2020 |
| `glassnode_metrics.csv` | 6,157 | ~80 Bitcoin on-chain features (regime indicators) |

---

## Progress Tracking

- [ ] **Phase 0:** Foundation & Data Understanding
- [ ] **Phase 1:** Naive Baseline (End-to-End Pipeline)
- [ ] **Phase 2:** Feature Engineering
- [ ] **Phase 3:** Better Labels & Costs
- [ ] **Phase 4:** Better Models
- [ ] **Phase 5:** Regime Detection
- [ ] **Phase 6:** Threshold Optimization & Robustness

---

## PHASE 0: Foundation (Day 1 Morning)

**Goal:** Understand what you're working with before any modeling

### Step 0.1: Data Loading & Sanity Checks

**What:** Load all CSVs, check dtypes, missing values, date ranges
**Why:** Garbage in = garbage out. Find issues now, not during modeling.
**Physics parallel:** Like calibrating instruments before an experiment

Key questions to answer:
- When does each dataset start/end?
- How many NaN values per column?
- Are timestamps aligned across files?
- What's the trading frequency?

### Step 0.2: Understand the Baseline Strategy

**What:** Compute baseline performance WITHOUT any ML
**Why:** You need a benchmark. If baseline already performs well, your improvement might be marginal (and that's OK to report)
**Physics parallel:** Control experiment

Metrics to compute:
- **Total return**: How much money did baseline make?
- **Sharpe ratio**: Return / volatility (risk-adjusted return)
- **Max drawdown**: Worst peak-to-trough decline
- **Win rate**: % of trades that were profitable

### Step 0.3: Visualize the Data

**What:** Plot prices, signals, and equity curves
**Why:** Your eyes will catch patterns/anomalies that statistics miss

---

## PHASE 1: Naive Baseline (Day 1 Afternoon)

**Goal:** Get something working end-to-end, even if trivial

### Step 1.1: Define the Label (Target Variable)

**What:** What are you predicting?
**Answer:** Whether following the baseline signal will be profitable

Simplest label:
```
y[t,a] = 1 if return over next N periods > transaction_cost, else 0
```

Start with: N = 1 period (3 hours), cost = 0
**Why simple first:** Get the pipeline working, then add complexity

**Critical concept - No lookahead bias:**
- At time t, you can only use information available at time t
- The label uses future prices (that's what we predict), but features cannot

### Step 1.2: Simplest Features

**What:** Use only price-derived features at time t
**Why:** Start simple, establish a baseline for feature importance

Features:
- Return over last 1, 3, 7 days
- Volatility over last 7, 30 days
- Current signal value (0 or 1)
- Asset identifier (one-hot encoded)

### Step 1.3: Train-Test Split (Walk-Forward)

**What:** Train on past, test on future, slide the window forward
**Why:** Simulates real trading - you never see the future

**DO NOT use random train/test split!** Time series require temporal splits.

```
[====TRAIN====][=TEST=]
      [====TRAIN====][=TEST=]
            [====TRAIN====][=TEST=]
```

### Step 1.4: Simplest Model - Logistic Regression

**What:** P(profitable) = sigmoid(w·features + b)
**Why:**
- Interpretable (weights tell you feature importance)
- Fast to train
- Baseline for more complex models

### Step 1.5: Evaluate

**What:** Compare baseline vs filtered strategy
**Metrics:** Sharpe ratio, total return, max drawdown

**Expected outcome Phase 1:** Probably won't beat baseline much. That's fine - you have a working pipeline.

---

## PHASE 2: Feature Engineering (Day 2)

**Goal:** Add domain knowledge through better features

### Step 2.1: Technical Features (Price-Based)

Standard quantitative trading indicators:
- Moving averages (7d, 30d, 90d)
- RSI (Relative Strength Index) - momentum oscillator
- MACD (trend following)
- Bollinger Bands (volatility bands)
- Rate of change at multiple scales

### Step 2.2: On-Chain Features (Glassnode)

Key features to understand and use:
- **MVRV Z-Score**: Market value vs realized value (over/undervalued)
- **SOPR**: Spent Output Profit Ratio (are holders selling at profit?)
- **Fear & Greed Index**: Market sentiment
- **HODL Waves**: Coin age distribution (accumulation vs distribution)
- **Funding Rates**: Futures market sentiment
- **Exchange Flows**: Money moving in/out of exchanges

**Physics intuition:** These are like "order parameters" that characterize the market regime.

### Step 2.3: Cross-Asset Features

**What:** Use BTC features for all assets
**Why:** Crypto is highly correlated - BTC leads the market

### Step 2.4: Feature Engineering Best Practices

- Normalize/standardize features (z-score or rank)
- Handle missing values (forward fill, then drop)
- Create interaction features sparingly
- Log-transform skewed features

---

## PHASE 3: Better Labels & Costs (Day 2-3)

**Goal:** Make the problem more realistic

### Step 3.1: Transaction Costs

**What:** Assume each trade costs ~0.1-0.3% (typical for crypto)
**Why:** Many "profitable" signals aren't profitable after costs

New label:
```
y[t,a] = 1 if return[t+1:t+N] > cost_threshold
```

### Step 3.2: Horizon Selection

**What:** How far ahead do you predict?

Options:
- 1 period (3h): Noisy but responsive
- 8 periods (1 day): More stable
- 56 periods (1 week): Smoothest but slow to react

**Method:** Test multiple horizons, see which gives best risk-adjusted returns

### Step 3.3: Cost-Aware Labels

Consider: Only label "1" if the trade covers costs AND provides meaningful return.

---

## PHASE 4: Better Models (Day 3)

**Goal:** Try models that can capture non-linear patterns

### Step 4.1: Gradient Boosting (XGBoost/LightGBM)

**Why:**
- Handles non-linear relationships
- Built-in feature importance
- State-of-the-art for tabular data
- No need for feature scaling

### Step 4.2: Model Comparison

Compare with same walk-forward CV:
1. Logistic Regression (linear baseline)
2. Random Forest (simple ensemble)
3. XGBoost/LightGBM (boosted ensemble)

### Step 4.3: Hyperparameter Tuning

**Method:** Time-series cross-validation (not random CV!)
Keep it simple: Tune 2-3 most important parameters

---

## PHASE 5: Regime Detection (Day 3-4)

**Goal:** Uncover the "latent regime dynamics" mentioned in the challenge

### Step 5.1: Hidden Markov Model (HMM)

**What:** Model market as switching between hidden states
**Why:** The hint explicitly mentions "latent regime dynamics"

**Physics parallel:** Inferring hidden states in a dynamical system from observations

Approach:
- Fit HMM to returns (2-3 states: bull/bear/sideways)
- Use inferred state as a feature for your ML model

### Step 5.2: Clustering-Based Regimes

**What:** Cluster time periods by feature similarity
**Method:** K-means or Gaussian Mixture on Glassnode features

### Step 5.3: Regime-Conditional Models

**What:** Train different models for different regimes
**Why:** A model that works in bull markets might fail in bear markets

---

## PHASE 6: Threshold Optimization & Robustness (Day 4)

**Goal:** Turn probabilities into trading decisions

### Step 6.1: Threshold Selection

The formula uses threshold τ: take trade only if p̂(t,a) > τ

Trade-off:
- High τ: Fewer trades, higher confidence, might miss good trades
- Low τ: More trades, lower confidence, might take bad trades

**Method:** Plot Sharpe ratio vs threshold, find optimal

### Step 6.2: Robustness Analysis

Tests:
- Performance across different time periods
- Performance per asset
- Performance in different regimes
- Sensitivity to threshold choice

### Step 6.3: Failure Mode Analysis

**Why:** Explicitly requested in deliverables!

Common failures:
- Regime changes (model trained on bull, deployed in bear)
- Black swan events
- Low liquidity periods
- Correlated drawdowns across assets

---

## Key Concepts Reference

### Lookahead Bias
The cardinal sin of backtesting. At time t, you can ONLY use information available at time t.

### Walk-Forward Validation
Never use random train/test splits for time series. Always train on past, test on future.

### Sharpe Ratio
```
Sharpe = (Mean Return - Risk Free Rate) / Std(Returns)
```
Intuition: Return per unit of risk. Higher = better. Typical good: > 1.0, Excellent: > 2.0

### Overfitting in Finance
With enough parameters, you can fit any historical data perfectly. But past patterns may not repeat.

---

## Deliverables Checklist

- [ ] Jupyter notebook (clean, reproducible) covering methodology + results
- [ ] Short PDF report summarizing findings
- [ ] Clear explanation of expectations, limitations, and when the approach is expected to fail

---

## What Should Impress Them

1. **Clean, reproducible code** - They're testing if you can work like a quant
2. **No lookahead bias** - This is make-or-break
3. **Walk-forward validation** - Shows you understand time series
4. **Regime awareness** - Shows you read the hint and thought deeply
5. **Honest failure analysis** - Shows maturity; they know no model is perfect
6. **Clear reasoning** - "Strong reasoning" is explicitly valued
