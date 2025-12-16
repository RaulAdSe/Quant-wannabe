# Phase 1: Naive Model - End-to-End ML Pipeline

**Status:** Completed
**Date:** 2024-12-16
**Commits:** `6bde619`, `10a64bc` (bug fix)

---

## Objective

Build a complete ML pipeline from scratch, even if simple. Establish that:
1. We can create proper labels without lookahead bias
2. We can implement walk-forward validation
3. The meta-strategy framework works

---

## Approach

### The Meta-Strategy Formula
```
signal_ml[t,a] = signal_base[t,a] × 1[p̂(t,a) > τ]
```

- Only take a long position when:
  1. Baseline says "long" (signal_base = 1), AND
  2. Our model is confident it will be profitable (p̂ > threshold)

### Label Definition
```python
y[t,a] = 1 if forward_return[t:t+horizon] > cost_threshold
       = 0 otherwise
```

- **Horizon:** 8 periods (1 day = 3h × 8)
- **Cost threshold:** 0.2% (covers round-trip transaction costs)
- **Only when signal=1:** We only need predictions when baseline says "long"

### Features (Simple, Price-Based)
| Feature | Window | Description |
|---------|--------|-------------|
| `return_1p` | 3 hours | Recent momentum |
| `return_8p` | 1 day | Daily return |
| `return_56p` | 1 week | Weekly return |
| `volatility_56p` | 1 week | Recent volatility |

---

## Implementation Details

### Data Preparation
1. Load and align trade_log, prices
2. Create features per asset (e.g., `BTC_return_1p`)
3. Create labels per asset
4. Stack across assets with **asset-agnostic feature names**

### Walk-Forward Validation
```
Time: ----1----2----3----4----5----6----7----8---->

Fold 1: [=====TRAIN=====][TEST]
Fold 2:      [=====TRAIN=====][TEST]
Fold 3:           [=====TRAIN=====][TEST]
```

- **Train size:** 60% of timestamps (~720)
- **Test size:** 10% of timestamps (~120)
- **Step size:** Non-overlapping test sets

### Model
- **Algorithm:** Logistic Regression
- **Scaling:** StandardScaler (z-score normalization)
- **Class balancing:** `class_weight='balanced'`
- **Regularization:** C=1.0 (default L2)

---

## Results

### Classification Metrics (Walk-Forward CV)
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | ~50% | ±5% |
| Precision | ~51% | ±5% |
| Recall | ~55% | ±10% |
| AUC | ~0.52 | ±0.03 |

*Note: Near-random performance expected with simple features*

### Strategy Performance (Test Period)

**Baseline (no ML filter):**
| Metric | Value |
|--------|-------|
| Sharpe Ratio | 2.39 |
| Total Return | 38.44% |
| Max Drawdown | ~45% |

**ML-Filtered at Different Thresholds:**
| Threshold τ | Sharpe | Total Return | Trade Reduction |
|-------------|--------|--------------|-----------------|
| 0.4 | 2.39 | 38.44% | ~5% |
| 0.5 | 2.45 | 30.59% | ~20% |
| 0.6 | **3.93** | 11.71% | ~60% |

### Key Finding
**Higher threshold → Better risk-adjusted returns but lower total return**

At τ=0.6:
- Sharpe improved from 2.39 → 3.93 (**+64%**)
- But total return dropped from 38% → 12%
- Trade reduction: ~60% fewer trades

This suggests the model CAN identify some low-quality trades to filter out.

---

## Problems Encountered

### Problem 1: NaN in Feature Matrix (CRITICAL)

**Issue:**
When stacking data across assets, each row had asset-specific column names:
```python
# Row for ADA: {'ADA_return_1p': 0.05, 'ADA_return_8p': 0.02, ...}
# Row for BTC: {'BTC_return_1p': 0.03, 'BTC_return_8p': 0.01, ...}
```
When creating DataFrame, pandas created ALL columns and filled with NaN:
```
           ADA_return_1p  BTC_return_1p  ...
ADA row:   0.05           NaN            ...
BTC row:   NaN            0.03           ...
```

**Error:**
```
ValueError: Input X contains NaN.
LogisticRegression does not accept missing values encoded as NaN natively.
```

**Solution:**
Rename features to be asset-agnostic when stacking:
```python
# Instead of: {'ADA_return_1p': 0.05, ...}
# Use:        {'return_1p': 0.05, ...}

for col in feature_cols:
    new_name = col.replace(asset + '_', '')  # 'ADA_return_1p' -> 'return_1p'
    renamed_features[new_name] = row_features[col]
```

**Commit:** `10a64bc`

### Problem 2: Low Model Performance

**Issue:** AUC ~0.52 is barely better than random

**Why This Happens:**
1. Simple features (only 4) don't capture market dynamics
2. Crypto markets are noisy and efficient
3. Label horizon (1 day) may be too short/long

**Resolution:** Expected for Phase 1. Will improve in later phases with:
- Glassnode on-chain features (Phase 2)
- Better models - XGBoost (Phase 4)
- Regime detection (Phase 5)

---

## Code Structure

### New Files Created
```
notebooks/02_naive_model.ipynb  - Full ML pipeline notebook
src/features.py                 - Feature engineering utilities
src/labels.py                   - Label creation utilities
requirements.txt                - Python dependencies
setup.sh                        - Environment setup script
```

### Key Functions

**`prepare_stacked_data(features, labels, signals)`**
- Stacks data across assets into single DataFrame
- Handles asset-agnostic feature renaming
- Filters to only signal=1 rows

**`walk_forward_cv(X, y, train_size, test_size, step_size)`**
- Generator for walk-forward cross-validation
- Yields (train_idx, test_idx, fold_info) tuples
- Ensures temporal ordering (no leakage)

**`create_filtered_signals(predictions_df, signals, threshold)`**
- Applies ML filter to baseline signals
- Sets signal=0 when probability < threshold

---

## Lessons Learned

1. **Start simple:** Even a logistic regression reveals useful patterns
2. **Watch for data alignment:** Multi-asset stacking is tricky
3. **Threshold matters:** The τ parameter trades off return vs risk
4. **Baseline is strong:** The existing strategy has Sharpe ~2.4, hard to beat

---

## Next Steps

→ Phase 2: Add Glassnode on-chain features
- MVRV Z-score, SOPR, Fear & Greed
- These may capture "regime" information hinted in the challenge

→ Phase 3: Optimize label horizon and cost treatment
- Test different horizons (1 day, 3 days, 1 week)
- More realistic cost modeling
