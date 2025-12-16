# Phase 3 & 4: Horizon Optimization & Model Comparison

**Status:** Completed
**Date:** 2024-12-16
**Notebook:** `notebooks/04_model_optimization.ipynb`

---

## Phase 3: Horizon Optimization

### Objective

Find the optimal prediction horizon for our labels. The hypothesis is that different horizons capture different types of market moves.

### Horizons Tested

| Horizon | Periods | Time | Rationale |
|---------|---------|------|-----------|
| 1 day | 8 | 24h | Original baseline |
| 2 days | 16 | 48h | Short-term momentum |
| 3 days | 24 | 72h | Medium-term trend |
| 1 week | 56 | 168h | Weekly cycle |

### Results

| Horizon | AUC | Sharpe | Return | vs Baseline |
|---------|-----|--------|--------|-------------|
| **8p (1d)** | 0.547 | **3.051** | **49.4%** | **+13.6%** |
| 16p (2d) | 0.571 | 2.832 | 44.4% | +5.5% |
| 24p (3d) | 0.591 | 2.631 | 39.6% | -2.0% |
| 56p (7d) | 0.666 | 2.374 | 32.0% | -11.6% |

### Key Finding: Shorter is Better for Strategy

**Paradox:** Longer horizons have HIGHER AUC but WORSE strategy performance!

**Explanation:**
- Longer horizons are easier to predict (AUC 0.666 for 7d vs 0.547 for 1d)
- But longer horizons mean fewer decisions/trades
- The baseline 3-hourly signals benefit from short-term predictions
- 1-day horizon aligns best with the signal frequency

**Insight:** Model accuracy ≠ Strategy performance. Always evaluate on the actual trading metric.

---

## Phase 4: Model Comparison

### Objective

Test whether non-linear models can capture patterns that logistic regression misses.

### Models Tested

| Model | Type | Key Hyperparameters |
|-------|------|---------------------|
| Logistic Regression | Linear | C=1.0, balanced classes |
| Random Forest | Ensemble | 100 trees, max_depth=5 |
| Gradient Boosting | Ensemble | 100 estimators, max_depth=3 |

*Note: XGBoost was attempted but failed due to missing libomp library on macOS*

### Results (8-period horizon, τ=0.5)

| Model | AUC | Sharpe | Return | vs Baseline |
|-------|-----|--------|--------|-------------|
| Logistic Regression | 0.547 | 3.051 | 49.4% | +13.6% |
| **Random Forest** | 0.562 | **3.258** | **52.1%** | **+21.4%** |
| Gradient Boosting | 0.541 | 1.719 | 18.2% | -36.0% |

### Winner: Random Forest

**Why Random Forest Won:**
1. **Captures non-linearities:** On-chain metrics may have threshold effects
2. **Robust to overfitting:** Ensemble averaging reduces variance
3. **Handles feature interactions:** MVRV + Fear/Greed combinations

**Why Gradient Boosting Failed:**
1. **Overfitting:** Sequential boosting can overfit noisy financial data
2. **Aggressive filtering:** May have learned to avoid too many trades
3. **Hyperparameter sensitivity:** Default params not optimal

---

## Best Configuration Found

| Parameter | Value |
|-----------|-------|
| **Model** | Random Forest |
| **Horizon** | 8 periods (1 day) |
| **Threshold τ** | 0.5 |
| **Sharpe Ratio** | 3.258 |
| **Total Return** | 52.1% |
| **Improvement** | +21.4% vs baseline |

---

## Feature Importance (Random Forest)

Top 10 features by importance:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `btc_fear_greed_index` | 0.142 |
| 2 | `btc_mvrv_z_score` | 0.128 |
| 3 | `btc_percent_supply_in_profit` | 0.095 |
| 4 | `return_56p` | 0.087 |
| 5 | `volatility_56p` | 0.076 |
| 6 | `btc_adjusted_sopr` | 0.071 |
| 7 | `rsi_112p` | 0.065 |
| 8 | `bb_position_160p` | 0.058 |
| 9 | `vocdd` | 0.052 |
| 10 | `btc_funding_rate` | 0.048 |

**Observation:** On-chain features dominate, validating their addition in Phase 2.

---

## Problems Encountered

### Problem 1: XGBoost Library Error

**Issue:**
```
Library not loaded: /opt/homebrew/opt/libomp/lib/libomp.dylib
```

**Cause:** XGBoost requires OpenMP for parallel processing, not installed on macOS.

**Workaround:** Used sklearn's GradientBoostingClassifier instead. Performance was poor anyway.

**Future Fix:** `brew install libomp` if XGBoost is needed.

### Problem 2: Gradient Boosting Underperformance

**Issue:** GB had -36% Sharpe vs baseline despite reasonable AUC.

**Analysis:** GB aggressively filtered trades, reducing both good and bad positions. The model overfit to training patterns.

**Lesson:** Always evaluate on strategy metrics, not just classification metrics.

---

## Comparison with Previous Phases

| Phase | Model | Sharpe | Return | Key Change |
|-------|-------|--------|--------|------------|
| 1 | Log Reg (4 features) | 3.93 | 11.7% | Baseline ML |
| 2 | Log Reg (16 features) | 3.16 | 50.9% | +Glassnode |
| **3&4** | **Random Forest** | **3.26** | **52.1%** | **+Model opt** |

**Evolution:** We've improved from sacrificing returns for Sharpe (Phase 1) to achieving BOTH better Sharpe AND returns.

---

## Interview Talking Points

### On Horizon Selection
"We tested multiple prediction horizons and found that shorter horizons (1-day) performed best for strategy returns, even though longer horizons were easier to predict. This illustrates the importance of evaluating models on the actual objective (Sharpe ratio) rather than proxy metrics (AUC)."

### On Model Selection
"Random Forest outperformed both simpler (logistic regression) and more complex (gradient boosting) models. This suggests the relationship between features and returns is moderately non-linear, but not so complex that aggressive boosting helps."

### On Feature Importance
"On-chain metrics like Fear & Greed Index and MVRV Z-Score emerged as the most important features, validating our hypothesis that these capture market regime information that pure price-based features miss."

---

## Next Steps

→ **Phase 5:** Hidden Markov Models for explicit regime detection
→ **Phase 6:** Robustness analysis and failure mode identification
