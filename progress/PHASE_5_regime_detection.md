# Phase 5: Regime Detection with Hidden Markov Models

**Status:** Completed
**Date:** 2024-12-16
**Notebook:** `notebooks/05_regime_detection.ipynb`

---

## Objective

Use Hidden Markov Models to identify market regimes (bull/bear/sideways) and incorporate this information to improve trading strategy performance.

---

## Hidden Markov Model Setup

### Model Configuration
| Parameter | Value |
|-----------|-------|
| States | 3 (Bull, Sideways, Bear) |
| Features | BTC returns + volatility |
| Covariance | Full |
| Iterations | 100 |

### Why BTC as Market Proxy?
- BTC dominates crypto market sentiment
- All altcoins are correlated with BTC
- BTC has the most liquidity and price discovery

---

## Regime Detection Results

### State Characteristics

| Regime | Mean Return | Annualized Sharpe | % of Data |
|--------|-------------|-------------------|-----------|
| **Bull** | +3.40% | 138.6 | 10.5% |
| Sideways | -0.03% | -0.4 | 30.8% |
| Bear | -0.44% | -17.5 | 58.6% |

**Key Finding:** The test period was predominantly in Bear regime (58.6%), which explains why avoiding Bear trades has such impact.

### Regime Persistence

The transition matrix reveals regimes are "sticky":
- Bull tends to persist with high self-transition probability
- Transitions between regimes are relatively rare
- Average regime duration: several days to weeks

---

## Strategy Approaches Tested

### Approach 1: Regime as Feature

Add regime one-hot encoding to the ML model's feature set.

| Configuration | AUC | Sharpe | Return | vs Baseline |
|---------------|-----|--------|--------|-------------|
| RF (no regime) | 0.536 | 3.258 | 48.3% | +21.4% |
| RF (with regime) | 0.525 | 2.640 | 34.3% | -1.6% |

**Finding:** Adding regime as a feature HURTS performance! The model overfits to regime information.

### Approach 2: Regime-Conditional Trading

Use regime as a hard filter - only trade in allowed regimes.

| Configuration | Sharpe | Return | Trade Reduction | vs Baseline |
|---------------|--------|--------|-----------------|-------------|
| All regimes (ML only) | 3.258 | 48.3% | 23.4% | +21.4% |
| **Bull + Sideways** | **6.464** | **70.0%** | 79.1% | **+140.8%** |
| Bull only | 5.394 | 42.0% | 91.0% | +100.9% |

**MAJOR FINDING:** Avoiding Bear market trades dramatically improves performance!

---

## Best Configuration Found

| Parameter | Value |
|-----------|-------|
| **Strategy** | RF + Bull/Sideways Filter |
| **Sharpe Ratio** | 6.464 |
| **Total Return** | 70.0% |
| **Improvement** | +140.8% vs baseline |
| **Trade Reduction** | 79.1% |

This is by far the best result in all phases!

---

## Comparison with Previous Phases

| Phase | Configuration | Sharpe | Return | vs Baseline |
|-------|---------------|--------|--------|-------------|
| 1 | LogReg (4 features) | 3.93 | 11.7% | +64% |
| 2 | LogReg (16 features) | 3.16 | 50.9% | +17.5% |
| 4 | Random Forest | 3.26 | 52.1% | +21.4% |
| **5** | **RF + Regime Filter** | **6.46** | **70.0%** | **+140.8%** |

**Phase 5 delivers 2x the Sharpe of Phase 4!**

---

## Why Regime Filtering Works

### Analysis

1. **Bear regime = 58.6% of test period**: Most of the data is in unfavorable conditions
2. **Avoiding Bear saves bad trades**: The model can't predict well in Bear markets
3. **Quality over quantity**: 79% fewer trades, but much better trades
4. **Risk management**: Sitting out during downturns preserves capital

### Physical Intuition

Think of regimes as **thermodynamic phases**:
- **Bull** = High energy, expansive phase (favorable for longs)
- **Sideways** = Equilibrium state (neutral)
- **Bear** = Contracting phase (unfavorable for longs)

The HMM identifies which "phase" the market is in. Just like you wouldn't try to boil water at sub-zero temperatures, you shouldn't try to profit from longs in a Bear market.

---

## Problems Encountered

### Problem 1: Timezone Mismatch

**Issue:**
```
TypeError: Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp
```

**Cause:** Regimes from HMM had timezone-aware index (UTC), signals had timezone-naive index.

**Solution:** Localize regimes to naive timestamps before comparison:
```python
regimes_labeled_naive = regimes_labeled.copy()
regimes_labeled_naive.index = regimes_labeled_naive.index.tz_localize(None)
```

### Problem 2: Regime as Feature Hurts Performance

**Issue:** Adding regime dummies decreased Sharpe from 3.26 to 2.64.

**Analysis:** The model likely overfit to the regime labels, creating spurious correlations.

**Resolution:** Use regime as a filter, not a feature. This works better because it's a binary decision (trade/don't trade) rather than a continuous weight.

---

## Interview Talking Points

### On HMM for Regime Detection
"We used a Gaussian Hidden Markov Model with 3 states to identify latent market regimes. The model found distinct bull, bear, and sideways states based on BTC returns and volatility distributions."

### On the Key Finding
"The most impactful discovery was that regime-conditional trading - avoiding trades during Bear markets - improved Sharpe by 140%. This demonstrates that knowing WHEN not to trade can be as important as knowing when to trade."

### On Feature vs Filter
"Interestingly, adding regime as a model feature hurt performance, but using it as a hard filter dramatically helped. This suggests regime is best used for binary trade/no-trade decisions rather than as a continuous predictor."

### Physics Analogy
"Think of market regimes like phases in thermodynamics. The HMM identifies which phase the market is in - bull (expansive), bear (contracting), or sideways (equilibrium). Just as physical systems behave differently in different phases, trading strategies should adapt to market phases. The key insight is that some phases are fundamentally unfavorable for certain strategies."

---

## Next Steps

-> **Phase 6:** Robustness analysis
- Test across different time periods
- Analyze failure modes
- Stress test the regime filter approach
