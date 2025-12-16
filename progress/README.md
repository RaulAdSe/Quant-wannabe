# Progress Tracker

This folder documents the progress through each phase of the Iqana Quant Challenge.

---

## Quick Status

| Phase | Status | Sharpe | Return | Key Achievement |
|-------|--------|--------|--------|-----------------|
| 0 - Data Exploration | Done | 2.39 (baseline) | 38% | Data aligned, baseline evaluated |
| 1 - Naive Model | Done | 3.93 (τ=0.6) | 12% | Working ML pipeline |
| 2 - Feature Engineering | **Done** | **3.16** (τ=0.6) | **51%** | **Glassnode + technicals** |
| 3 - Labels & Costs | Pending | - | - | - |
| 4 - Better Models | Pending | - | - | - |
| 5 - Regime Detection | Pending | - | - | - |
| 6 - Robustness | Pending | - | - | - |

**Best Result So Far:** Phase 2 with Sharpe 3.16 AND 51% return (best risk-adjusted returns)

---

## Phase Documents

1. **[PHASE_0_data_exploration.md](PHASE_0_data_exploration.md)**
   - Data loading and alignment
   - Baseline strategy evaluation
   - Key metrics established

2. **[PHASE_1_naive_model.md](PHASE_1_naive_model.md)**
   - End-to-end ML pipeline
   - Walk-forward validation
   - First filtering results

3. **[PHASE_2_enhanced_features.md](PHASE_2_enhanced_features.md)**
   - Glassnode on-chain features (10 metrics)
   - Technical indicators (RSI, Bollinger Bands)
   - +17.5% Sharpe improvement with +24% more return

4. **PHASE_3_labels.md** *(Coming Soon)*
   - Horizon optimization
   - Cost-adjusted labels

5. **PHASE_4_models.md** *(Coming Soon)*
   - XGBoost, Random Forest
   - Model comparison

6. **PHASE_5_regimes.md** *(Coming Soon)*
   - Hidden Markov Models
   - Regime detection

7. **PHASE_6_robustness.md** *(Coming Soon)*
   - Failure analysis
   - Final evaluation

---

## Git Commits Log

| Commit | Description |
|--------|-------------|
| `6cf3bcd` | Initial project setup |
| `26217c5` | Phase 0: Data exploration notebook |
| `6bde619` | Phase 1: Naive model + environment |
| `10a64bc` | Fix: NaN in feature stacking |
| `529c6eb` | Add progress documentation |
| `1f5ee33` | Phase 2: Enhanced feature engineering |

---

## Key Metrics Tracked

### Baseline Performance
- **Sharpe Ratio:** 2.69 (test period, with 0.1% costs)
- **Total Return:** 41.02%
- **Max Drawdown:** ~45%

### Best ML-Filtered Performance

| Phase | Sharpe | Return | Trade-off |
|-------|--------|--------|-----------|
| Phase 1 (τ=0.6) | 3.93 | 11.71% | High Sharpe, low return |
| **Phase 2 (τ=0.6)** | **3.16** | **50.88%** | **Best balance** |

**Phase 2 Insight:** Adding Glassnode features enables the model to identify GOOD trades to keep, not just filter conservatively.

---

## Problems Solved

| Problem | Phase | Solution |
|---------|-------|----------|
| AAVE missing from prices | 0 | Excluded (5 assets remain) |
| Timestamp alignment | 0 | Index intersection |
| NaN in feature stacking | 1 | Asset-agnostic column names |
| Timezone mismatch | 2 | Localize to naive timestamps |
| Daily vs 3-hourly data | 2 | Forward-fill Glassnode values |

---

## Feature Evolution

| Phase | Features | Description |
|-------|----------|-------------|
| 1 | 4 | Returns, volatility |
| 2 | 16 | + RSI, BB, Glassnode (10 metrics) |

---

## Running the Code

```bash
# Setup environment
cd "/Users/rauladell/Work/Quant challenge"
source venv/bin/activate

# Run notebooks in order
jupyter lab
# 1. 01_data_exploration.ipynb
# 2. 02_naive_model.ipynb
# 3. 03_enhanced_features.ipynb
```
