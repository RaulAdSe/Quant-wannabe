# Progress Tracker

This folder documents the progress through each phase of the Iqana Quant Challenge.

---

## Quick Status

| Phase | Status | Sharpe | Return | Key Achievement |
|-------|--------|--------|--------|-----------------|
| 0 - Data Exploration | Done | 2.39 (baseline) | 38% | Data aligned, baseline evaluated |
| 1 - Naive Model | Done | 3.93 (τ=0.6) | 12% | Working ML pipeline |
| 2 - Feature Engineering | Done | 3.16 (τ=0.6) | 51% | Glassnode + technicals |
| 3 - Horizon Optimization | Done | 3.05 (8p best) | 49% | 1-day horizon optimal |
| 4 - Model Comparison | **Done** | **3.26** | **52%** | **Random Forest wins** |
| 5 - Regime Detection | Pending | - | - | - |
| 6 - Robustness | Pending | - | - | - |

**Best Result So Far:** Phase 4 with Sharpe 3.26 AND 52% return (Random Forest, 8p horizon)

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

4. **[PHASE_3_4_optimization.md](PHASE_3_4_optimization.md)**
   - Horizon optimization (1d, 2d, 3d, 1w)
   - Model comparison (LogReg, RF, GB)
   - Random Forest achieves best balance

5. **PHASE_5_regimes.md** *(Coming Soon)*
   - Hidden Markov Models
   - Regime detection

6. **PHASE_6_robustness.md** *(Coming Soon)*
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

| Phase | Model | Sharpe | Return | Trade-off |
|-------|-------|--------|--------|-----------|
| Phase 1 (τ=0.6) | LogReg | 3.93 | 11.71% | High Sharpe, low return |
| Phase 2 (τ=0.6) | LogReg | 3.16 | 50.88% | Better balance |
| **Phase 4 (τ=0.5)** | **Random Forest** | **3.26** | **52.1%** | **Best overall** |

**Key Insight:** Random Forest with on-chain features achieves +21.4% Sharpe improvement vs baseline while INCREASING returns.

---

## Problems Solved

| Problem | Phase | Solution |
|---------|-------|----------|
| AAVE missing from prices | 0 | Excluded (5 assets remain) |
| Timestamp alignment | 0 | Index intersection |
| NaN in feature stacking | 1 | Asset-agnostic column names |
| Timezone mismatch | 2 | Localize to naive timestamps |
| Daily vs 3-hourly data | 2 | Forward-fill Glassnode values |
| XGBoost libomp missing | 3&4 | Used GradientBoostingClassifier |
| GB underperformance | 3&4 | Selected Random Forest instead |

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
# 4. 04_model_optimization.ipynb
```
