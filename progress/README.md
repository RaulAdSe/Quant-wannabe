# Progress Tracker

This folder documents the progress through each phase of the Iqana Quant Challenge.

---

## Quick Status

| Phase | Status | Sharpe Improvement | Key Achievement |
|-------|--------|-------------------|-----------------|
| 0 - Data Exploration | Done | - | Baseline: 2.39 Sharpe |
| 1 - Naive Model | Done | +64% at τ=0.6 | Working ML pipeline |
| 2 - Feature Engineering | Pending | - | - |
| 3 - Labels & Costs | Pending | - | - |
| 4 - Better Models | Pending | - | - |
| 5 - Regime Detection | Pending | - | - |
| 6 - Robustness | Pending | - | - |

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

3. **PHASE_2_features.md** *(Coming Soon)*
   - Glassnode on-chain features
   - Technical indicators

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

---

## Key Metrics Tracked

### Baseline Performance
- **Sharpe Ratio:** 2.39 (with 0.1% transaction costs)
- **Total Return:** 38.44% (test period)
- **Max Drawdown:** ~55%

### Best ML-Filtered Performance (So Far)
- **Sharpe Ratio:** 3.93 (at τ=0.6)
- **Total Return:** 11.71%
- **Improvement:** +64% Sharpe, -70% trades

---

## Problems Solved

| Problem | Phase | Solution |
|---------|-------|----------|
| AAVE missing from prices | 0 | Excluded (5 assets) |
| Timestamp alignment | 0 | Index intersection |
| NaN in feature stacking | 1 | Asset-agnostic column names |

---

## Running the Code

```bash
# Setup environment
cd "/Users/rauladell/Work/Quant challenge"
source venv/bin/activate

# Run notebooks in order
jupyter lab
# Open: 01_data_exploration.ipynb
# Open: 02_naive_model.ipynb
```
