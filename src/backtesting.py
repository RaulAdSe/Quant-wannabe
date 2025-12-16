"""
Backtesting utilities for strategy evaluation.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def compute_strategy_returns(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    transaction_cost: float = 0.001
) -> pd.DataFrame:
    """
    Compute returns for a long/cash strategy with transaction costs.

    IMPORTANT: Signals at time t determine position for period t to t+1.
    Returns are computed from price at t to price at t+1.

    Args:
        signals: DataFrame of signals (0=cash, 1=long) with same index as prices
        prices: DataFrame of asset prices
        transaction_cost: Cost per trade as decimal (e.g., 0.001 = 0.1%)

    Returns:
        DataFrame of strategy returns per asset
    """
    # Compute asset returns (price change from t to t+1)
    asset_returns = prices.pct_change().shift(-1)  # Forward returns

    # Align signals and returns
    common_cols = signals.columns.intersection(prices.columns)
    signals_aligned = signals[common_cols]
    returns_aligned = asset_returns[common_cols]

    # Strategy returns: get asset return when signal is 1, else 0
    strategy_returns = signals_aligned * returns_aligned

    # Compute transaction costs (cost when signal changes)
    signal_changes = signals_aligned.diff().abs()
    costs = signal_changes * transaction_cost

    # Net returns
    net_returns = strategy_returns - costs

    return net_returns


def compute_portfolio_returns(
    strategy_returns: pd.DataFrame,
    weights: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Compute portfolio returns from individual asset returns.

    Args:
        strategy_returns: DataFrame of returns per asset
        weights: Optional DataFrame of weights (default: equal weight)

    Returns:
        Series of portfolio returns
    """
    if weights is None:
        # Equal weight among assets with valid returns
        n_assets = strategy_returns.notna().sum(axis=1)
        weights = strategy_returns.notna().astype(float).div(n_assets, axis=0)

    portfolio_returns = (strategy_returns * weights).sum(axis=1)
    return portfolio_returns


def compute_equity_curve(returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    """
    Compute equity curve from returns.

    Args:
        returns: Series of period returns
        initial_capital: Starting capital

    Returns:
        Series of portfolio values over time
    """
    return initial_capital * (1 + returns).cumprod()


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Compute drawdown series from equity curve.

    Args:
        equity_curve: Series of portfolio values

    Returns:
        Series of drawdowns (negative values)
    """
    rolling_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    return drawdowns


def apply_ml_filter(
    baseline_signals: pd.DataFrame,
    probabilities: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Apply ML filter to baseline signals.

    signal_ml[t,a] = signal_base[t,a] × 1[p̂(t,a) > τ]

    Args:
        baseline_signals: DataFrame of baseline signals (0/1)
        probabilities: DataFrame of predicted probabilities
        threshold: Probability threshold τ

    Returns:
        DataFrame of filtered signals
    """
    # Create indicator for probability above threshold
    above_threshold = (probabilities > threshold).astype(int)

    # Apply filter: only take position when baseline says long AND model is confident
    filtered_signals = baseline_signals * above_threshold

    return filtered_signals


def walk_forward_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    model_func,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    train_size: int,
    test_size: int,
    step_size: Optional[int] = None,
    transaction_cost: float = 0.001,
    threshold: float = 0.5
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Perform walk-forward backtesting.

    Args:
        signals: Baseline trading signals
        prices: Asset prices
        model_func: Function that takes (X_train, y_train) and returns fitted model
        features: Feature DataFrame
        labels: Label DataFrame
        train_size: Number of periods for training
        test_size: Number of periods for testing
        step_size: Step size for rolling window (default: test_size)
        transaction_cost: Transaction cost per trade
        threshold: Probability threshold for ML filter

    Returns:
        Tuple of (filtered_signals, portfolio_returns, fold_info)
    """
    if step_size is None:
        step_size = test_size

    n_periods = len(features)
    all_predictions = []
    fold_info = []

    start_idx = 0
    fold_num = 0

    while start_idx + train_size + test_size <= n_periods:
        train_end = start_idx + train_size
        test_end = train_end + test_size

        # Split data
        X_train = features.iloc[start_idx:train_end]
        y_train = labels.iloc[start_idx:train_end]
        X_test = features.iloc[train_end:test_end]

        # Train model and predict
        model = model_func(X_train, y_train)
        predictions = model.predict_proba(X_test)[:, 1]

        # Store predictions with proper index
        pred_series = pd.Series(predictions, index=X_test.index)
        all_predictions.append(pred_series)

        # Store fold info
        fold_info.append({
            "fold": fold_num,
            "train_start": features.index[start_idx],
            "train_end": features.index[train_end - 1],
            "test_start": features.index[train_end],
            "test_end": features.index[test_end - 1],
            "train_size": train_size,
            "test_size": test_size
        })

        start_idx += step_size
        fold_num += 1

    # Combine all predictions
    all_predictions = pd.concat(all_predictions)

    # Create probability DataFrame matching signals shape
    probabilities = pd.DataFrame(
        index=all_predictions.index,
        columns=signals.columns
    )
    for col in signals.columns:
        probabilities[col] = all_predictions

    # Apply ML filter
    filtered_signals = apply_ml_filter(
        signals.loc[probabilities.index],
        probabilities,
        threshold
    )

    # Compute returns
    strategy_returns = compute_strategy_returns(
        filtered_signals,
        prices.loc[probabilities.index],
        transaction_cost
    )
    portfolio_returns = compute_portfolio_returns(strategy_returns)

    return filtered_signals, portfolio_returns, fold_info
