"""
Label creation utilities for supervised learning.

IMPORTANT: Labels are created using FUTURE information (that's what we predict),
but they must be properly aligned so that at prediction time we don't have access
to the label's future data.
"""
import pandas as pd
import numpy as np
from typing import Optional


def create_forward_return_labels(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Create binary labels based on forward returns.

    Label = 1 if forward return > threshold, else 0

    Args:
        prices: Price DataFrame
        signals: Signal DataFrame (to filter only when signal=1)
        horizon: Number of periods to look ahead
        threshold: Return threshold (e.g., 0.001 for 0.1% cost)

    Returns:
        DataFrame of binary labels (same shape as signals)
    """
    # Compute forward returns
    forward_returns = prices.pct_change(horizon).shift(-horizon)

    # Align columns
    common_cols = signals.columns.intersection(prices.columns)

    # Create labels only where signal is 1
    labels = pd.DataFrame(index=signals.index, columns=common_cols)

    for col in common_cols:
        # Only create label when baseline signal is 1
        mask = signals[col] == 1
        returns_when_long = forward_returns[col].where(mask)

        # Binary label: 1 if return exceeds threshold
        labels[col] = (returns_when_long > threshold).astype(float)

        # Set NaN where signal is 0 (we don't need predictions there)
        labels.loc[~mask, col] = np.nan

    return labels


def create_cost_adjusted_labels(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    horizon: int = 8,
    entry_cost: float = 0.001,
    exit_cost: float = 0.001
) -> pd.DataFrame:
    """
    Create labels accounting for transaction costs.

    A trade is profitable if: forward_return > entry_cost + exit_cost

    Args:
        prices: Price DataFrame
        signals: Signal DataFrame
        horizon: Holding period
        entry_cost: Cost to enter position
        exit_cost: Cost to exit position

    Returns:
        DataFrame of binary labels
    """
    total_cost = entry_cost + exit_cost
    return create_forward_return_labels(prices, signals, horizon, total_cost)


def create_risk_adjusted_labels(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    horizon: int = 8,
    threshold: float = 0.0,
    vol_window: int = 56
) -> pd.DataFrame:
    """
    Create labels based on risk-adjusted returns.

    Label = 1 if return / volatility > threshold

    Args:
        prices: Price DataFrame
        signals: Signal DataFrame
        horizon: Forward return horizon
        threshold: Risk-adjusted return threshold
        vol_window: Window for volatility estimation

    Returns:
        DataFrame of binary labels
    """
    # Compute forward returns
    forward_returns = prices.pct_change(horizon).shift(-horizon)

    # Compute rolling volatility (using past data only)
    returns = prices.pct_change()
    volatility = returns.rolling(vol_window).std()

    # Risk-adjusted returns
    risk_adj_returns = forward_returns / volatility

    # Align columns
    common_cols = signals.columns.intersection(prices.columns)

    # Create labels
    labels = pd.DataFrame(index=signals.index, columns=common_cols)

    for col in common_cols:
        mask = signals[col] == 1
        adj_returns_when_long = risk_adj_returns[col].where(mask)
        labels[col] = (adj_returns_when_long > threshold).astype(float)
        labels.loc[~mask, col] = np.nan

    return labels


def create_stacked_labels(
    labels_df: pd.DataFrame,
    signals_df: pd.DataFrame
) -> pd.Series:
    """
    Stack asset-specific labels into a single Series for modeling.

    This is useful when training a single model across all assets.

    Args:
        labels_df: DataFrame of labels (timestamp x asset)
        signals_df: DataFrame of signals (to filter)

    Returns:
        Series with MultiIndex (timestamp, asset)
    """
    # Stack and drop NaN
    stacked = labels_df.stack()
    stacked.index.names = ["timestamp", "asset"]

    # Drop NaN values
    stacked = stacked.dropna()

    return stacked


def analyze_label_distribution(
    labels: pd.DataFrame,
    signals: pd.DataFrame
) -> dict:
    """
    Analyze label distribution for class imbalance.

    Args:
        labels: DataFrame of labels
        signals: DataFrame of signals

    Returns:
        Dictionary with label statistics
    """
    # Overall statistics
    all_labels = labels.values.flatten()
    all_labels = all_labels[~np.isnan(all_labels)]

    # Per-asset statistics
    per_asset = {}
    for col in labels.columns:
        col_labels = labels[col].dropna()
        per_asset[col] = {
            "total": len(col_labels),
            "positive": (col_labels == 1).sum(),
            "negative": (col_labels == 0).sum(),
            "positive_rate": (col_labels == 1).mean() if len(col_labels) > 0 else 0
        }

    return {
        "overall": {
            "total": len(all_labels),
            "positive": (all_labels == 1).sum(),
            "negative": (all_labels == 0).sum(),
            "positive_rate": (all_labels == 1).mean() if len(all_labels) > 0 else 0,
            "imbalance_ratio": (all_labels == 0).sum() / max((all_labels == 1).sum(), 1)
        },
        "per_asset": per_asset
    }
