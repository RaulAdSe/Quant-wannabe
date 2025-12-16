"""
Data loading and preprocessing utilities.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "docs" / "datasets"


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three datasets with proper datetime parsing.

    Returns:
        Tuple of (trade_log, prices, glassnode) DataFrames
    """
    trade_log = load_trade_log()
    prices = load_prices()
    glassnode = load_glassnode()

    return trade_log, prices, glassnode


def load_trade_log() -> pd.DataFrame:
    """Load trade log with timestamp index."""
    df = pd.read_csv(DATA_DIR / "trade_log.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def load_prices() -> pd.DataFrame:
    """Load price data with timestamp index."""
    df = pd.read_csv(DATA_DIR / "price_data.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def load_glassnode() -> pd.DataFrame:
    """Load Glassnode on-chain metrics with timestamp index."""
    df = pd.read_csv(DATA_DIR / "glassnode_metrics.csv")
    # Handle the date format (appears to be M/D/YY format)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.set_index("timestamp").sort_index()
    return df


def compute_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Compute simple returns over specified periods.

    Args:
        prices: DataFrame of prices
        periods: Number of periods for return calculation

    Returns:
        DataFrame of returns (same shape as prices)
    """
    return prices.pct_change(periods)


def compute_log_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Compute log returns over specified periods.

    Args:
        prices: DataFrame of prices
        periods: Number of periods for return calculation

    Returns:
        DataFrame of log returns
    """
    return np.log(prices / prices.shift(periods))


def align_datasets(
    trade_log: pd.DataFrame,
    prices: pd.DataFrame,
    glassnode: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align all datasets to common timestamps.

    IMPORTANT: This ensures no lookahead bias by using only
    information available at each timestamp.

    Args:
        trade_log: Trading signals
        prices: Asset prices
        glassnode: On-chain metrics

    Returns:
        Tuple of aligned DataFrames
    """
    # Find common date range
    start_date = max(
        trade_log.index.min(),
        prices.index.min(),
        glassnode.index.min()
    )
    end_date = min(
        trade_log.index.max(),
        prices.index.max(),
        glassnode.index.max()
    )

    # Filter to common range
    trade_log_aligned = trade_log.loc[start_date:end_date]
    prices_aligned = prices.loc[start_date:end_date]
    glassnode_aligned = glassnode.loc[start_date:end_date]

    return trade_log_aligned, prices_aligned, glassnode_aligned


def get_data_summary(df: pd.DataFrame, name: str) -> dict:
    """
    Generate summary statistics for a DataFrame.

    Args:
        df: DataFrame to summarize
        name: Name for the dataset

    Returns:
        Dictionary of summary statistics
    """
    return {
        "name": name,
        "shape": df.shape,
        "date_range": (df.index.min(), df.index.max()),
        "columns": list(df.columns),
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing_pct": (df.isna().sum() / len(df) * 100).to_dict(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2
    }
