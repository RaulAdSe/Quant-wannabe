"""
Feature engineering utilities.

IMPORTANT: All features must be computed using only information
available at time t (no lookahead bias).
"""
import pandas as pd
import numpy as np
from typing import List, Optional


def compute_rolling_returns(
    prices: pd.DataFrame,
    windows: List[int] = [1, 8, 56, 224]
) -> pd.DataFrame:
    """
    Compute rolling returns over multiple windows.

    Args:
        prices: DataFrame of prices
        windows: List of lookback periods (in 3-hour units)
            1 = 3h, 8 = 1 day, 56 = 1 week, 224 = 1 month

    Returns:
        DataFrame with return features
    """
    features = pd.DataFrame(index=prices.index)

    for window in windows:
        for col in prices.columns:
            feature_name = f"{col}_return_{window}p"
            features[feature_name] = prices[col].pct_change(window)

    return features


def compute_rolling_volatility(
    prices: pd.DataFrame,
    windows: List[int] = [56, 224]
) -> pd.DataFrame:
    """
    Compute rolling volatility (std of returns) over multiple windows.

    Args:
        prices: DataFrame of prices
        windows: List of lookback periods

    Returns:
        DataFrame with volatility features
    """
    returns = prices.pct_change()
    features = pd.DataFrame(index=prices.index)

    for window in windows:
        for col in prices.columns:
            feature_name = f"{col}_volatility_{window}p"
            features[feature_name] = returns[col].rolling(window).std()

    return features


def compute_moving_averages(
    prices: pd.DataFrame,
    windows: List[int] = [56, 224, 672]
) -> pd.DataFrame:
    """
    Compute price relative to moving averages.

    Args:
        prices: DataFrame of prices
        windows: List of MA periods (56=1 week, 224=1 month, 672=3 months)

    Returns:
        DataFrame with MA features (price/MA ratio)
    """
    features = pd.DataFrame(index=prices.index)

    for window in windows:
        for col in prices.columns:
            ma = prices[col].rolling(window).mean()
            feature_name = f"{col}_ma_ratio_{window}p"
            features[feature_name] = prices[col] / ma

    return features


def compute_rsi(
    prices: pd.DataFrame,
    window: int = 112  # 14 days in 3-hour periods
) -> pd.DataFrame:
    """
    Compute Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss

    Args:
        prices: DataFrame of prices
        window: Lookback period

    Returns:
        DataFrame with RSI features (0-100 scale)
    """
    features = pd.DataFrame(index=prices.index)

    for col in prices.columns:
        delta = prices[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        features[f"{col}_rsi_{window}p"] = rsi

    return features


def compute_bollinger_bands(
    prices: pd.DataFrame,
    window: int = 160,  # ~20 days
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Compute Bollinger Band position.

    Returns position within bands: (price - lower) / (upper - lower)
    0 = at lower band, 1 = at upper band

    Args:
        prices: DataFrame of prices
        window: Lookback period for MA and std
        num_std: Number of standard deviations for bands

    Returns:
        DataFrame with BB position features (0-1 scale, can exceed)
    """
    features = pd.DataFrame(index=prices.index)

    for col in prices.columns:
        ma = prices[col].rolling(window).mean()
        std = prices[col].rolling(window).std()

        upper = ma + num_std * std
        lower = ma - num_std * std

        bb_position = (prices[col] - lower) / (upper - lower)
        features[f"{col}_bb_position_{window}p"] = bb_position

    return features


def compute_momentum_features(
    prices: pd.DataFrame,
    windows: List[int] = [8, 56, 224]
) -> pd.DataFrame:
    """
    Compute momentum indicators (rate of change).

    Args:
        prices: DataFrame of prices
        windows: Lookback periods

    Returns:
        DataFrame with momentum features
    """
    features = pd.DataFrame(index=prices.index)

    for window in windows:
        for col in prices.columns:
            # Rate of change
            roc = (prices[col] - prices[col].shift(window)) / prices[col].shift(window)
            features[f"{col}_roc_{window}p"] = roc

    return features


def select_glassnode_features(
    glassnode: pd.DataFrame,
    key_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Select and clean key Glassnode features.

    Args:
        glassnode: Full Glassnode DataFrame
        key_features: List of feature names to select (None = use defaults)

    Returns:
        DataFrame with selected features
    """
    if key_features is None:
        key_features = [
            "btc_mvrv_z_score",
            "btc_adjusted_sopr",
            "btc_fear_greed_index",
            "reserve_risk",
            "btc_puell_multiple",
            "btc_percent_upply_in_profit",
            "btc_futures_perpetual_funding_rate_mean",
            "btc_stablecoin_supply_ratio_oscillator"
        ]

    # Select features that exist
    available = [f for f in key_features if f in glassnode.columns]
    selected = glassnode[available].copy()

    # Forward fill missing values (conservative approach)
    selected = selected.ffill()

    return selected


def create_feature_matrix(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    glassnode: Optional[pd.DataFrame] = None,
    include_signal: bool = True
) -> pd.DataFrame:
    """
    Create full feature matrix for modeling.

    IMPORTANT: All features use only information available at time t.

    Args:
        prices: Price DataFrame
        signals: Signal DataFrame
        glassnode: Optional Glassnode DataFrame
        include_signal: Whether to include current signal as feature

    Returns:
        Feature matrix with all engineered features
    """
    feature_dfs = []

    # Price-based features
    feature_dfs.append(compute_rolling_returns(prices))
    feature_dfs.append(compute_rolling_volatility(prices))
    feature_dfs.append(compute_moving_averages(prices))
    feature_dfs.append(compute_rsi(prices))
    feature_dfs.append(compute_bollinger_bands(prices))
    feature_dfs.append(compute_momentum_features(prices))

    # Signal features
    if include_signal:
        signal_features = signals.add_prefix("signal_")
        feature_dfs.append(signal_features)

    # Glassnode features (if provided)
    if glassnode is not None:
        gn_features = select_glassnode_features(glassnode)
        gn_features = gn_features.add_prefix("gn_")
        feature_dfs.append(gn_features)

    # Combine all features
    features = pd.concat(feature_dfs, axis=1)

    return features


def prepare_ml_data(
    features: pd.DataFrame,
    labels: pd.Series,
    dropna: bool = True
) -> tuple:
    """
    Prepare data for ML modeling by aligning features and labels.

    Args:
        features: Feature DataFrame
        labels: Label Series
        dropna: Whether to drop rows with missing values

    Returns:
        Tuple of (X, y) with aligned indices
    """
    # Align indices
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]

    if dropna:
        # Find rows with no NaN in features
        valid_rows = ~X.isna().any(axis=1)
        X = X[valid_rows]
        y = y[valid_rows]

    return X, y
