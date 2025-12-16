"""
Performance metrics for strategy evaluation.
"""
import pandas as pd
import numpy as np
from typing import Optional


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 2920  # 3-hour periods in a year (24/3 * 365)
) -> float:
    """
    Compute annualized Sharpe ratio.

    Sharpe = (Mean Return - Risk Free Rate) / Std(Returns) * sqrt(periods_per_year)

    Args:
        returns: Series of period returns
        risk_free_rate: Annualized risk-free rate (default 0)
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized Sharpe ratio
    """
    if returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 2920
) -> float:
    """
    Compute annualized Sortino ratio (uses downside deviation only).

    Args:
        returns: Series of period returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0

    downside_std = downside_returns.std()
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute maximum drawdown from equity curve.

    Args:
        equity_curve: Series of cumulative portfolio values

    Returns:
        Maximum drawdown as a positive decimal (e.g., 0.25 = 25% drawdown)
    """
    rolling_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    return abs(drawdowns.min())


def compute_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 2920
) -> float:
    """
    Compute Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Series of period returns
        periods_per_year: Number of trading periods per year

    Returns:
        Calmar ratio
    """
    equity_curve = (1 + returns).cumprod()
    max_dd = compute_max_drawdown(equity_curve)

    if max_dd == 0:
        return np.inf if returns.mean() > 0 else 0.0

    annualized_return = (1 + returns.mean()) ** periods_per_year - 1
    return annualized_return / max_dd


def compute_win_rate(returns: pd.Series) -> float:
    """
    Compute win rate (fraction of positive returns).

    Args:
        returns: Series of returns

    Returns:
        Win rate as decimal
    """
    return (returns > 0).mean()


def compute_profit_factor(returns: pd.Series) -> float:
    """
    Compute profit factor (gross profits / gross losses).

    Args:
        returns: Series of returns

    Returns:
        Profit factor
    """
    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())

    if gross_losses == 0:
        return np.inf if gross_profits > 0 else 0.0

    return gross_profits / gross_losses


def compute_total_return(returns: pd.Series) -> float:
    """
    Compute total cumulative return.

    Args:
        returns: Series of period returns

    Returns:
        Total return as decimal (e.g., 1.5 = 150% total return)
    """
    return (1 + returns).prod() - 1


def compute_annualized_return(
    returns: pd.Series,
    periods_per_year: int = 2920
) -> float:
    """
    Compute annualized return.

    Args:
        returns: Series of period returns
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized return as decimal
    """
    total_return = compute_total_return(returns)
    n_periods = len(returns)
    n_years = n_periods / periods_per_year

    if n_years == 0:
        return 0.0

    return (1 + total_return) ** (1 / n_years) - 1


def compute_all_metrics(
    returns: pd.Series,
    periods_per_year: int = 2920
) -> dict:
    """
    Compute all performance metrics.

    Args:
        returns: Series of period returns
        periods_per_year: Number of trading periods per year

    Returns:
        Dictionary of all metrics
    """
    equity_curve = (1 + returns).cumprod()

    return {
        "total_return": compute_total_return(returns),
        "annualized_return": compute_annualized_return(returns, periods_per_year),
        "sharpe_ratio": compute_sharpe_ratio(returns, periods_per_year=periods_per_year),
        "sortino_ratio": compute_sortino_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": compute_max_drawdown(equity_curve),
        "calmar_ratio": compute_calmar_ratio(returns, periods_per_year),
        "win_rate": compute_win_rate(returns),
        "profit_factor": compute_profit_factor(returns),
        "n_trades": len(returns[returns != 0]),
        "volatility_annual": returns.std() * np.sqrt(periods_per_year)
    }


def compare_strategies(
    baseline_returns: pd.Series,
    improved_returns: pd.Series,
    periods_per_year: int = 2920
) -> pd.DataFrame:
    """
    Compare two strategies side by side.

    Args:
        baseline_returns: Returns from baseline strategy
        improved_returns: Returns from improved strategy
        periods_per_year: Number of trading periods per year

    Returns:
        DataFrame comparing both strategies
    """
    baseline_metrics = compute_all_metrics(baseline_returns, periods_per_year)
    improved_metrics = compute_all_metrics(improved_returns, periods_per_year)

    comparison = pd.DataFrame({
        "Baseline": baseline_metrics,
        "Improved": improved_metrics
    })

    # Add improvement column
    comparison["Improvement"] = comparison["Improved"] - comparison["Baseline"]
    comparison["Improvement %"] = (
        (comparison["Improved"] - comparison["Baseline"]) /
        comparison["Baseline"].abs().replace(0, np.nan) * 100
    )

    return comparison
