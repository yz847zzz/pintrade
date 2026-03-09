import pandas as pd
import numpy as np

def sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    """
    Calculates the annualized Sharpe Ratio.
    Assumes daily returns by default (252 trading days in a year).
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    return (returns.mean() * periods) / (returns.std() * np.sqrt(periods))

def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculates the maximum drawdown of an equity curve.
    """
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def annualized_return(equity_curve: pd.Series, periods: int = 252) -> float:
    """
    Calculates the annualized return from an equity curve.
    Assumes daily data by default (252 trading days in a year).
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    num_years = len(equity_curve) / periods
    if num_years <= 0:
        return 0.0
    return (1 + total_return)**(1 / num_years) - 1