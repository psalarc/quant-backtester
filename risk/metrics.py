# risk/metrics.py

import numpy as np
import pandas as pd


def _as_series(x) -> pd.Series:
    """
    Coerce a single-column DataFrame to a Series. Guards against upstream
    data sources (e.g. yfinance) occasionally returning a 1-column DataFrame
    where a Series is expected.
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"Expected a Series or single-column DataFrame, got shape {x.shape}")
        return x.iloc[:, 0]
    return x


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.045, periods: int = 252) -> float:
    """Annualized Sharpe ratio."""
    returns = _as_series(returns)
    excess = returns - risk_free_rate / periods
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.045, periods: int = 252) -> float:
    """
    Annualized Sortino ratio — like Sharpe, but only penalizes downside volatility.
    More relevant for strategies with asymmetric return distributions.
    """
    returns = _as_series(returns)
    excess = returns - risk_free_rate / periods
    downside = excess[excess < 0].std()
    if downside == 0:
        return 0.0
    return float((excess.mean() / downside) * np.sqrt(periods))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough decline in the equity curve."""
    equity_curve = _as_series(equity_curve)
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series, periods: int = 252) -> float:
    """Annualized return divided by absolute max drawdown."""
    ann_return = annualized_return(returns, periods)
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return float(ann_return / mdd)


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    """Compound annualized growth rate from daily returns."""
    returns = _as_series(returns)
    total = (1 + returns).prod()
    n_years = len(returns) / periods
    return float(total ** (1 / n_years) - 1)


def annualized_volatility(returns: pd.Series, periods: int = 252) -> float:
    returns = _as_series(returns)
    return float(returns.std() * np.sqrt(periods))


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (returns a positive loss value)."""
    returns = _as_series(returns)
    return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))


def summarize(
    returns: pd.Series,
    equity_curve: pd.Series,
    risk_free_rate: float = 0.045,
    name: str = "Strategy",
) -> pd.Series:
    """Compute the full risk metrics summary for a strategy."""
    return pd.Series(
        {
            "Strategy": name,
            "Ann. Return": f"{annualized_return(returns):.1%}",
            "Ann. Volatility": f"{annualized_volatility(returns):.1%}",
            "Sharpe Ratio": f"{sharpe_ratio(returns, risk_free_rate):.2f}",
            "Sortino Ratio": f"{sortino_ratio(returns, risk_free_rate):.2f}",
            "Max Drawdown": f"{max_drawdown(equity_curve):.1%}",
            "Calmar Ratio": f"{calmar_ratio(returns, equity_curve):.2f}",
            "VaR (95%)": f"{value_at_risk(returns):.2%}",
        }
    )
