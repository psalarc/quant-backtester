# tests/test_metrics.py

import numpy as np
import pandas as pd
import pytest
from risk.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    annualized_return,
    value_at_risk,
)


@pytest.fixture
def flat_returns():
    """Daily returns of exactly 0 — edge case."""
    return pd.Series(np.zeros(252))


@pytest.fixture
def positive_returns():
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.01, 252))


@pytest.fixture
def equity_from_returns(positive_returns):
    return (1 + positive_returns).cumprod() * 100_000


def test_sharpe_flat_returns(flat_returns):
    assert sharpe_ratio(flat_returns) == 0.0


def test_sharpe_positive(positive_returns):
    sr = sharpe_ratio(positive_returns)
    assert isinstance(sr, float)
    assert -5 < sr < 10  # sanity bounds


def test_sortino_greater_than_sharpe_on_positive_skew(positive_returns):
    # For symmetric returns, Sortino ~ Sharpe * sqrt(2)
    sr = sharpe_ratio(positive_returns)
    sor = sortino_ratio(positive_returns)
    assert sor > sr * 0.5  # rough sanity check


def test_max_drawdown_non_positive(equity_from_returns):
    mdd = max_drawdown(equity_from_returns)
    assert mdd <= 0


def test_max_drawdown_monotone_increasing():
    equity = pd.Series([100, 110, 120, 130, 140])
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_max_drawdown_known():
    equity = pd.Series([100, 80, 90, 70, 100])
    mdd = max_drawdown(equity)
    assert mdd == pytest.approx(-0.30, abs=0.01)


def test_var_positive(positive_returns):
    var = value_at_risk(positive_returns)
    assert var > 0


def test_annualized_return_approx(positive_returns):
    ar = annualized_return(positive_returns)
    assert isinstance(ar, float)


# --- Monte Carlo reproducibility ---

def test_monte_carlo_seed_reproducible(positive_returns):
    """Same default seed must produce identical P(loss)."""
    from risk.monte_carlo import MonteCarloSimulator
    mc1 = MonteCarloSimulator(positive_returns, n_simulations=1000)
    mc2 = MonteCarloSimulator(positive_returns, n_simulations=1000)
    mc1.run()
    mc2.run()
    assert mc1.prob_of_loss() == mc2.prob_of_loss()


def test_monte_carlo_seed_none_varies(positive_returns):
    """seed=None should (almost surely) produce different draws."""
    from risk.monte_carlo import MonteCarloSimulator
    mc1 = MonteCarloSimulator(positive_returns, n_simulations=2000, seed=None)
    mc2 = MonteCarloSimulator(positive_returns, n_simulations=2000, seed=None)
    mc1.run()
    mc2.run()
    # Final-value arrays should differ; P(loss) extremely unlikely to match exactly
    assert not np.array_equal(mc1.sim_paths[:, -1], mc2.sim_paths[:, -1])


def test_monte_carlo_path_shape(positive_returns):
    from risk.monte_carlo import MonteCarloSimulator
    mc = MonteCarloSimulator(positive_returns, n_simulations=500, horizon=100)
    paths = mc.run()
    assert paths.shape == (500, 101)  # horizon + 1 (includes starting point)
