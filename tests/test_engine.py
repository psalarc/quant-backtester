# tests/test_engine.py

import numpy as np
import pandas as pd
import pytest

from engine.portfolio import Portfolio
from engine.backtester import Backtester
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.pairs_trading import PairsTradingStrategy, find_cointegrated_pairs


def make_trending_prices(n=300, start=100.0, drift=0.001):
    """Synthetically trending price series."""
    np.random.seed(0)
    returns = np.random.normal(drift, 0.01, n)
    prices = start * np.cumprod(1 + returns)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"MOCK": prices}, index=dates)


def make_mean_reverting_prices(n=300, mu=100.0, theta=0.1, sigma=1.5):
    """Ornstein-Uhlenbeck process — a classic mean-reverting series."""
    np.random.seed(1)
    prices = [mu]
    for _ in range(n - 1):
        dp = theta * (mu - prices[-1]) + sigma * np.random.randn()
        prices.append(prices[-1] + dp)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"MOCK": prices}, index=dates)


class TestPortfolio:
    def test_initial_equity_equals_capital(self):
        port = Portfolio(initial_capital=100_000)
        port.update(pd.Timestamp("2020-01-01"), price=100.0, signal=0)
        equity = port.get_equity_curve()
        assert equity.iloc[0] == pytest.approx(100_000)

    def test_buy_then_sell_reduces_cash_then_restores(self):
        port = Portfolio(initial_capital=100_000, transaction_cost=0.0)
        port.update(pd.Timestamp("2020-01-01"), price=100.0, signal=1)
        assert port.cash == pytest.approx(0.0)
        port.update(pd.Timestamp("2020-01-02"), price=110.0, signal=0)
        assert port.cash == pytest.approx(110_000 * 1.0)  # no cost

    def test_transaction_cost_reduces_final_equity(self):
        port_no_cost   = Portfolio(100_000, transaction_cost=0.0)
        port_with_cost = Portfolio(100_000, transaction_cost=0.01)

        for port in [port_no_cost, port_with_cost]:
            port.update(pd.Timestamp("2020-01-01"), price=100.0, signal=1)
            port.update(pd.Timestamp("2020-01-02"), price=110.0, signal=0)

        assert port_no_cost.cash > port_with_cost.cash

    def test_n_trades_correct(self):
        port = Portfolio(100_000)
        signals = [1, 1, 0, 1, 0]
        for i, sig in enumerate(signals):
            port.update(pd.Timestamp(f"2020-01-0{i+1}"), price=100.0, signal=sig)
        assert port.n_trades() == 2


class TestMomentumStrategy:
    def test_no_signal_before_slow_window(self):
        prices = make_trending_prices(300)
        strategy = MomentumStrategy(fast_window=50, slow_window=200)
        signals = strategy.generate_signals(prices)
        assert (signals.iloc[:200] == 0).all()

    def test_generates_some_trades_on_trending_data(self):
        prices = make_trending_prices(500)
        strategy = MomentumStrategy(fast_window=50, slow_window=200)
        bt = Backtester(strategy, prices)
        port = bt.run()
        assert port.n_trades() >= 1

    def test_equity_curve_length_matches_prices(self):
        prices = make_trending_prices(300)
        strategy = MomentumStrategy()
        bt = Backtester(strategy, prices)
        port = bt.run()
        assert len(port.get_equity_curve()) == len(prices)


class TestMeanReversionStrategy:
    def test_enters_on_dip(self):
        prices = make_mean_reverting_prices(500)
        strategy = MeanReversionStrategy(window=20, n_std=1.5)
        signals = strategy.generate_signals(prices)
        assert signals.sum() > 0  # should enter at least once

    def test_signal_is_binary(self):
        prices = make_mean_reverting_prices(300)
        strategy = MeanReversionStrategy()
        signals = strategy.generate_signals(prices)
        assert set(signals.unique()).issubset({0, 1})


def make_cointegrated_pair(n=300, mu=100.0, theta=0.05, sigma=1.0, beta=1.5):
    """
    Two series that are genuinely cointegrated by construction: B follows a
    random walk, and A = beta * B + a stationary (mean-reverting) noise term.
    The spread A - beta*B is then guaranteed stationary.
    """
    np.random.seed(2)
    b_returns = np.random.normal(0.0003, 0.01, n)
    price_b = mu * np.cumprod(1 + b_returns)

    spread = [0.0]
    for _ in range(n - 1):
        d = theta * (0 - spread[-1]) + sigma * np.random.randn()
        spread.append(spread[-1] + d)
    spread = np.array(spread)

    price_a = beta * price_b + spread
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": price_a, "B": price_b}, index=dates)


def make_independent_pair(n=300):
    """Two unrelated random walks — should NOT be cointegrated."""
    np.random.seed(3)
    price_a = 100 * np.cumprod(1 + np.random.normal(0.0008, 0.015, n))
    price_b = 50 * np.cumprod(1 + np.random.normal(-0.0002, 0.012, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": price_a, "B": price_b}, index=dates)


class TestPairsTrading:
    def test_detects_genuine_cointegration(self):
        prices = make_cointegrated_pair(400)
        strategy = PairsTradingStrategy(ticker_a="A", ticker_b="B", lookback=60)
        strategy.generate_signals(prices)
        assert strategy.is_cointegrated_ == True
        assert strategy.cointegration_pvalue_ < 0.05

    def test_stays_flat_when_not_cointegrated(self):
        prices = make_independent_pair(400)
        strategy = PairsTradingStrategy(
            ticker_a="A", ticker_b="B", lookback=60, require_cointegration=True
        )
        signals = strategy.generate_signals(prices)
        # Strategy should refuse to trade a non-cointegrated pair
        if not strategy.is_cointegrated_:
            assert (signals == 0).all()

    def test_signal_binary_on_cointegrated_pair(self):
        prices = make_cointegrated_pair(400)
        strategy = PairsTradingStrategy(ticker_a="A", ticker_b="B", lookback=60)
        signals = strategy.generate_signals(prices)
        assert set(signals.unique()).issubset({0, 1})

    def test_signal_length_matches_prices(self):
        prices = make_cointegrated_pair(400)
        strategy = PairsTradingStrategy(ticker_a="A", ticker_b="B", lookback=60)
        signals = strategy.generate_signals(prices)
        assert len(signals) == len(prices)

    def test_missing_ticker_raises(self):
        prices = make_cointegrated_pair(300)
        strategy = PairsTradingStrategy(ticker_a="A", ticker_b="NOT_PRESENT")
        with pytest.raises(ValueError):
            strategy.generate_signals(prices)

    def test_find_cointegrated_pairs_identifies_known_pair(self):
        prices = make_cointegrated_pair(400)
        results = find_cointegrated_pairs(prices, significance=0.05)
        assert len(results) >= 1
        a, b, p = results[0]
        assert p < 0.05
