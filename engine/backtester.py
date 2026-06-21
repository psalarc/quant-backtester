# engine/backtester.py

import pandas as pd
from engine.portfolio import Portfolio
from strategies.base_strategy import BaseStrategy


class Backtester:
    """
    Drives a strategy through historical price data bar-by-bar.

    Intentionally simple: no look-ahead bias protection beyond signal generation
    being called before portfolio.update(). For research use.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        prices: pd.DataFrame,
        initial_capital: float = 100_000,
        transaction_cost: float = 0.001,
    ):
        self.strategy = strategy
        self.prices = prices
        self.portfolio = Portfolio(initial_capital, transaction_cost)

    def run(self) -> Portfolio:
        self.portfolio.reset()

        # Generate signals for the full price series up front
        signals = self.strategy.generate_signals(self.prices)

        for date, row in self.prices.iterrows():
            price = row.iloc[0] if isinstance(row, pd.Series) else float(row)
            signal = signals.loc[date] if date in signals.index else 0
            self.portfolio.update(date, price, int(signal))

        return self.portfolio
