# engine/portfolio.py

import numpy as np
import pandas as pd


class Portfolio:
    """
    Tracks positions, cash, and equity over a backtest.

    Designed to be driven by a Strategy that emits signals (-1, 0, 1).
    Applies a flat transaction cost on every position change.
    """

    def __init__(self, initial_capital: float = 100_000, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.position = 0        # shares held (can be fractional for simplicity)
        self.equity_curve = []
        self.trade_log = []

    def update(self, date, price: float, signal: int):
        """
        Process one bar. Signal: 1 = long, 0 = flat, -1 = short (not implemented).
        Position changes trigger a transaction cost.
        """
        prev_position = self.position

        if signal == 1 and self.position == 0:
            # Enter long: invest all cash
            cost = self.cash * self.transaction_cost
            shares = (self.cash - cost) / price
            self.position = shares
            self.cash = 0.0
            self.trade_log.append({"date": date, "action": "BUY", "price": price, "shares": shares})

        elif signal == 0 and self.position > 0:
            # Exit to cash
            proceeds = self.position * price
            cost = proceeds * self.transaction_cost
            self.cash = proceeds - cost
            self.trade_log.append({"date": date, "action": "SELL", "price": price, "shares": self.position})
            self.position = 0.0

        # Mark-to-market equity
        equity = self.cash + self.position * price
        self.equity_curve.append({"date": date, "equity": equity})

    def get_equity_curve(self) -> pd.Series:
        df = pd.DataFrame(self.equity_curve).set_index("date")
        return df["equity"]

    def get_trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log)

    def get_returns(self) -> pd.Series:
        equity = self.get_equity_curve()
        return equity.pct_change().dropna()

    def total_return(self) -> float:
        equity = self.get_equity_curve()
        return (equity.iloc[-1] / equity.iloc[0]) - 1

    def n_trades(self) -> int:
        log = self.get_trade_log()
        return len(log[log["action"] == "BUY"]) if not log.empty else 0
