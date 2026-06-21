# strategies/momentum.py

import pandas as pd
from strategies.base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Dual Simple Moving Average Crossover (Golden Cross / Death Cross).

    Signal logic:
        - Long (1) when fast SMA > slow SMA
        - Flat (0) otherwise

    Classic trend-following heuristic. Works best in trending, low-chop regimes.
    No shorting — this version is long-or-flat only.
    """

    def __init__(self, fast_window: int = 50, slow_window: int = 200):
        super().__init__(name=f"SMA_Crossover_{fast_window}_{slow_window}")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices.iloc[:, 0] if isinstance(prices, pd.DataFrame) else prices

        sma_fast = close.rolling(self.fast_window).mean()
        sma_slow = close.rolling(self.slow_window).mean()

        signal = (sma_fast > sma_slow).astype(int)

        # Avoid trading until both windows are warm
        signal.iloc[: self.slow_window] = 0

        return signal
