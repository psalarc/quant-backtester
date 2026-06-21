# strategies/mean_reversion.py

import pandas as pd
from strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Bollinger Band Mean Reversion.

    Signal logic:
        - Enter long (1) when price drops below lower band (mean - n*std)
        - Exit (0) when price reverts to or above the rolling mean
        - Stay flat while price is between bands

    Assumes short-term price overreaction in liquid equities. Works best
    in range-bound, low-trend regimes — the opposite of momentum.
    """

    def __init__(self, window: int = 20, n_std: float = 2.0):
        super().__init__(name=f"Bollinger_MeanReversion_w{window}_s{n_std}")
        self.window = window
        self.n_std = n_std

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices.iloc[:, 0] if isinstance(prices, pd.DataFrame) else prices

        rolling_mean = close.rolling(self.window).mean()
        rolling_std  = close.rolling(self.window).std()
        lower_band   = rolling_mean - self.n_std * rolling_std

        signal = pd.Series(0, index=close.index)
        in_trade = False

        for i in range(self.window, len(close)):
            if not in_trade and close.iloc[i] < lower_band.iloc[i]:
                in_trade = True
            elif in_trade and close.iloc[i] >= rolling_mean.iloc[i]:
                in_trade = False
            signal.iloc[i] = 1 if in_trade else 0

        return signal
