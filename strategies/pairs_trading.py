# strategies/pairs_trading.py

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

from strategies.base_strategy import BaseStrategy


class PairsTradingStrategy(BaseStrategy):
    """
    Cointegration-based statistical arbitrage on a pair of equities.

    Methodology:
        1. Estimate the hedge ratio (beta) via OLS regression on a rolling window.
        2. Compute the spread: spread = price_A - beta * price_B
        3. Compute the rolling z-score of the spread.
        4. Enter long the spread when z-score < -entry_z (spread is cheap).
        5. Exit when z-score reverts toward zero (< exit_z in abs terms).

    The ADF test is run on the full in-sample spread to verify cointegration
    before committing to the strategy. This is a pre-trade sanity check, not
    a guarantee of out-of-sample stationarity.

    Note: This implementation trades a single leg (the spread as a synthetic asset)
    for simplicity. A full implementation would go long asset A and short asset B.
    """

    def __init__(
        self,
        ticker_a: str,
        ticker_b: str,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ):
        super().__init__(name=f"PairsTrading_{ticker_a}_{ticker_b}")
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

# strategies/pairs_trading.py

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint

from strategies.base_strategy import BaseStrategy


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    significance: float = 0.05,
) -> list[tuple[str, str, float]]:
    """
    Screen all pairs of columns in `prices` for cointegration using the
    Engle-Granger two-step test (via statsmodels.tsa.stattools.coint).

    Returns a list of (ticker_a, ticker_b, p_value) tuples for pairs that
    pass the significance threshold, sorted by p-value ascending (most
    confidently cointegrated first).

    This is the step that was missing originally: instead of assuming a
    pair (e.g. AAPL/MSFT) is cointegrated, we test candidates and only
    trade ones that actually pass.
    """
    tickers = prices.columns.tolist()
    candidates = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            series_a = prices[a].dropna()
            series_b = prices[b].dropna()
            aligned = pd.concat([series_a, series_b], axis=1).dropna()
            if len(aligned) < 100:
                continue

            _, p_value, _ = coint(aligned[a], aligned[b])
            if p_value < significance:
                candidates.append((a, b, p_value))

    return sorted(candidates, key=lambda x: x[2])


class PairsTradingStrategy(BaseStrategy):
    """
    Cointegration-based statistical arbitrage on a pair of equities.

    Methodology:
        1. Verify cointegration via the Engle-Granger test (coint) up front.
           If the pair fails, the strategy stays flat for the entire period
           rather than trading a spread with no statistical basis.
        2. Estimate the hedge ratio (beta) via OLS on a ROLLING window —
           not the full sample — to avoid look-ahead bias.
        3. Compute the rolling z-score of the spread: price_A - beta * price_B.
        4. Enter long the spread when z-score < -entry_z (spread is cheap
           relative to its recent history).
        5. Exit when the z-score reverts to within exit_z of zero.

    Note: This implementation trades a single synthetic leg (the spread)
    for simplicity, sized as if it were one asset. A full implementation
    would explicitly go long asset A and short beta-weighted units of asset B.
    """

    def __init__(
        self,
        ticker_a: str,
        ticker_b: str,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        require_cointegration: bool = True,
        cointegration_significance: float = 0.05,
    ):
        super().__init__(name=f"PairsTrading_{ticker_a}_{ticker_b}")
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.require_cointegration = require_cointegration
        self.cointegration_significance = cointegration_significance
        self.is_cointegrated_: bool | None = None
        self.cointegration_pvalue_: float | None = None

    def _check_cointegration(self, price_a: pd.Series, price_b: pd.Series) -> bool:
        """Engle-Granger test on the full sample. Used as a pre-trade gate only."""
        _, p_value, _ = coint(price_a, price_b)
        self.cointegration_pvalue_ = p_value
        passed = p_value < self.cointegration_significance

        if passed:
            print(f"  [OK] {self.ticker_a}/{self.ticker_b} cointegrated (Engle-Granger p = {p_value:.4f})")
        else:
            print(
                f"  [Warning] {self.ticker_a}/{self.ticker_b} NOT cointegrated "
                f"(Engle-Granger p = {p_value:.4f}, threshold = {self.cointegration_significance}). "
                f"This pair lacks statistical basis for mean-reversion trading."
            )
        return passed

    def _rolling_hedge_ratio(self, price_a: pd.Series, price_b: pd.Series) -> pd.Series:
        """
        Estimate beta on a trailing rolling window so the hedge ratio only
        uses information available up to each point in time (no look-ahead).
        """
        betas = pd.Series(index=price_a.index, dtype=float)

        for i in range(self.lookback, len(price_a)):
            window_a = price_a.iloc[i - self.lookback : i]
            window_b = price_b.iloc[i - self.lookback : i]
            model = OLS(window_a, add_constant(window_b)).fit()
            betas.iloc[i] = model.params[window_b.name]

        return betas

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if self.ticker_a not in prices.columns or self.ticker_b not in prices.columns:
            raise ValueError(
                f"Prices DataFrame must contain columns '{self.ticker_a}' and '{self.ticker_b}'."
            )

        price_a = prices[self.ticker_a].dropna()
        price_b = prices[self.ticker_b].dropna()
        aligned = pd.concat([price_a, price_b], axis=1).dropna()
        price_a, price_b = aligned[self.ticker_a], aligned[self.ticker_b]

        self.is_cointegrated_ = self._check_cointegration(price_a, price_b)

        if self.require_cointegration and not self.is_cointegrated_:
            print(f"  Strategy will stay flat for the full period — no valid statistical edge.")
            return pd.Series(0, index=prices.index)

        # Rolling hedge ratio (no look-ahead)
        beta = self._rolling_hedge_ratio(price_a, price_b)
        spread = price_a - beta * price_b

        spread_mean = spread.rolling(self.lookback).mean()
        spread_std  = spread.rolling(self.lookback).std()
        z_score     = (spread - spread_mean) / spread_std

        signal = pd.Series(0, index=prices.index)
        in_trade = False

        for date in z_score.index:
            z = z_score.loc[date]
            if pd.isna(z):
                continue
            if not in_trade and z < -self.entry_z:
                in_trade = True
            elif in_trade and abs(z) < self.exit_z:
                in_trade = False
            signal.loc[date] = 1 if in_trade else 0

        return signal.reindex(prices.index, fill_value=0)
