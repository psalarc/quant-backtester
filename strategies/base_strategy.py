# strategies/base_strategy.py

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """
    All strategies inherit from this class and implement generate_signals().

    generate_signals() takes a price DataFrame and returns a Series of
    integer signals indexed by date: 1 = long, 0 = flat.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Return a pd.Series of signals (0 or 1) aligned to prices.index."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
