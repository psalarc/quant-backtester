# engine/data_loader.py

import os
import pandas as pd
import yfinance as yf


CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    price_col: str = "Adj Close",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Caches results locally as parquet to avoid redundant API calls.
    Returns a DataFrame indexed by date with tickers as columns.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_key = "_".join(sorted(tickers)) + f"_{start}_{end}.parquet"
    cache_path = os.path.join(CACHE_DIR, cache_key)

    if use_cache and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)

    # Recent yfinance versions return MultiIndex columns (field, ticker) even
    # for a single ticker. Flatten to plain ticker-named columns either way.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw[price_col].copy()
        if isinstance(prices, pd.Series):
            ticker = tickers if isinstance(tickers, str) else tickers[0]
            prices = prices.to_frame(name=ticker)
    else:
        ticker = tickers if isinstance(tickers, str) else tickers[0]
        prices = raw[[price_col]].rename(columns={price_col: ticker})

    prices = prices.dropna(how="all")

    if use_cache:
        prices.to_parquet(cache_path)

    return prices


def compute_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """Compute simple or log daily returns from a price DataFrame."""
    if log:
        import numpy as np
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()
