# config.py

# Backtest window
START_DATE = "2018-01-01"
END_DATE   = "2023-12-31"

# Risk-free rate (annualized) — approximate 3-month T-bill avg over period
RISK_FREE_RATE = 0.045

# Transaction cost per trade (round-trip applied on entry + exit)
TRANSACTION_COST = 0.001  # 0.1%

# Starting portfolio value
INITIAL_CAPITAL = 100_000

# Tickers
MOMENTUM_TICKERS  = ["SPY", "QQQ", "IWM"]
MEAN_REV_TICKERS  = ["AAPL", "MSFT", "GOOGL"]

# Candidate universe for pairs trading — the strategy screens these for
# cointegration (Engle-Granger test) and only trades a pair that passes.
# Grouped roughly by sector since cointegration is far more plausible
# within-sector (shared business drivers) than across unrelated sectors.
PAIRS_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL",   # mega-cap tech
    "KO", "PEP",               # consumer staples / beverages
    "XOM", "CVX",              # energy / oil majors
    "JPM", "BAC", "WFC",       # money-center banks
    "HD", "LOW",               # home improvement retail
]
PAIRS             = [("AAPL", "MSFT")]  # fallback default if screening finds nothing
BENCHMARK_TICKER  = "SPY"

# Strategy parameters
MOMENTUM_PARAMS = {
    "fast_window": 50,
    "slow_window": 200,
}

MEAN_REVERSION_PARAMS = {
    "window": 20,
    "n_std": 2.0,
}

PAIRS_PARAMS = {
    "lookback": 60,      # rolling window for spread z-score
    "entry_z": 2.0,
    "exit_z": 0.5,
}

# Monte Carlo
MC_SIMULATIONS = 10_000
MC_HORIZON     = 252  # trading days
