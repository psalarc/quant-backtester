# main.py

"""
Quantitative Backtester — Entry Point

Runs all three strategies against historical data, computes risk metrics,
generates result plots, and prints a performance summary table.

Usage:
    python main.py
    python main.py --strategy momentum --ticker SPY --start 2018-01-01 --end 2023-12-31
"""

import os
import argparse
import pandas as pd

import config
from engine.data_loader import fetch_prices, compute_returns
from engine.backtester import Backtester
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.pairs_trading import PairsTradingStrategy, find_cointegrated_pairs
from risk.metrics import summarize
from risk.monte_carlo import MonteCarloSimulator
from visualization.plots import (
    plot_equity_curves,
    plot_drawdown,
    plot_return_distribution,
    plot_metrics_table,
)

os.makedirs("results", exist_ok=True)


def run_momentum(start, end):
    print("\n[1/3] Running Momentum Strategy (SMA Crossover)...")
    prices = fetch_prices(["SPY"], start=start, end=end)
    strategy = MomentumStrategy(**config.MOMENTUM_PARAMS)
    bt = Backtester(strategy, prices, config.INITIAL_CAPITAL, config.TRANSACTION_COST)
    portfolio = bt.run()
    return portfolio, strategy.name


def run_mean_reversion(start, end):
    print("\n[2/3] Running Mean Reversion Strategy (Bollinger Bands)...")
    prices = fetch_prices(["AAPL"], start=start, end=end)
    strategy = MeanReversionStrategy(**config.MEAN_REVERSION_PARAMS)
    bt = Backtester(strategy, prices, config.INITIAL_CAPITAL, config.TRANSACTION_COST)
    portfolio = bt.run()
    return portfolio, strategy.name


def run_pairs(start, end):
    print("\n[3/3] Running Pairs Trading Strategy...")
    print(f"  Screening {len(config.PAIRS_UNIVERSE)} candidate tickers for cointegrated pairs...")

    universe_prices = fetch_prices(config.PAIRS_UNIVERSE, start=start, end=end)
    candidates = find_cointegrated_pairs(universe_prices, significance=0.05)

    if candidates:
        ticker_a, ticker_b, p_value = candidates[0]
        print(f"  Found {len(candidates)} cointegrated pair(s). Trading best: "
              f"{ticker_a}/{ticker_b} (Engle-Granger p = {p_value:.4f})")
    else:
        ticker_a, ticker_b = config.PAIRS[0]
        print(f"  No pairs in the universe passed cointegration screening. "
              f"Falling back to {ticker_a}/{ticker_b} (strategy will likely stay flat).")

    prices = fetch_prices([ticker_a, ticker_b], start=start, end=end)
    strategy = PairsTradingStrategy(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        **config.PAIRS_PARAMS,
    )
    # Use price_a as the proxy equity for portfolio tracking
    prices_a = prices[[ticker_a]]
    signals = strategy.generate_signals(prices)

    from engine.portfolio import Portfolio
    portfolio = Portfolio(config.INITIAL_CAPITAL, config.TRANSACTION_COST)
    portfolio.reset()
    for date, row in prices_a.iterrows():
        price = float(row.iloc[0])
        signal = signals.loc[date] if date in signals.index else 0
        portfolio.update(date, price, int(signal))

    return portfolio, strategy.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all", choices=["all", "momentum", "mean_reversion", "pairs"])
    parser.add_argument("--start", default=config.START_DATE)
    parser.add_argument("--end", default=config.END_DATE)
    args = parser.parse_args()

    start, end = args.start, args.end
    results = {}

    if args.strategy in ("all", "momentum"):
        port, name = run_momentum(start, end)
        results["Momentum"] = port

    if args.strategy in ("all", "mean_reversion"):
        port, name = run_mean_reversion(start, end)
        results["Mean Reversion"] = port

    if args.strategy in ("all", "pairs"):
        port, name = run_pairs(start, end)
        results["Pairs Trading"] = port

    # Benchmark
    print("\nFetching benchmark (SPY buy & hold)...")
    spy_prices = fetch_prices(["SPY"], start=start, end=end)
    spy_equity = spy_prices["SPY"] / spy_prices["SPY"].iloc[0] * config.INITIAL_CAPITAL

    # Equity curves
    equity_curves = {name: port.get_equity_curve() for name, port in results.items()}
    plot_equity_curves(
        equity_curves,
        benchmark=spy_equity,
        title="Strategy Equity Curves vs. SPY Benchmark",
        save_path="results/equity_curves.png",
    )

    # Per-strategy: drawdown + return distribution + Monte Carlo
    for name, port in results.items():
        returns = port.get_returns()
        equity  = port.get_equity_curve()

        plot_drawdown(equity, strategy_name=name, save_path=f"results/drawdown_{name.lower().replace(' ', '_')}.png")
        plot_return_distribution(returns, strategy_name=name, save_path=f"results/returns_{name.lower().replace(' ', '_')}.png")

        print(f"\n  Running Monte Carlo for {name}...")
        mc = MonteCarloSimulator(returns, n_simulations=config.MC_SIMULATIONS, horizon=config.MC_HORIZON)
        mc.run()
        mc.plot_fan_chart(strategy_name=name, save_path=f"results/mc_{name.lower().replace(' ', '_')}.png")
        print(f"  Probability of loss over {config.MC_HORIZON} days: {mc.prob_of_loss():.1%}")

    # Summary metrics table
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    summaries = []
    for name, port in results.items():
        s = summarize(port.get_returns(), port.get_equity_curve(), name=name)
        summaries.append(s)

    # Add benchmark
    spy_returns = compute_returns(spy_prices)["SPY"]
    summaries.append(summarize(spy_returns, spy_equity, name="SPY (Buy & Hold)"))

    summary_df = pd.DataFrame(summaries)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("results/performance_summary.csv", index=False)

    plot_metrics_table(summary_df, save_path="results/metrics_table.png")
    print("\nAll results saved to results/")


if __name__ == "__main__":
    main()
