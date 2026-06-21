# risk/monte_carlo.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


class MonteCarloSimulator:
    """
    Bootstrap-based Monte Carlo stress tester for strategy returns.

    Resamples observed daily returns with replacement to generate N simulated
    equity paths over a given horizon. This makes no assumptions about the
    return distribution — it uses the empirical distribution directly.

    Borrowed conceptually from actuarial reserve risk modeling: we want to
    know not just the expected outcome, but the full distribution of outcomes.
    """

    def __init__(
        self,
        returns: pd.Series,
        n_simulations: int = 10_000,
        horizon: int = 252,
        initial_value: float = 100_000,
        seed: Optional[int] = 42,
    ):
        self.returns = returns.dropna().values
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.initial_value = initial_value
        # Use a local Generator (rather than the global np.random state) so the
        # seed only affects this simulator and doesn't leak into the rest of the
        # program. seed=42 gives reproducible figures; pass seed=None for true
        # randomness.
        self.rng = np.random.default_rng(seed)
        self.sim_paths: Optional[np.ndarray] = None

    def run(self) -> np.ndarray:
        """
        Run simulations. Returns array of shape (n_simulations, horizon+1)
        representing equity paths starting at initial_value.
        """
        sampled = self.rng.choice(
            self.returns,
            size=(self.n_simulations, self.horizon),
            replace=True,
        )
        # Cumulative product of (1 + r) gives the growth factor
        growth = np.cumprod(1 + sampled, axis=1)
        paths = self.initial_value * np.hstack(
            [np.ones((self.n_simulations, 1)), growth]
        )
        self.sim_paths = paths
        return paths

    def percentiles(self, levels: list[float] = [5, 25, 50, 75, 95]) -> pd.DataFrame:
        """Return equity path percentiles at each time step."""
        if self.sim_paths is None:
            raise RuntimeError("Call .run() before accessing percentiles.")
        return pd.DataFrame(
            np.percentile(self.sim_paths, levels, axis=0).T,
            columns=[f"p{p}" for p in levels],
        )

    def prob_of_loss(self) -> float:
        """Fraction of simulations ending below initial value."""
        if self.sim_paths is None:
            raise RuntimeError("Call .run() before accessing results.")
        final_values = self.sim_paths[:, -1]
        return float((final_values < self.initial_value).mean())

    def expected_shortfall(self, confidence: float = 0.95) -> float:
        """CVaR: expected value of the worst (1-confidence) fraction of outcomes."""
        if self.sim_paths is None:
            raise RuntimeError("Call .run() before accessing results.")
        final_returns = (self.sim_paths[:, -1] / self.initial_value) - 1
        cutoff = np.percentile(final_returns, (1 - confidence) * 100)
        tail = final_returns[final_returns <= cutoff]
        return float(tail.mean())

    def plot_fan_chart(
        self,
        strategy_name: str = "Strategy",
        save_path: Optional[str] = None,
    ):
        """
        Fan chart showing the distribution of simulated equity paths.
        5th/25th/50th/75th/95th percentile bands.
        """
        if self.sim_paths is None:
            raise RuntimeError("Call .run() before plotting.")

        pcts = self.percentiles([5, 25, 50, 75, 95])
        x = np.arange(self.horizon + 1)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.fill_between(x, pcts["p5"], pcts["p95"], alpha=0.15, color="#2563eb", label="5th–95th pct")
        ax.fill_between(x, pcts["p25"], pcts["p75"], alpha=0.25, color="#2563eb", label="25th–75th pct")
        ax.plot(x, pcts["p50"], color="#2563eb", lw=2, label="Median path")
        ax.axhline(self.initial_value, color="gray", lw=1, linestyle="--", label="Initial capital")

        ax.set_title(
            f"Monte Carlo Simulation — {strategy_name}\n"
            f"{self.n_simulations:,} simulations · {self.horizon}-day horizon · "
            f"P(loss) = {self.prob_of_loss():.1%}",
            fontsize=12,
        )
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(loc="upper left", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved: {save_path}")
        plt.show()
        plt.close()
