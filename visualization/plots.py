# visualization/plots.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
BLUE = "#2563eb"
RED  = "#dc2626"
GRAY = "#6b7280"


def plot_equity_curves(
    curves: dict[str, pd.Series],
    benchmark: pd.Series | None = None,
    title: str = "Strategy Equity Curves",
    save_path: str | None = None,
):
    """Overlay equity curves for multiple strategies + optional benchmark."""
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = [BLUE, "#16a34a", "#9333ea", "#ea580c"]
    for (name, curve), color in zip(curves.items(), colors):
        normalized = curve / curve.iloc[0] * 100
        ax.plot(normalized.index, normalized, label=name, lw=1.8, color=color)

    if benchmark is not None:
        norm_bench = benchmark / benchmark.iloc[0] * 100
        ax.plot(norm_bench.index, norm_bench, label="SPY (benchmark)",
                lw=1.2, color=GRAY, linestyle="--", alpha=0.8)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalized Value (base = 100)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_drawdown(
    equity_curve: pd.Series,
    strategy_name: str = "Strategy",
    save_path: str | None = None,
):
    """Plot the rolling drawdown from peak equity."""
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(equity_curve.index, equity_curve, color=BLUE, lw=1.5)
    axes[0].set_title(f"{strategy_name} — Equity Curve & Drawdown", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.4, color=RED)
    axes[1].plot(drawdown.index, drawdown, color=RED, lw=0.8)
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_return_distribution(
    returns: pd.Series,
    strategy_name: str = "Strategy",
    save_path: str | None = None,
):
    """Histogram of daily returns with VaR line."""
    from risk.metrics import value_at_risk

    var_95 = value_at_risk(returns)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(returns * 100, bins=60, ax=ax, color=BLUE, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(-var_95 * 100, color=RED, lw=1.5, linestyle="--", label=f"VaR (95%) = {var_95:.2%}")
    ax.axvline(returns.mean() * 100, color=GRAY, lw=1.2, linestyle="-", label=f"Mean = {returns.mean():.3%}")

    ax.set_title(f"{strategy_name} — Daily Return Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_metrics_table(
    summary_df: pd.DataFrame,
    save_path: str | None = None,
):
    """Render a clean metrics comparison table as a figure."""
    fig, ax = plt.subplots(figsize=(10, len(summary_df) * 0.7 + 1))
    ax.axis("off")

    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(summary_df.columns)):
        table[0, j].set_facecolor("#1e3a5f")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            table[i, j].set_facecolor("#f0f4ff" if i % 2 == 0 else "white")

    plt.title("Strategy Performance Summary", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
