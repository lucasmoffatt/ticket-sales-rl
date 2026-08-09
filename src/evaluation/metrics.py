"""
Episode-level and aggregate metrics for pricing strategies.

We separate metric *definitions* from the evaluation loop so formulas stay easy
to audit in an interview.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class EpisodeResult:
    """Outcome of one simulated selling season."""

    strategy: str
    episode: int
    total_revenue: float
    tickets_sold: int
    initial_inventory: int
    unsold_tickets: int
    average_selling_price: float
    sell_through: float
    sold_out: bool
    steps: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_episode_result(
    *,
    strategy: str,
    episode: int,
    revenues: Sequence[float],
    tickets_sold_per_step: Sequence[int],
    prices: Sequence[float],
    initial_inventory: int,
    final_inventory: int,
) -> EpisodeResult:
    """Build metrics for a single finished episode."""
    tickets_sold = int(sum(tickets_sold_per_step))
    total_revenue = float(sum(revenues))
    unsold = int(final_inventory)
    avg_price = float(total_revenue / tickets_sold) if tickets_sold > 0 else 0.0
    sell_through = tickets_sold / initial_inventory if initial_inventory > 0 else 0.0

    return EpisodeResult(
        strategy=strategy,
        episode=episode,
        total_revenue=total_revenue,
        tickets_sold=tickets_sold,
        initial_inventory=initial_inventory,
        unsold_tickets=unsold,
        average_selling_price=avg_price,
        sell_through=sell_through,
        sold_out=unsold == 0,
        steps=len(revenues),
    )


def summarize_results(results: Sequence[EpisodeResult]) -> pd.DataFrame:
    """
    Aggregate many episodes into recruiter-friendly summary statistics.

    Metrics:
    - average / median / std total revenue
    - average sell-through percentage
    - average selling price
    - average unsold tickets
    - percentage of episodes that sell out
    """
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([r.to_dict() for r in results])
    summary = (
        df.groupby("strategy", sort=False)
        .agg(
            episodes=("episode", "count"),
            avg_total_revenue=("total_revenue", "mean"),
            median_total_revenue=("total_revenue", "median"),
            std_total_revenue=("total_revenue", "std"),
            avg_sell_through=("sell_through", "mean"),
            avg_selling_price=("average_selling_price", "mean"),
            avg_unsold_tickets=("unsold_tickets", "mean"),
            sellout_rate=("sold_out", "mean"),
        )
        .reset_index()
    )

    # Percent-style columns for readability.
    summary["avg_sell_through_pct"] = summary["avg_sell_through"] * 100.0
    summary["sellout_rate_pct"] = summary["sellout_rate"] * 100.0
    return summary


def results_to_frame(results: Sequence[EpisodeResult]) -> pd.DataFrame:
    """Convert episode results to a tidy DataFrame."""
    return pd.DataFrame([r.to_dict() for r in results])


def moving_average(values: Sequence[float], window: int = 50) -> np.ndarray:
    """Smooth a training curve for visualization."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    if window <= 1:
        return arr
    window = min(window, len(arr))
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")
