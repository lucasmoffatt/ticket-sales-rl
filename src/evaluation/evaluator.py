"""
Evaluation helpers for the tabular Q-learning pricing agent.

Fairness notes
--------------
- Fixed environment parameters across evaluation episodes.
- Deterministic (greedy) action selection — no exploration.
- Episode k uses seed = base_seed + k for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from agents.q_learning_agent import QLearningAgent
from environment.dynamic_pricing_env import DynamicPricingEnv
from evaluation.metrics import EpisodeResult, compute_episode_result, summarize_results


ActionFn = Callable[[np.ndarray, Dict[str, Any]], int]


@dataclass
class EpisodeTrace:
    """Step-by-step trajectory for plotting a representative episode."""

    strategy: str
    prices: List[float] = field(default_factory=list)
    inventory: List[int] = field(default_factory=list)
    cumulative_revenue: List[float] = field(default_factory=list)
    days_remaining: List[int] = field(default_factory=list)
    tickets_sold: List[int] = field(default_factory=list)
    revenues: List[float] = field(default_factory=list)


def run_episode(
    env: DynamicPricingEnv,
    action_fn: ActionFn,
    *,
    strategy_name: str,
    episode_index: int,
    seed: Optional[int] = None,
    record_trace: bool = False,
) -> tuple[EpisodeResult, Optional[EpisodeTrace]]:
    """Simulate one episode and optionally keep a plotting trace."""
    observation, info = env.reset(seed=seed)
    revenues: List[float] = []
    tickets_sold_list: List[int] = []
    prices: List[float] = []
    terminated = False

    trace: Optional[EpisodeTrace] = None
    if record_trace:
        trace = EpisodeTrace(strategy=strategy_name)
        trace.inventory.append(int(info["tickets_remaining"]))
        trace.days_remaining.append(int(info["days_remaining"]))
        trace.cumulative_revenue.append(0.0)

    cumulative = 0.0
    while not terminated:
        action = action_fn(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        del truncated, reward
        revenues.append(float(info["revenue"]))
        tickets_sold_list.append(int(info["tickets_sold"]))
        prices.append(float(info["price"]))
        cumulative += float(info["revenue"])

        if trace is not None:
            trace.prices.append(float(info["price"]))
            trace.inventory.append(int(info["tickets_remaining"]))
            trace.days_remaining.append(int(info["days_remaining"]))
            trace.cumulative_revenue.append(cumulative)
            trace.tickets_sold.append(int(info["tickets_sold"]))
            trace.revenues.append(float(info["revenue"]))

    result = compute_episode_result(
        strategy=strategy_name,
        episode=episode_index,
        revenues=revenues,
        tickets_sold_per_step=tickets_sold_list,
        prices=prices,
        initial_inventory=env.initial_inventory,
        final_inventory=int(info["tickets_remaining"]),
    )
    return result, trace


def evaluate_q_learning(
    agent: QLearningAgent,
    *,
    n_episodes: int = 100,
    env_kwargs: Optional[Dict[str, Any]] = None,
    base_seed: int = 123,
    record_first_trace: bool = True,
) -> tuple[List[EpisodeResult], Optional[EpisodeTrace]]:
    """Evaluate a greedy Q-learning policy over many episodes."""
    kwargs = dict(env_kwargs or {})
    env = DynamicPricingEnv(**kwargs)

    def action_fn(obs: np.ndarray, info: Dict[str, Any]) -> int:
        return agent.select_action(obs, info, explore=False)

    results: List[EpisodeResult] = []
    first_trace: Optional[EpisodeTrace] = None

    for ep in range(n_episodes):
        seed = base_seed + ep
        result, trace = run_episode(
            env,
            action_fn,
            strategy_name=agent.name,
            episode_index=ep,
            seed=seed,
            record_trace=record_first_trace and ep == 0,
        )
        results.append(result)
        if trace is not None:
            first_trace = trace

    return results, first_trace


def summary_table(results: List[EpisodeResult]):
    """Convenience wrapper used by scripts and the Streamlit app."""
    return summarize_results(results)
