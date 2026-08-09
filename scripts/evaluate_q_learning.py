"""
Evaluate the trained tabular Q-learning agent and write charts/metrics.

Usage:

    PYTHONPATH=src python scripts/evaluate_q_learning.py
    PYTHONPATH=src python scripts/evaluate_q_learning.py --episodes 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agents.q_learning_agent import QLearningAgent
from environment.dynamic_pricing_env import DynamicPricingEnv
from evaluation.evaluator import evaluate_q_learning
from evaluation.metrics import results_to_frame, summarize_results
from visualization.plots import (
    plot_cumulative_revenue,
    plot_inventory_over_time,
    plot_policy_heatmap,
    plot_price_over_time,
    plot_revenue_distribution,
    plot_revenue_summary,
    plot_sellthrough_summary,
    plot_training_rewards,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Q-learning pricing agent")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--inventory", type=int, default=100)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--demand-level", type=float, default=1.0)
    parser.add_argument("--q-model", type=str, default="models/q_learning_agent.json")
    parser.add_argument("--train-if-missing", action="store_true", default=True)
    parser.add_argument(
        "--no-train-if-missing", action="store_false", dest="train_if_missing"
    )
    parser.add_argument("--q-episodes", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="data")
    return parser.parse_args()


def ensure_q_agent(path: str, env_kwargs: dict, episodes: int, seed: int) -> QLearningAgent:
    model_path = Path(path)
    if model_path.exists():
        print(f"Loading Q-learning agent from {path}")
        return QLearningAgent.load(path)

    print(f"No Q-learning model at {path}; training for {episodes} episodes...")
    env = DynamicPricingEnv(**env_kwargs, seed=seed)
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        price_levels=env.price_levels,
        seed=seed,
    )
    agent.train(env, n_episodes=episodes, seed=seed)
    agent.save(path)
    return agent


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "initial_inventory": args.inventory,
        "selling_days": args.days,
        "demand_level": args.demand_level,
    }

    if not args.train_if_missing and not Path(args.q_model).exists():
        raise FileNotFoundError(
            f"Missing model at {args.q_model}. Train with scripts/train_q_learning.py"
        )

    agent = ensure_q_agent(args.q_model, env_kwargs, args.q_episodes, args.seed)

    print(f"Evaluating Q-learning over {args.episodes} episodes...")
    results, trace = evaluate_q_learning(
        agent,
        n_episodes=args.episodes,
        env_kwargs=env_kwargs,
        base_seed=args.seed,
        record_first_trace=True,
    )

    summary = summarize_results(results)
    episode_df = results_to_frame(results)
    summary.to_csv(out / "evaluation_summary.csv", index=False)
    episode_df.to_csv(out / "evaluation_episodes.csv", index=False)

    print("\n=== Summary ===")
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    save_figure(plot_revenue_summary(results), str(out / "revenue_summary.html"))
    save_figure(plot_sellthrough_summary(results), str(out / "sellthrough_summary.html"))
    save_figure(plot_revenue_distribution(results), str(out / "revenue_distribution.html"))
    save_figure(plot_policy_heatmap(agent), str(out / "policy_heatmap.html"))

    if trace is not None:
        save_figure(plot_price_over_time(trace), str(out / "price_over_time.html"))
        save_figure(plot_inventory_over_time(trace), str(out / "inventory_over_time.html"))
        save_figure(
            plot_cumulative_revenue(trace), str(out / "cumulative_revenue.html")
        )

    if agent.training_rewards:
        save_figure(
            plot_training_rewards(agent.training_rewards),
            str(out / "q_learning_training.html"),
        )

    print(f"\nWrote tables and charts to {out}/")


if __name__ == "__main__":
    main()
