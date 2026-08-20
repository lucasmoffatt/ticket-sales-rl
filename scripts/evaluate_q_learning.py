"""
Evaluate the trained pricing agents and write charts/metrics.

By default this compares the tabular Q-learning agent against the Keras DQN
agent under identical seeds. If the DQN model has not been trained yet, the
script still runs and evaluates Q-learning alone.

Usage:

    PYTHONPATH=src python scripts/evaluate_q_learning.py
    PYTHONPATH=src python scripts/evaluate_q_learning.py --episodes 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from evaluation.evaluator import GreedyAgent, compare_agents
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
    parser = argparse.ArgumentParser(description="Evaluate and compare pricing agents")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--inventory", type=int, default=100)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--demand-level", type=float, default=1.0)
    parser.add_argument("--q-model", type=str, default="models/q_learning_agent.json")
    parser.add_argument("--dqn-model", type=str, default="models/dqn_agent")
    parser.add_argument("--output-dir", type=str, default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "initial_inventory": args.inventory,
        "selling_days": args.days,
        "demand_level": args.demand_level,
    }

    agents: List[GreedyAgent] = []

    if not Path(args.q_model).exists():
        raise FileNotFoundError(
            f"Missing Q-learning model at {args.q_model}. "
            "Train it with scripts/train_q_learning.py"
        )
    q_agent = QLearningAgent.load(args.q_model)
    agents.append(q_agent)
    print(f"Loaded Q-learning agent from {args.q_model}")

    dqn_agent = None
    if Path(f"{args.dqn_model}.keras").exists():
        dqn_agent = DQNAgent.load(args.dqn_model)
        agents.append(dqn_agent)
        print(f"Loaded DQN agent from {args.dqn_model}.keras")
    else:
        print(
            f"No DQN model at {args.dqn_model}.keras; evaluating Q-learning only. "
            "Train the DQN with scripts/train_dqn.py to compare."
        )

    print(f"Evaluating {len(agents)} agent(s) over {args.episodes} episodes each...")
    results, traces = compare_agents(
        agents,
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
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    save_figure(plot_revenue_summary(results), str(out / "revenue_summary.html"))
    save_figure(plot_sellthrough_summary(results), str(out / "sellthrough_summary.html"))
    save_figure(plot_revenue_distribution(results), str(out / "revenue_distribution.html"))
    save_figure(plot_policy_heatmap(q_agent), str(out / "policy_heatmap.html"))

    # Per-episode trajectory charts for each agent that produced a trace.
    for name, trace in traces.items():
        slug = name.lower().replace(" ", "_").replace("-", "_")
        save_figure(plot_price_over_time(trace), str(out / f"price_over_time_{slug}.html"))
        save_figure(
            plot_inventory_over_time(trace),
            str(out / f"inventory_over_time_{slug}.html"),
        )
        save_figure(
            plot_cumulative_revenue(trace),
            str(out / f"cumulative_revenue_{slug}.html"),
        )

    if q_agent.training_rewards:
        save_figure(
            plot_training_rewards(q_agent.training_rewards, title="Q-Learning training reward"),
            str(out / "q_learning_training.html"),
        )
    if dqn_agent is not None and dqn_agent.training_rewards:
        save_figure(
            plot_training_rewards(dqn_agent.training_rewards, title="DQN training reward"),
            str(out / "dqn_training.html"),
        )

    print(f"\nWrote tables and charts to {out}/")


if __name__ == "__main__":
    main()
