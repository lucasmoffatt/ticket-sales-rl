"""Train and save the tabular Q-learning agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.q_learning_agent import QLearningAgent
from environment.dynamic_pricing_env import DynamicPricingEnv
from visualization.plots import plot_training_rewards, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular Q-learning agent")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inventory", type=int, default=100)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--demand-level", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=str,
        default="models/q_learning_agent.json",
        help="Where to save the trained agent",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="data/q_learning_training.html",
        help="HTML path for the training curve",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = DynamicPricingEnv(
        initial_inventory=args.inventory,
        selling_days=args.days,
        demand_level=args.demand_level,
        seed=args.seed,
    )
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        price_levels=env.price_levels,
        seed=args.seed,
    )

    print(f"Training Q-learning for {args.episodes} episodes...")
    rewards = agent.train(env, n_episodes=args.episodes, seed=args.seed)
    agent.save(args.output)

    fig = plot_training_rewards(rewards, title="Q-Learning training reward")
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.plot)

    print(f"Saved agent → {args.output}")
    print(f"Saved plot  → {args.plot}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Last 100 avg reward: {sum(rewards[-100:]) / min(100, len(rewards)):.1f}")


if __name__ == "__main__":
    main()
