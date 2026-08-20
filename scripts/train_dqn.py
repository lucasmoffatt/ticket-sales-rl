"""
Train the Keras Deep Q-Network agent and save the model + reward history.

Usage (from project root):

    PYTHONPATH=src python scripts/train_dqn.py
    PYTHONPATH=src python scripts/train_dqn.py --episodes 800
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.dqn_agent import DQNAgent
from environment.dynamic_pricing_env import DynamicPricingEnv
from visualization.plots import plot_training_rewards, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Keras DQN pricing agent")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inventory", type=int, default=100)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--demand-level", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=str,
        default="models/dqn_agent",
        help="Base path for the saved model (.keras + .json are appended)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="data/dqn_training.html",
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
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        price_levels=env.price_levels,
        seed=args.seed,
    )

    print(f"Training DQN for {args.episodes} episodes...")
    rewards = agent.train(env, n_episodes=args.episodes, seed=args.seed)
    agent.save(args.output)

    fig = plot_training_rewards(rewards, title="DQN training reward")
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.plot)

    print(f"Saved model → {args.output}.keras / {args.output}.json")
    print(f"Saved plot  → {args.plot}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Last 100 avg reward: {sum(rewards[-100:]) / min(100, len(rewards)):.1f}")


if __name__ == "__main__":
    main()
