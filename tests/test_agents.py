"""Tests for the tabular Q-learning agent and evaluation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from agents.q_learning_agent import (
    QLearningAgent,
    discretize_inventory,
    discretize_observation,
    discretize_time,
)
from environment.dynamic_pricing_env import DynamicPricingEnv
from evaluation.evaluator import evaluate_q_learning, run_episode
from evaluation.metrics import summarize_results


def test_discretization_bins() -> None:
    assert discretize_inventory(0.1) == "Low"
    assert discretize_inventory(0.5) == "Medium"
    assert discretize_inventory(0.9) == "High"
    assert discretize_time(0.9) == "Early"
    assert discretize_time(0.5) == "Middle"
    assert discretize_time(0.1) == "Late"
    assert discretize_observation(np.array([0.9, 0.2, 0.0, 0.0])) == ("High", "Late")


def test_q_learning_update_moves_q_toward_target() -> None:
    agent = QLearningAgent(n_actions=7, alpha=0.5, gamma=0.0, epsilon=0.0, seed=0)
    state = ("High", "Early")
    before = float(agent.get_q_values(state)[2])
    agent.update(
        state, action=2, reward=100.0, next_state=("High", "Early"), terminated=True
    )
    after = float(agent.get_q_values(state)[2])
    assert after == pytest.approx(50.0)
    assert after > before


def test_q_learning_trains_and_saves(tmp_path) -> None:
    env = DynamicPricingEnv(initial_inventory=40, selling_days=8, seed=1)
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        price_levels=env.price_levels,
        epsilon=0.5,
        seed=1,
    )
    rewards = agent.train(env, n_episodes=30, seed=1)
    assert len(rewards) == 30

    path = tmp_path / "q.json"
    agent.save(path)
    loaded = QLearningAgent.load(path)
    obs = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert loaded.select_action(obs, explore=False) == agent.select_action(
        obs, explore=False
    )


def test_evaluate_q_learning_returns_metrics() -> None:
    env = DynamicPricingEnv(initial_inventory=50, selling_days=10, seed=0)
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        price_levels=env.price_levels,
        epsilon=0.2,
        seed=0,
    )
    agent.train(env, n_episodes=20, seed=0)

    results, trace = evaluate_q_learning(
        agent,
        n_episodes=5,
        env_kwargs={"initial_inventory": 50, "selling_days": 10, "demand_level": 1.0},
        base_seed=0,
        record_first_trace=True,
    )
    assert len(results) == 5
    assert trace is not None
    assert len(trace.prices) >= 1
    summary = summarize_results(results)
    assert "avg_total_revenue" in summary.columns
    assert summary.iloc[0]["strategy"] == "Q-Learning"


def test_run_episode_inventory_conservation() -> None:
    env = DynamicPricingEnv(seed=2)
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        price_levels=env.price_levels,
        epsilon=0.0,
        seed=2,
    )

    def action_fn(obs, info):
        return agent.select_action(obs, info, explore=False)

    result, _ = run_episode(
        env, action_fn, strategy_name="Q-Learning", episode_index=0, seed=2
    )
    assert result.total_revenue >= 0
    assert result.tickets_sold + result.unsold_tickets == env.initial_inventory
