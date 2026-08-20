"""Tests for the Keras Deep Q-Network agent."""

from __future__ import annotations

import numpy as np

from agents.dqn_agent import DQNAgent, ReplayBuffer, build_q_network
from environment.dynamic_pricing_env import DynamicPricingEnv


def test_build_q_network_output_shape() -> None:
    model = build_q_network(obs_dim=4, n_actions=7)
    batch = np.zeros((5, 4), dtype=np.float32)
    output = model(batch, training=False).numpy()
    assert output.shape == (5, 7)


def test_replay_buffer_add_and_sample() -> None:
    buffer = ReplayBuffer(capacity=100)
    rng = np.random.default_rng(0)
    for i in range(10):
        state = np.full(4, i, dtype=np.float32)
        buffer.add(state, action=i % 7, reward=float(i), next_state=state, done=False)
    assert len(buffer) == 10

    states, actions, rewards, next_states, dones = buffer.sample(4, rng)
    assert states.shape == (4, 4)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 4)
    assert dones.shape == (4,)


def test_select_action_returns_valid_index() -> None:
    agent = DQNAgent(obs_dim=4, n_actions=7, seed=0)
    obs = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    action = agent.select_action(obs, explore=False)
    assert isinstance(action, int)
    assert 0 <= action < 7


def test_dqn_trains_and_saves(tmp_path) -> None:
    env = DynamicPricingEnv(initial_inventory=30, selling_days=6, seed=1)
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        price_levels=env.price_levels,
        batch_size=8,
        seed=1,
    )
    rewards = agent.train(env, n_episodes=3, seed=1, warmup_steps=4)
    assert len(rewards) == 3

    base = tmp_path / "dqn_agent"
    agent.save(base)
    assert (tmp_path / "dqn_agent.keras").exists()
    assert (tmp_path / "dqn_agent.json").exists()

    loaded = DQNAgent.load(base)
    obs = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    # Same weights -> same greedy action.
    assert loaded.select_action(obs, explore=False) == agent.select_action(
        obs, explore=False
    )
