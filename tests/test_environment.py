"""Unit tests for DynamicPricingEnv."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment.dynamic_pricing_env import DynamicPricingEnv


def test_gymnasium_env_checker_passes() -> None:
    env = DynamicPricingEnv(seed=0)
    check_env(env, skip_render_check=True)


def test_inventory_never_negative() -> None:
    env = DynamicPricingEnv(initial_inventory=10, selling_days=30, seed=1)
    obs, _ = env.reset(seed=1)
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.tickets_remaining >= 0
        assert info["tickets_remaining"] >= 0
        assert truncated is False


def test_tickets_sold_cannot_exceed_remaining_inventory() -> None:
    env = DynamicPricingEnv(
        initial_inventory=5,
        selling_days=50,
        demand_level=10.0,  # intentionally high demand
        seed=2,
    )
    env.reset(seed=2)
    remaining_before = env.tickets_remaining
    _, _, _, _, info = env.step(0)  # cheapest price → high demand
    assert info["tickets_sold"] <= remaining_before
    assert info["tickets_sold"] <= info["demand"]


def test_episode_ends_at_zero_inventory() -> None:
    env = DynamicPricingEnv(
        initial_inventory=3,
        selling_days=100,
        demand_level=20.0,
        seed=3,
    )
    env.reset(seed=3)
    terminated = False
    steps = 0
    while not terminated and steps < 100:
        _, _, terminated, _, info = env.step(0)
        steps += 1
    assert terminated
    assert info["tickets_remaining"] == 0


def test_episode_ends_when_days_reach_zero() -> None:
    env = DynamicPricingEnv(
        initial_inventory=10_000,
        selling_days=5,
        demand_level=0.01,  # almost no demand → time should expire first
        seed=4,
    )
    env.reset(seed=4)
    terminated = False
    for _ in range(5):
        _, _, terminated, _, info = env.step(6)  # expensive price
    assert terminated
    assert info["days_remaining"] == 0
    assert info["tickets_remaining"] > 0


def test_reward_equals_daily_revenue_without_penalty() -> None:
    env = DynamicPricingEnv(
        initial_inventory=100,
        selling_days=10,
        terminal_inventory_penalty=0.0,
        seed=5,
    )
    env.reset(seed=5)
    action = 2  # $100
    _, reward, _, _, info = env.step(action)
    assert reward == pytest.approx(info["price"] * info["tickets_sold"])
    assert info["penalty"] == 0.0


def test_terminal_inventory_penalty_applied_when_time_expires() -> None:
    env = DynamicPricingEnv(
        initial_inventory=50,
        selling_days=2,
        demand_level=0.0,  # zero demand → unsold inventory remains
        terminal_inventory_penalty=3.0,
        seed=6,
    )
    env.reset(seed=6)
    env.step(2)
    _, reward, terminated, _, info = env.step(2)

    assert terminated
    assert info["days_remaining"] == 0
    assert info["tickets_remaining"] == 50
    # No sales, so reward is only the penalty.
    assert info["revenue"] == 0.0
    assert info["penalty"] == pytest.approx(3.0 * 50)
    assert reward == pytest.approx(-150.0)


def test_observations_remain_inside_observation_space() -> None:
    env = DynamicPricingEnv(seed=7)
    obs, _ = env.reset(seed=7)
    assert env.observation_space.contains(obs)

    terminated = False
    while not terminated:
        obs, _, terminated, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)


def test_seeded_episodes_are_reproducible() -> None:
    def run(seed: int) -> list[float]:
        env = DynamicPricingEnv(seed=seed)
        env.reset(seed=seed)
        rewards = []
        terminated = False
        # Use a fixed action sequence so only env randomness differs.
        action = 2
        while not terminated:
            _, reward, terminated, _, _ = env.step(action)
            rewards.append(reward)
        return rewards

    assert run(42) == run(42)
    # Different seeds should usually differ; if not, at least lengths are valid.
    assert len(run(1)) >= 1


def test_action_maps_to_configured_prices() -> None:
    prices = [50, 100, 200]
    env = DynamicPricingEnv(price_levels=prices, seed=8)
    env.reset(seed=8)
    for action, expected_price in enumerate(prices):
        env.reset(seed=8)
        _, _, _, _, info = env.step(action)
        assert info["price"] == expected_price
