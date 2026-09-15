"""Demand-model tests."""

from __future__ import annotations

import numpy as np

from simulation.demand import expected_demand, simulate_demand, urgency_factor


def test_higher_price_has_lower_expected_demand() -> None:
    low = expected_demand(price=50, days_remaining=10, selling_days=20)
    mid = expected_demand(price=100, days_remaining=10, selling_days=20)
    high = expected_demand(price=200, days_remaining=10, selling_days=20)

    assert low > mid > high


def test_demand_samples_are_non_negative_integers() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        demand = simulate_demand(
            price=100,
            days_remaining=5,
            selling_days=20,
            rng=rng,
        )
        assert isinstance(demand, int)
        assert demand >= 0


def test_urgency_increases_expected_demand_near_event() -> None:
    early = expected_demand(
        price=100,
        days_remaining=20,
        selling_days=20,
        urgency_strength=0.5,
    )
    late = expected_demand(
        price=100,
        days_remaining=1,
        selling_days=20,
        urgency_strength=0.5,
    )
    assert late > early

    assert urgency_factor(20, 20, urgency_strength=0.5) == 1.0
    assert urgency_factor(0, 20, urgency_strength=0.5) == 1.5


def test_zero_urgency_disables_time_effect() -> None:
    early = expected_demand(
        price=100,
        days_remaining=20,
        selling_days=20,
        urgency_strength=0.0,
    )
    late = expected_demand(
        price=100,
        days_remaining=1,
        selling_days=20,
        urgency_strength=0.0,
    )
    assert early == late


def test_seeded_demand_is_reproducible() -> None:
    a = simulate_demand(
        price=125,
        days_remaining=7,
        selling_days=20,
        rng=np.random.default_rng(123),
    )
    b = simulate_demand(
        price=125,
        days_remaining=7,
        selling_days=20,
        rng=np.random.default_rng(123),
    )
    assert a == b
