"""
Simulated customer demand for the ticket-pricing environment.

Why a separate module?
----------------------
Keeping demand logic out of the Gymnasium environment makes the model easy to
read, unit-test, and swap later (e.g. different elasticity curves) without
touching the RL loop.

Why Poisson?
------------
Daily ticket requests are non-negative integer counts. A Poisson distribution is
a simple, standard model for count data: you choose an expected demand λ
(lambda), then sample an integer around that mean. Alternatives include
Negative Binomial (extra variance) or deterministic demand (no randomness).

Important: this is a *simulation assumption*, not real market data. The goal is
an understandable training environment for learning reinforcement learning.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.random import Generator


def price_factor(
    price: float,
    reference_price: float = 100.0,
    price_elasticity: float = 1.2,
) -> float:
    """
    How strongly price reduces expected demand.

    We use: (reference_price / price) ** price_elasticity

    - At the reference price (default $100), the factor is 1.0.
    - Above the reference price, demand falls.
    - Below the reference price, demand rises.
    - Higher elasticity means customers are more sensitive to price.

    Alternative approaches: linear demand curves, logit choice models, or
    piecewise tables calibrated from sales data.
    """
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    if reference_price <= 0:
        raise ValueError(f"reference_price must be positive, got {reference_price}")

    return float((reference_price / price) ** price_elasticity)


def urgency_factor(
    days_remaining: int,
    selling_days: int,
    urgency_strength: float = 0.5,
) -> float:
    """
    Mild demand boost as the event approaches.

    Formula: 1 + urgency_strength * (1 - days_remaining / selling_days)

    - Early in the selling window (days_remaining ≈ selling_days): factor ≈ 1.
    - Near the event (days_remaining → 0): factor approaches 1 + urgency_strength.
    - Set urgency_strength=0 to disable this effect.

    Why include this? Real ticket markets often see last-minute interest. Keeping
    it configurable lets us experiment without baking urgency into every run.
    """
    if selling_days <= 0:
        raise ValueError(f"selling_days must be positive, got {selling_days}")
    if days_remaining < 0:
        raise ValueError(f"days_remaining must be non-negative, got {days_remaining}")

    # Clamp so we never divide in a way that produces a negative boost schedule.
    fraction_elapsed = 1.0 - (days_remaining / selling_days)
    fraction_elapsed = float(np.clip(fraction_elapsed, 0.0, 1.0))
    return 1.0 + urgency_strength * fraction_elapsed


def expected_demand(
    price: float,
    days_remaining: int,
    selling_days: int,
    *,
    base_demand: float = 8.0,
    demand_level: float = 1.0,
    reference_price: float = 100.0,
    price_elasticity: float = 1.2,
    urgency_strength: float = 0.5,
) -> float:
    """
    Compute λ (lambda), the Poisson mean for today's demand.

    λ = base_demand × demand_level × price_factor(price) × urgency_factor(days)

    - base_demand: typical daily requests at the reference price with no urgency.
    - demand_level: global intensity knob (e.g. 0.5 = quiet market, 2.0 = hot).
    """
    lam = (
        base_demand
        * demand_level
        * price_factor(price, reference_price, price_elasticity)
        * urgency_factor(days_remaining, selling_days, urgency_strength)
    )
    # Poisson mean must be non-negative.
    return float(max(lam, 0.0))


def simulate_demand(
    price: float,
    days_remaining: int,
    selling_days: int,
    *,
    base_demand: float = 8.0,
    demand_level: float = 1.0,
    reference_price: float = 100.0,
    price_elasticity: float = 1.2,
    urgency_strength: float = 0.5,
    rng: Optional[Generator] = None,
) -> int:
    """
    Sample customer demand for one day as a non-negative integer.

    Returns requested demand only. The environment is responsible for capping
    sales by remaining inventory so this function stays pure and easy to test.
    """
    if rng is None:
        rng = np.random.default_rng()

    lam = expected_demand(
        price=price,
        days_remaining=days_remaining,
        selling_days=selling_days,
        base_demand=base_demand,
        demand_level=demand_level,
        reference_price=reference_price,
        price_elasticity=price_elasticity,
        urgency_strength=urgency_strength,
    )
    return int(rng.poisson(lam))
