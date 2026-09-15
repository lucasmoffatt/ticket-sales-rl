"""Synthetic demand model used by the pricing environment."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.random import Generator


def price_factor(
    price: float,
    reference_price: float = 100.0,
    price_elasticity: float = 1.2,
) -> float:
    """Return the demand multiplier implied by price elasticity."""
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
    """Return the linear demand boost as the event approaches."""
    if selling_days <= 0:
        raise ValueError(f"selling_days must be positive, got {selling_days}")
    if days_remaining < 0:
        raise ValueError(f"days_remaining must be non-negative, got {days_remaining}")

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
    """Calculate the Poisson rate for daily ticket requests."""
    lam = (
        base_demand
        * demand_level
        * price_factor(price, reference_price, price_elasticity)
        * urgency_factor(days_remaining, selling_days, urgency_strength)
    )
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
    """Draw one day's ticket requests from the demand distribution."""
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
