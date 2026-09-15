"""Gymnasium environment for a finite-horizon ticket sale."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.random import Generator

from simulation.demand import simulate_demand


DEFAULT_PRICE_LEVELS: List[int] = [50, 75, 100, 125, 150, 175, 200]


class DynamicPricingEnv(gym.Env):
    """Sell a fixed ticket inventory over a limited number of days."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        initial_inventory: int = 100,
        selling_days: int = 20,
        price_levels: Optional[List[int]] = None,
        demand_level: float = 1.0,
        terminal_inventory_penalty: float = 0.0,
        base_demand: float = 8.0,
        reference_price: float = 100.0,
        price_elasticity: float = 1.2,
        urgency_strength: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        """Configure the inventory, demand model, and available prices."""
        super().__init__()

        if initial_inventory <= 0:
            raise ValueError("initial_inventory must be positive")
        if selling_days <= 0:
            raise ValueError("selling_days must be positive")
        if demand_level < 0:
            raise ValueError("demand_level must be non-negative")
        if terminal_inventory_penalty < 0:
            raise ValueError("terminal_inventory_penalty must be non-negative")

        self.initial_inventory = int(initial_inventory)
        self.selling_days = int(selling_days)
        self.price_levels = list(price_levels) if price_levels is not None else list(
            DEFAULT_PRICE_LEVELS
        )
        if len(self.price_levels) == 0:
            raise ValueError("price_levels must be non-empty")
        if any(p <= 0 for p in self.price_levels):
            raise ValueError("all price_levels must be positive")

        self.demand_level = float(demand_level)
        self.terminal_inventory_penalty = float(terminal_inventory_penalty)
        self.base_demand = float(base_demand)
        self.reference_price = float(reference_price)
        self.price_elasticity = float(price_elasticity)
        self.urgency_strength = float(urgency_strength)

        self.max_price = float(max(self.price_levels))

        # Actions are indices into price_levels.
        self.action_space = spaces.Discrete(len(self.price_levels))

        # Observations contain inventory, time, previous price, and previous sales.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.tickets_remaining: int = self.initial_inventory
        self.days_remaining: int = self.selling_days
        self.previous_price: float = 0.0
        self.previous_sales: int = 0
        self._np_random: Generator

        self.reset(seed=seed)

    def _get_obs(self) -> np.ndarray:
        """Return the current state normalized to the observation bounds."""
        obs = np.array(
            [
                self.tickets_remaining / self.initial_inventory,
                self.days_remaining / self.selling_days,
                self.previous_price / self.max_price,
                self.previous_sales / self.initial_inventory,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)

    def _get_info(self, **extra: Any) -> Dict[str, Any]:
        """Build the diagnostic data returned by ``reset`` and ``step``."""
        info: Dict[str, Any] = {
            "tickets_remaining": self.tickets_remaining,
            "days_remaining": self.days_remaining,
            "previous_price": self.previous_price,
            "previous_sales": self.previous_sales,
        }
        info.update(extra)
        return info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the selling period and return its initial state."""
        super().reset(seed=seed)
        self._np_random = self.np_random

        self.tickets_remaining = self.initial_inventory
        self.days_remaining = self.selling_days
        self.previous_price = 0.0
        self.previous_sales = 0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply a price for one day and return the resulting transition."""
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}; expected integer in "
                f"[0, {self.action_space.n})"
            )

        price = float(self.price_levels[int(action)])

        demand = simulate_demand(
            price=price,
            days_remaining=self.days_remaining,
            selling_days=self.selling_days,
            base_demand=self.base_demand,
            demand_level=self.demand_level,
            reference_price=self.reference_price,
            price_elasticity=self.price_elasticity,
            urgency_strength=self.urgency_strength,
            rng=self._np_random,
        )

        tickets_sold = min(int(demand), self.tickets_remaining)
        revenue = price * tickets_sold

        self.tickets_remaining -= tickets_sold
        self.days_remaining -= 1

        self.previous_price = price
        self.previous_sales = tickets_sold

        reward = float(revenue)

        sold_out = self.tickets_remaining == 0
        time_up = self.days_remaining == 0
        terminated = bool(sold_out or time_up)
        truncated = False

        # Charge the inventory penalty only when the selling window expires.
        penalty = 0.0
        if time_up and self.tickets_remaining > 0 and self.terminal_inventory_penalty > 0:
            penalty = self.terminal_inventory_penalty * self.tickets_remaining
            reward -= penalty

        observation = self._get_obs()
        info = self._get_info(
            price=price,
            demand=int(demand),
            tickets_sold=tickets_sold,
            revenue=float(revenue),
            penalty=float(penalty),
            reward=float(reward),
        )
        return observation, reward, terminated, truncated, info
