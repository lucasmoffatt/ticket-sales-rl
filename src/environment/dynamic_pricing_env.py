"""
Gymnasium environment for dynamic ticket pricing.

What is a Gymnasium environment?
--------------------------------
Reinforcement learning needs a standard interface between an *agent* and the
*world* it interacts with. Gymnasium defines that interface:

- reset(): start a new episode, return the first observation
- step(action): apply one action, return (obs, reward, terminated, truncated, info)

Why build a custom env?
-----------------------
Ticket pricing is not a built-in Gymnasium game. A custom env lets us define
exactly the business rules we care about: fixed inventory, limited selling days,
discrete prices, and stochastic demand.

Episode termination vs truncation
---------------------------------
- terminated: the task naturally ended (sold out, or event day reached).
- truncated: we stopped early for some external reason (time limit wrapper, etc.).
In this env we only use natural termination; truncated is always False.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.random import Generator

from simulation.demand import simulate_demand


DEFAULT_PRICE_LEVELS: List[int] = [50, 75, 100, 125, 150, 175, 200]


class DynamicPricingEnv(gym.Env):
    """
    Sell a fixed inventory of tickets over a limited number of days.

    State (observation) — no look-ahead / future information:
        1. normalized tickets remaining
        2. normalized days remaining
        3. previous ticket price (normalized)
        4. previous day's sales (normalized by initial inventory)

    Action:
        Discrete index into ``price_levels``.

    Reward (version 1):
        daily revenue = price × tickets_sold

    Optional reward shaping:
        When the selling window ends with unsold tickets, subtract
        ``terminal_inventory_penalty × unsold``. Default penalty is 0.0 so the
        base problem stays simple. Reward shaping can help learning later, but
        can also distort the true business objective if overused.
    """

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
        """
        Configure the pricing problem.

        Parameters are stored on the instance so nothing important is hard-coded
        inside ``step`` / ``reset``. That makes experiments and the future
        Streamlit dashboard much easier.
        """
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

        # Action space: choose one of the allowed ticket prices by index.
        # Discrete actions keep Stage 1–3 simple (tabular Q-learning needs this).
        # Continuous prices are an alternative, but need different algorithms.
        self.action_space = spaces.Discrete(len(self.price_levels))

        # Observation space: four normalized features in [0, 1].
        # Tabular Q-learning discretizes the first two features into bins;
        # keeping a Box observation makes that conversion straightforward.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Internal episode state (set fully in reset).
        self.tickets_remaining: int = self.initial_inventory
        self.days_remaining: int = self.selling_days
        self.previous_price: float = 0.0
        self.previous_sales: int = 0
        self._np_random: Generator

        # Seed at construction for convenience; reset(seed=...) can re-seed.
        self.reset(seed=seed)

    def _get_obs(self) -> np.ndarray:
        """Build the normalized observation vector (no future information)."""
        obs = np.array(
            [
                self.tickets_remaining / self.initial_inventory,
                self.days_remaining / self.selling_days,
                self.previous_price / self.max_price,
                self.previous_sales / self.initial_inventory,
            ],
            dtype=np.float32,
        )
        # Numerical safety: keep values inside the declared observation space.
        return np.clip(obs, 0.0, 1.0)

    def _get_info(self, **extra: Any) -> Dict[str, Any]:
        """Diagnostic info dict (not used for learning decisions by the agent)."""
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
        """
        Start a new selling season (episode).

        Gymnasium contract:
            return (observation, info)

        Why reset exists:
            RL agents learn over many episodes. Each episode must start from a
            well-defined initial state so comparisons and training curves are
            meaningful. Seeding here makes randomness reproducible.
        """
        # gym.Env.reset handles seeding of self.np_random when seed is provided.
        super().reset(seed=seed)
        self._np_random = self.np_random

        self.tickets_remaining = self.initial_inventory
        self.days_remaining = self.selling_days
        # No previous market activity at the start of an episode.
        self.previous_price = 0.0
        self.previous_sales = 0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advance the market by one day given a pricing action.

        Gymnasium contract:
            return (observation, reward, terminated, truncated, info)

        Sequence each day:
            1. Agent chooses a price (action index)
            2. Demand is simulated
            3. Sales are limited by remaining inventory
            4. Revenue / reward is computed
            5. Clock moves forward one day
            6. Episode ends if sold out or the event date is reached
        """
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

        # Never sell more tickets than remain.
        tickets_sold = min(int(demand), self.tickets_remaining)
        revenue = price * tickets_sold

        self.tickets_remaining -= tickets_sold
        self.days_remaining -= 1

        self.previous_price = price
        self.previous_sales = tickets_sold

        # Version 1 reward: daily revenue.
        # Alternative designs: only reward at episode end (sparse), or penalize
        # unsold inventory (reward shaping). We keep daily revenue as the default
        # because it is intuitive and gives the agent a learning signal every day.
        reward = float(revenue)

        sold_out = self.tickets_remaining == 0
        time_up = self.days_remaining == 0
        terminated = bool(sold_out or time_up)
        truncated = False

        # Optional terminal penalty for leftover inventory when time runs out.
        # Applied only on natural end-of-horizon (not when we sell out early),
        # so successful sell-outs are not punished.
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
