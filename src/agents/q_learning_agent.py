"""Tabular Q-learning agent for the ticket-pricing environment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import Generator

from environment.dynamic_pricing_env import DEFAULT_PRICE_LEVELS, DynamicPricingEnv


InventoryBin = str  # "Low" | "Medium" | "High"
TimeBin = str  # "Early" | "Middle" | "Late"
DiscreteState = Tuple[InventoryBin, TimeBin]


def discretize_inventory(tickets_frac: float) -> InventoryBin:
    """Assign normalized inventory to one of three bins."""
    if tickets_frac < 1.0 / 3.0:
        return "Low"
    if tickets_frac < 2.0 / 3.0:
        return "Medium"
    return "High"


def discretize_time(days_frac: float) -> TimeBin:
    """Assign normalized time remaining to a selling-period bin."""
    if days_frac > 2.0 / 3.0:
        return "Early"
    if days_frac > 1.0 / 3.0:
        return "Middle"
    return "Late"


def discretize_observation(observation: np.ndarray) -> DiscreteState:
    """Reduce an observation to the dimensions used by the Q-table."""
    return (
        discretize_inventory(float(observation[0])),
        discretize_time(float(observation[1])),
    )


class QLearningAgent:
    """Tabular Q-learning with an epsilon-greedy policy."""

    name = "Q-Learning"

    INVENTORY_BINS: Tuple[InventoryBin, ...] = ("Low", "Medium", "High")
    TIME_BINS: Tuple[TimeBin, ...] = ("Early", "Middle", "Late")

    def __init__(
        self,
        n_actions: int = 7,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        price_levels: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.price_levels: List[float] = list(
            price_levels if price_levels is not None else DEFAULT_PRICE_LEVELS
        )
        if len(self.price_levels) != self.n_actions:
            raise ValueError("n_actions must match len(price_levels)")

        self._rng: Generator = np.random.default_rng(seed)
        # Each state maps to one Q-value per available price.
        self.q_table: Dict[DiscreteState, np.ndarray] = {}
        self.training_rewards: List[float] = []
        self._ensure_all_states()

    def _ensure_all_states(self) -> None:
        for inv in self.INVENTORY_BINS:
            for time_bin in self.TIME_BINS:
                key = (inv, time_bin)
                if key not in self.q_table:
                    self.q_table[key] = np.zeros(self.n_actions, dtype=np.float64)

    def get_q_values(self, state: DiscreteState) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions, dtype=np.float64)
        return self.q_table[state]

    def select_action(
        self,
        observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        *,
        explore: bool = True,
    ) -> int:
        """Choose an action, optionally allowing epsilon-greedy exploration."""
        state = discretize_observation(observation)
        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        q_values = self.get_q_values(state)
        # Avoid consistently favoring the lowest-index action on ties.
        max_q = float(np.max(q_values))
        best_actions = np.flatnonzero(np.isclose(q_values, max_q))
        return int(self._rng.choice(best_actions))

    def update(
        self,
        state: DiscreteState,
        action: int,
        reward: float,
        next_state: DiscreteState,
        terminated: bool,
    ) -> None:
        """Apply one temporal-difference update."""
        q_values = self.get_q_values(state)
        current_q = float(q_values[action])

        if terminated:
            target = float(reward)
        else:
            next_q = self.get_q_values(next_state)
            target = float(reward) + self.gamma * float(np.max(next_q))

        td_error = target - current_q
        q_values[action] = current_q + self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Reduce the exploration rate without dropping below its floor."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_action_for_bins(self, inventory_bin: str, time_bin: str) -> int:
        """Return the best action for an inventory/time bin pair."""
        q_values = self.get_q_values((inventory_bin, time_bin))
        return int(np.argmax(q_values))

    def preferred_price_for_bins(self, inventory_bin: str, time_bin: str) -> float:
        action = self.greedy_action_for_bins(inventory_bin, time_bin)
        return float(self.price_levels[action])

    def train(
        self,
        env: DynamicPricingEnv,
        n_episodes: int = 3000,
        seed: Optional[int] = None,
    ) -> List[float]:
        """Train for ``n_episodes`` and return the reward from each episode."""
        self.training_rewards = []
        for episode in range(n_episodes):
            episode_seed = None if seed is None else seed + episode
            observation, _ = env.reset(seed=episode_seed)
            state = discretize_observation(observation)
            total_reward = 0.0
            terminated = False

            while not terminated:
                action = self.select_action(observation, explore=True)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_state = discretize_observation(next_observation)
                self.update(state, action, reward, next_state, terminated or truncated)
                total_reward += float(reward)
                observation = next_observation
                state = next_state

            self.decay_epsilon()
            self.training_rewards.append(total_reward)

        return self.training_rewards

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "price_levels": self.price_levels,
            "training_rewards": self.training_rewards,
            "q_table": {
                f"{inv}|{time_bin}": self.q_table[(inv, time_bin)].tolist()
                for inv in self.INVENTORY_BINS
                for time_bin in self.TIME_BINS
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            n_actions=payload["n_actions"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            price_levels=payload["price_levels"],
        )
        agent.training_rewards = list(payload.get("training_rewards", []))
        for key, values in payload["q_table"].items():
            inv, time_bin = key.split("|", 1)
            agent.q_table[(inv, time_bin)] = np.array(values, dtype=np.float64)
        return agent
