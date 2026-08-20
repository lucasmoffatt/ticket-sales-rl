"""
Deep Q-Network (DQN) agent for dynamic ticket pricing (TensorFlow / Keras).

Why a neural network after tabular Q-learning?
----------------------------------------------
The tabular agent (``q_learning_agent.py``) has to *discretize* the market into
9 coarse buckets (3 inventory bins x 3 time bins) so a lookup table can store a
value for every situation. That is easy to inspect but throws away detail: two
very different days can land in the same bucket.

A DQN replaces the table with a small neural network that reads the *full*
continuous observation the environment already produces:

    [ normalized tickets remaining,
      normalized days remaining,
      normalized previous price,
      normalized previous sales ]

The network learns a function Q(state) -> value for each price, so it can react
to fine-grained differences the table cannot see.

Three ideas that make DQN stable (interview cheat-sheet)
-------------------------------------------------------
- Replay buffer: we store past transitions and train on random mini-batches.
  This breaks the strong correlation between consecutive days, which would
  otherwise make gradient descent unstable.
- Target network: a slowly-updated copy of the network provides the learning
  target r + gamma * max_a' Q_target(s', a'). Bootstrapping off a *moving*
  target is unstable, so we freeze it and refresh it every so often.
- Epsilon-greedy: same exploration/exploitation trade-off as tabular Q-learning.

The learning rule is the same Bellman/temporal-difference target as the tabular
agent; only the value store (network vs table) changes.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random import Generator

import tensorflow as tf
from tensorflow import keras

from environment.dynamic_pricing_env import DEFAULT_PRICE_LEVELS


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    """Fixed-size memory of past transitions, sampled uniformly at random."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = int(capacity)
        self._memory: Deque[Transition] = deque(maxlen=self.capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._memory.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(
        self, batch_size: int, rng: Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a random mini-batch as stacked NumPy arrays."""
        indices = rng.integers(0, len(self._memory), size=batch_size)
        batch = [self._memory[int(i)] for i in indices]
        states = np.stack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.stack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._memory)


def build_q_network(
    obs_dim: int, n_actions: int, hidden: Sequence[int] = (64, 64)
) -> keras.Model:
    """A small feed-forward network mapping an observation to one Q-value per action."""
    layers = [keras.layers.Input(shape=(obs_dim,))]
    for units in hidden:
        layers.append(keras.layers.Dense(units, activation="relu"))
    layers.append(keras.layers.Dense(n_actions, activation="linear"))
    return keras.Sequential(layers)


class DQNAgent:
    """
    Deep Q-Network agent with a replay buffer and a target network.

    The public interface (``name``, ``select_action``) matches ``QLearningAgent``
    so the same evaluation loop and Streamlit app work with either agent.
    """

    name = "DQN"

    def __init__(
        self,
        obs_dim: int = 4,
        n_actions: int = 7,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 200,
        reward_scale: float = 1e-3,
        hidden: Sequence[int] = (64, 64),
        price_levels: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        # Revenue is measured in thousands of dollars, which makes raw Q-targets
        # large and training unstable. Scaling the reward keeps targets small.
        # Action selection uses argmax, so a constant scale does not change the
        # learned policy — only its numerical stability.
        self.reward_scale = float(reward_scale)
        self.hidden = tuple(int(h) for h in hidden)
        self.price_levels: List[float] = list(
            price_levels if price_levels is not None else DEFAULT_PRICE_LEVELS
        )
        if len(self.price_levels) != self.n_actions:
            raise ValueError("n_actions must match len(price_levels)")

        if seed is not None:
            tf.random.set_seed(seed)
        self._rng: Generator = np.random.default_rng(seed)

        self.buffer = ReplayBuffer(buffer_size)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = keras.losses.MeanSquaredError()

        self.model = build_q_network(self.obs_dim, self.n_actions, self.hidden)
        self.target_model = build_q_network(self.obs_dim, self.n_actions, self.hidden)
        self._update_target()

        self.training_rewards: List[float] = []

    def _update_target(self) -> None:
        """Copy the online network's weights into the frozen target network."""
        self.target_model.set_weights(self.model.get_weights())

    def select_action(
        self,
        observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        *,
        explore: bool = True,
    ) -> int:
        """Epsilon-greedy over the network's predicted Q-values."""
        del info
        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        q_values = self.model(obs, training=False).numpy()[0]
        return int(np.argmax(q_values))

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.add(state, action, reward, next_state, done)

    def learn_from_buffer(self) -> Optional[float]:
        """Run one gradient step on a random mini-batch. Returns the loss."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size, self._rng
        )

        # TD target from the frozen target network.
        next_q = self.target_model(next_states, training=False).numpy()
        max_next_q = np.max(next_q, axis=1)
        targets = self.reward_scale * rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self._train_step(
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(actions, dtype=tf.int32),
            tf.convert_to_tensor(targets, dtype=tf.float32),
        )
        return float(loss)

    def _train_step(
        self, states: tf.Tensor, actions: tf.Tensor, targets: tf.Tensor
    ) -> tf.Tensor:
        """Gradient descent on the Q-value of the actions actually taken."""
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            action_masks = tf.one_hot(actions, self.n_actions)
            q_taken = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = self.loss_fn(targets, q_taken)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def decay_epsilon(self) -> None:
        """Shrink exploration over training so the agent exploits more later."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(
        self,
        env: Any,
        n_episodes: int = 500,
        seed: Optional[int] = None,
        warmup_steps: int = 200,
    ) -> List[float]:
        """
        Train the DQN and return total (unscaled) reward per episode.

        We fill the replay buffer for ``warmup_steps`` before learning so the
        first gradient updates see a variety of transitions, not just the very
        first few correlated days.
        """
        self.training_rewards = []
        step_count = 0

        for episode in range(n_episodes):
            episode_seed = None if seed is None else seed + episode
            observation, _ = env.reset(seed=episode_seed)
            terminated = False
            total_reward = 0.0

            while not terminated:
                action = self.select_action(observation, explore=True)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                self.remember(observation, action, float(reward), next_observation, done)

                observation = next_observation
                total_reward += float(reward)
                step_count += 1

                if step_count >= warmup_steps:
                    self.learn_from_buffer()
                if step_count % self.target_update_freq == 0:
                    self._update_target()

                terminated = done

            self.decay_epsilon()
            self.training_rewards.append(total_reward)

        return self.training_rewards

    def _hyperparameters(self) -> Dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "reward_scale": self.reward_scale,
            "hidden": list(self.hidden),
            "price_levels": self.price_levels,
        }

    @staticmethod
    def _base_path(path: Union[str, Path]) -> Path:
        """Normalize a path to a base with no extension (we add .keras / .json)."""
        p = Path(path)
        if p.suffix in {".keras", ".json"}:
            p = p.with_suffix("")
        return p

    def save(self, path: Union[str, Path]) -> None:
        """Save the network (``.keras``) and hyperparameters (``.json``)."""
        base = self._base_path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(f"{base}.keras")
        payload = self._hyperparameters()
        payload["training_rewards"] = self.training_rewards
        Path(f"{base}.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DQNAgent":
        base = cls._base_path(path)
        payload = json.loads(Path(f"{base}.json").read_text(encoding="utf-8"))
        training_rewards = payload.pop("training_rewards", [])
        agent = cls(**payload)
        agent.model = keras.models.load_model(f"{base}.keras")
        agent._update_target()
        agent.training_rewards = list(training_rewards)
        # A loaded agent is for evaluation: exploit, don't explore.
        agent.epsilon = agent.epsilon_min
        return agent
