"""Streamlit dashboard for running trained pricing agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import streamlit as st

from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from environment.dynamic_pricing_env import DynamicPricingEnv
from evaluation.evaluator import run_episode
from visualization.plots import (
    plot_cumulative_revenue,
    plot_inventory_over_time,
    plot_policy_heatmap,
    plot_price_over_time,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q_MODEL_PATH = PROJECT_ROOT / "models" / "q_learning_agent.json"
DQN_MODEL_PATH = PROJECT_ROOT / "models" / "dqn_agent"

Q_STRATEGY = "Q-Learning (table)"
DQN_STRATEGY = "DQN (neural network)"


st.set_page_config(
    page_title="Dynamic Pricing Agents",
    page_icon=None,
    layout="wide",
)


@st.cache_resource
def load_q_agent() -> Optional[QLearningAgent]:
    if Q_MODEL_PATH.exists():
        return QLearningAgent.load(Q_MODEL_PATH)
    return None


@st.cache_resource
def load_dqn_agent() -> Optional[DQNAgent]:
    if Path(f"{DQN_MODEL_PATH}.keras").exists():
        return DQNAgent.load(DQN_MODEL_PATH)
    return None


def main() -> None:
    st.title("Dynamic Pricing Agents")
    st.caption(
        "Simulate a trained pricing policy over a fixed ticket inventory and "
        "selling window. Compare a from-scratch Q-learning table against a "
        "Keras Deep Q-Network."
    )

    q_agent = load_q_agent()
    dqn_agent = load_dqn_agent()

    available: dict[str, Any] = {}
    if q_agent is not None:
        available[Q_STRATEGY] = q_agent
    if dqn_agent is not None:
        available[DQN_STRATEGY] = dqn_agent

    with st.sidebar:
        st.header("Simulation settings")
        if available:
            strategy = st.selectbox("Pricing strategy", list(available.keys()))
        else:
            strategy = None
        initial_inventory = st.slider("Initial ticket inventory", 20, 300, 100, 10)
        selling_days = st.slider("Selling days", 5, 60, 20, 1)
        demand_level = st.slider("Demand level", 0.2, 2.5, 1.0, 0.1)
        terminal_penalty = st.slider(
            "Terminal unsold inventory penalty", 0.0, 20.0, 0.0, 0.5
        )
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        run_button = st.button("Run simulation", type="primary")

        if q_agent is None:
            st.warning(
                "No Q-learning model found. Train one with:\n"
                "`PYTHONPATH=src python scripts/train_q_learning.py`"
            )
        if dqn_agent is None:
            st.warning(
                "No DQN model found. Train one with:\n"
                "`PYTHONPATH=src python scripts/train_dqn.py`"
            )

    if strategy is None:
        st.error("No trained models found. Train an agent, then reload this page.")
        return

    agent = available[strategy]

    if not run_button:
        st.markdown(
            f"""
            Use the sidebar to configure inventory, selling days, and demand,
            then click **Run simulation**.

            Selected agent: **{strategy}** (greedy policy, no exploration).
            """
        )
        if strategy == Q_STRATEGY:
            st.subheader("Learned pricing policy")
            st.plotly_chart(plot_policy_heatmap(agent), use_container_width=True)
        return

    env = DynamicPricingEnv(
        initial_inventory=initial_inventory,
        selling_days=selling_days,
        demand_level=demand_level,
        terminal_inventory_penalty=terminal_penalty,
        seed=int(seed),
    )

    def action_fn(obs, info):
        return agent.select_action(obs, info, explore=False)

    result, trace = run_episode(
        env,
        action_fn,
        strategy_name=strategy,
        episode_index=0,
        seed=int(seed),
        record_trace=True,
    )
    assert trace is not None

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total revenue", f"${result.total_revenue:,.0f}")
    m2.metric("Tickets sold", f"{result.tickets_sold}")
    m3.metric("Sell-through", f"{result.sell_through:.1%}")
    m4.metric("Avg ticket price", f"${result.average_selling_price:,.2f}")
    m5.metric("Unsold inventory", f"{result.unsold_tickets}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(plot_price_over_time(trace), use_container_width=True)
    with c2:
        st.plotly_chart(plot_inventory_over_time(trace), use_container_width=True)
    with c3:
        st.plotly_chart(plot_cumulative_revenue(trace), use_container_width=True)

    if strategy == Q_STRATEGY:
        st.subheader("Learned pricing policy")
        st.plotly_chart(plot_policy_heatmap(agent), use_container_width=True)


if __name__ == "__main__":
    main()
