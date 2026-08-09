"""
Streamlit dashboard for the Q-learning dynamic pricing agent.

Run from the project root:

    PYTHONPATH=src streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

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


st.set_page_config(
    page_title="Q-Learning Dynamic Pricing",
    page_icon=None,
    layout="wide",
)


@st.cache_resource
def load_q_agent() -> Optional[QLearningAgent]:
    if Q_MODEL_PATH.exists():
        return QLearningAgent.load(Q_MODEL_PATH)
    return None


def main() -> None:
    st.title("Q-Learning Dynamic Pricing Agent")
    st.caption(
        "Simulate a trained tabular Q-learning policy over a fixed ticket inventory "
        "and selling window."
    )

    agent = load_q_agent()

    with st.sidebar:
        st.header("Simulation settings")
        initial_inventory = st.slider("Initial ticket inventory", 20, 300, 100, 10)
        selling_days = st.slider("Selling days", 5, 60, 20, 1)
        demand_level = st.slider("Demand level", 0.2, 2.5, 1.0, 0.1)
        terminal_penalty = st.slider(
            "Terminal unsold inventory penalty", 0.0, 20.0, 0.0, 0.5
        )
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        run_button = st.button("Run simulation", type="primary")

        if agent is None:
            st.error(
                "No Q-learning model found.\n\n"
                "Train one with:\n"
                "`PYTHONPATH=src python scripts/train_q_learning.py`"
            )

    if agent is None:
        return

    if not run_button:
        st.markdown(
            """
            Use the sidebar to configure inventory, selling days, and demand,
            then click **Run simulation**.

            The agent uses a **greedy Q-learning policy** (no exploration) loaded from
            `models/q_learning_agent.json`.
            """
        )
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
        strategy_name="Q-Learning",
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

    st.subheader("Learned pricing policy")
    st.plotly_chart(plot_policy_heatmap(agent), use_container_width=True)


if __name__ == "__main__":
    main()
