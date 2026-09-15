"""Plotly charts for training and evaluation results."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from agents.q_learning_agent import QLearningAgent
from evaluation.evaluator import EpisodeTrace
from evaluation.metrics import EpisodeResult, moving_average, results_to_frame, summarize_results


def plot_training_rewards(
    rewards: Sequence[float],
    *,
    title: str = "Q-Learning training reward over episodes",
    window: int = 50,
) -> go.Figure:
    """Plot episode rewards and an optional moving average."""
    episodes = np.arange(1, len(rewards) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=list(rewards),
            mode="lines",
            name="Episode reward",
            opacity=0.35,
            line=dict(color="#6c757d"),
        )
    )
    if len(rewards) >= window:
        ma = moving_average(rewards, window=window)
        fig.add_trace(
            go.Scatter(
                x=np.arange(window, len(rewards) + 1),
                y=ma,
                mode="lines",
                name=f"{window}-episode MA",
                line=dict(color="#0B6E4F", width=2.5),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Total reward",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_revenue_summary(results: Sequence[EpisodeResult]) -> go.Figure:
    """Plot average revenue and standard deviation by strategy."""
    summary = summarize_results(results)
    fig = px.bar(
        summary,
        x="strategy",
        y="avg_total_revenue",
        error_y="std_total_revenue",
        title="Average total revenue (Q-Learning)",
        labels={"strategy": "Strategy", "avg_total_revenue": "Avg total revenue ($)"},
        color_discrete_sequence=["#0B6E4F"],
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_sellthrough_summary(results: Sequence[EpisodeResult]) -> go.Figure:
    summary = summarize_results(results)
    fig = px.bar(
        summary,
        x="strategy",
        y="avg_sell_through_pct",
        title="Average sell-through rate (Q-Learning)",
        labels={
            "strategy": "Strategy",
            "avg_sell_through_pct": "Sell-through (%)",
        },
        color_discrete_sequence=["#3D5A80"],
    )
    fig.update_layout(template="plotly_white", showlegend=False, yaxis_range=[0, 100])
    return fig


def plot_price_over_time(trace: EpisodeTrace) -> go.Figure:
    days = list(range(1, len(trace.prices) + 1))
    fig = px.line(
        x=days,
        y=trace.prices,
        markers=True,
        title=f"Ticket price over time — {trace.strategy}",
        labels={"x": "Day", "y": "Price ($)"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_inventory_over_time(trace: EpisodeTrace) -> go.Figure:
    steps = list(range(len(trace.inventory)))
    fig = px.line(
        x=steps,
        y=trace.inventory,
        markers=True,
        title=f"Tickets remaining over time — {trace.strategy}",
        labels={"x": "Step", "y": "Tickets remaining"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_cumulative_revenue(trace: EpisodeTrace) -> go.Figure:
    steps = list(range(len(trace.cumulative_revenue)))
    fig = px.line(
        x=steps,
        y=trace.cumulative_revenue,
        markers=True,
        title=f"Cumulative revenue over time — {trace.strategy}",
        labels={"x": "Step", "y": "Cumulative revenue ($)"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_policy_heatmap(agent: QLearningAgent) -> go.Figure:
    """Plot the preferred price for each inventory and time bin."""
    inventory_order = list(agent.INVENTORY_BINS)
    time_order = list(agent.TIME_BINS)

    z = []
    text = []
    for inv in inventory_order:
        row_prices = []
        row_text = []
        for time_bin in time_order:
            price = agent.preferred_price_for_bins(inv, time_bin)
            row_prices.append(price)
            row_text.append(f"${price:.0f}")
        z.append(row_prices)
        text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=time_order,
            y=inventory_order,
            text=text,
            texttemplate="%{text}",
            colorscale="YlGnBu",
            colorbar=dict(title="Price ($)"),
        )
    )
    fig.update_layout(
        title="Learned pricing policy heatmap (Q-Learning)",
        xaxis_title="Time remaining",
        yaxis_title="Inventory remaining",
        template="plotly_white",
    )
    return fig


def plot_revenue_distribution(results: Sequence[EpisodeResult]) -> go.Figure:
    df = results_to_frame(results)
    fig = px.box(
        df,
        x="strategy",
        y="total_revenue",
        title="Revenue distribution (Q-Learning)",
        labels={"strategy": "Strategy", "total_revenue": "Total revenue ($)"},
        color_discrete_sequence=["#0B6E4F"],
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def save_figure(fig: go.Figure, path: str) -> None:
    """Write a Plotly figure to an HTML file."""
    fig.write_html(path, include_plotlyjs="cdn")
