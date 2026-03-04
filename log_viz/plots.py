"""Plotly visualization functions for AFlow optimization results."""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from utils.config import COLORS, PLOT_HEIGHT


def create_score_progression_plot(
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    source: str = "val",
) -> go.Figure:
    """Line chart of scores across MCTS rounds with running maximum overlay."""
    fig = go.Figure()

    if not val_df.empty:
        fig.add_trace(
            go.Scatter(
                x=val_df["round"],
                y=val_df["score"] * 100,
                mode="lines+markers",
                name="Round Score",
                line=dict(color=COLORS["validation"], width=2),
                marker=dict(size=8),
                hovertemplate=("<b>Round %{x}</b><br>Score: %{y:.1f}%<extra></extra>"),
            )
        )

        # Running maximum overlay
        running_max = val_df["score"].cummax() * 100
        fig.add_trace(
            go.Scatter(
                x=val_df["round"],
                y=running_max,
                mode="lines",
                name="Running Max",
                line=dict(color=COLORS["success"], width=2, dash="dash"),
                hovertemplate=(
                    "<b>Round %{x}</b><br>Best so far: %{y:.1f}%<extra></extra>"
                ),
            )
        )

    if test_df is not None and not test_df.empty:
        fig.add_trace(
            go.Scatter(
                x=test_df["round"],
                y=test_df["score"] * 100,
                mode="lines+markers",
                name="Test",
                line=dict(color=COLORS["test"], width=2, dash="dash"),
                marker=dict(size=8, symbol="diamond"),
                hovertemplate=("<b>Round %{x}</b><br>Score: %{y:.1f}%<extra></extra>"),
            )
        )

    split_label = source.upper()
    fig.update_layout(
        title="Score Progression",
        xaxis_title="MCTS Round",
        yaxis_title=f"Score — {split_label} (%)",
        template="plotly_white",
        height=PLOT_HEIGHT,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(dtick=1),
    )
    return fig


def create_running_max_plot(val_df: pd.DataFrame, source: str = "val") -> go.Figure:
    """Running maximum score tracker."""
    fig = go.Figure()

    if not val_df.empty:
        running_max = val_df["score"].cummax() * 100

        fig.add_trace(
            go.Scatter(
                x=val_df["round"],
                y=running_max,
                mode="lines",
                name="Best So Far",
                fill="tozeroy",
                line=dict(color=COLORS["success"], width=2),
                hovertemplate=("<b>Round %{x}</b><br>Best: %{y:.1f}%<extra></extra>"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=val_df["round"],
                y=val_df["score"] * 100,
                mode="markers",
                name="Round Score",
                marker=dict(color=COLORS["secondary"], size=8),
                hovertemplate=("<b>Round %{x}</b><br>Score: %{y:.1f}%<extra></extra>"),
            )
        )

    split_label = source.upper()
    fig.update_layout(
        title="Running Maximum",
        xaxis_title="MCTS Round",
        yaxis_title=f"Score — {split_label} (%)",
        template="plotly_white",
        height=PLOT_HEIGHT,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(dtick=1),
    )
    return fig


def create_cost_progression_plot(val_df: pd.DataFrame) -> go.Figure:
    """Bar chart of total cost per round."""
    fig = go.Figure()

    if not val_df.empty and "total_cost" in val_df.columns:
        fig.add_trace(
            go.Bar(
                x=val_df["round"],
                y=val_df["total_cost"],
                name="Round Cost",
                marker_color=COLORS["secondary"],
                hovertemplate="<b>Round %{x}</b><br>Cost: $%{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Cost per Round",
        xaxis_title="MCTS Round",
        yaxis_title="Cost (USD)",
        template="plotly_white",
        height=PLOT_HEIGHT,
        xaxis=dict(dtick=1),
    )
    return fig


def create_val_vs_test_comparison(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> go.Figure:
    """Grouped bar chart comparing validation vs test scores."""
    # Find rounds present in both
    common_rounds = sorted(
        set(val_df["round"].tolist()) & set(test_df["round"].tolist())
    )

    val_scores = []
    test_scores = []
    for r in common_rounds:
        val_scores.append(val_df.loc[val_df["round"] == r, "score"].iloc[0] * 100)
        test_scores.append(test_df.loc[test_df["round"] == r, "score"].iloc[0] * 100)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"Round {r}" for r in common_rounds],
            y=val_scores,
            name="Validation",
            marker_color=COLORS["validation"],
            hovertemplate="<b>%{x}</b><br>Val: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[f"Round {r}" for r in common_rounds],
            y=test_scores,
            name="Test",
            marker_color=COLORS["test"],
            hovertemplate="<b>%{x}</b><br>Test: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Validation vs Test Scores",
        yaxis_title="F1 Score (%)",
        template="plotly_white",
        height=PLOT_HEIGHT,
        barmode="group",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig
