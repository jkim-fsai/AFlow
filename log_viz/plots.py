"""Plotly visualization functions for AFlow optimization results."""

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from utils.config import COLORS, PLOT_HEIGHT


def create_score_progression_plot(
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Line chart of scores across MCTS rounds."""
    fig = go.Figure()

    if not val_df.empty:
        fig.add_trace(
            go.Scatter(
                x=val_df["round"],
                y=val_df["score"] * 100,
                mode="lines+markers",
                name="Validation",
                line=dict(color=COLORS["validation"], width=2),
                marker=dict(size=8),
                hovertemplate=("<b>Round %{x}</b><br>Score: %{y:.1f}%<extra></extra>"),
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

    fig.update_layout(
        title="Score Progression",
        xaxis_title="MCTS Round",
        yaxis_title="F1 Score (%)",
        template="plotly_white",
        height=PLOT_HEIGHT,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(dtick=1),
    )
    return fig


def create_running_max_plot(val_df: pd.DataFrame) -> go.Figure:
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

    fig.update_layout(
        title="Running Maximum",
        xaxis_title="MCTS Round",
        yaxis_title="F1 Score (%)",
        template="plotly_white",
        height=PLOT_HEIGHT,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
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


def create_mcts_tree_figure(
    nodes: List[Dict],
    edges: List[Dict],
) -> go.Figure:
    """Render MCTS tree as a Plotly scatter + line figure.

    Args:
        nodes: [{id, x, y, score, is_best, label}]
        edges: [{x0, y0, x1, y1, success, modification}]
    """
    fig = go.Figure()

    # Draw edges
    for edge in edges:
        color = COLORS["mcts_success"] if edge["success"] else COLORS["mcts_failure"]
        fig.add_trace(
            go.Scatter(
                x=[edge["x0"], edge["x1"], None],
                y=[edge["y0"], edge["y1"], None],
                mode="lines",
                line=dict(color=color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Edge label (modification) at midpoint
        mid_x = (edge["x0"] + edge["x1"]) / 2
        mid_y = (edge["y0"] + edge["y1"]) / 2
        label = "Improved" if edge["success"] else "Regressed"
        mod_text = edge.get("modification", "")
        if len(mod_text) > 80:
            mod_text = mod_text[:80] + "..."
        fig.add_trace(
            go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode="markers",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                hovertemplate=(f"<b>{label}</b><br>{mod_text}<extra></extra>"),
                showlegend=False,
            )
        )

    # Draw nodes
    for node in nodes:
        color = COLORS["mcts_best"] if node["is_best"] else COLORS["mcts_node"]
        size = 30 if node["is_best"] else 22
        fig.add_trace(
            go.Scatter(
                x=[node["x"]],
                y=[node["y"]],
                mode="markers+text",
                marker=dict(size=size, color=color, line=dict(width=2, color="white")),
                text=[node["label"]],
                textposition="bottom center",
                textfont=dict(size=11),
                hovertemplate=(
                    f"<b>Round {node['id']}</b><br>"
                    f"Score: {node['score'] * 100:.1f}%<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="MCTS Search Tree",
        template="plotly_white",
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, autorange="reversed"
        ),
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig
