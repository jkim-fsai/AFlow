"""Workflow operator flow diagram — visualizes the pipeline of operators per round."""

import ast
import re
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st

from data_loader import AFlowDataLoader

# Operator colors
OP_COLORS = {
    "Custom": "#FF6F00",
    "AnswerGenerate": "#0091EA",
    "ScEnsemble": "#00C853",
    "Review": "#FFD600",
    "Conditional": "#9E9E9E",
    "Loop": "#AB47BC",
}

NODE_WIDTH = 1.6
NODE_HEIGHT = 0.5
X_CENTER = 0.0
Y_STEP = 1.0


def _parse_workflow_steps(code: str) -> List[Dict]:
    """Parse graph.py __call__ method to extract operator steps.

    Returns list of step dicts: {type, operator, label, detail, loop_count}
    """
    steps = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return steps

    # Find the __call__ method
    call_method = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "__call__":
                call_method = node
                break

    if call_method is None:
        return steps

    _extract_steps_from_body(call_method.body, steps)
    return steps


def _extract_steps_from_body(body: list, steps: list) -> None:
    """Recursively extract operator call steps from AST body."""
    for node in body:
        # For loop: look for operator calls inside
        if isinstance(node, ast.For):
            loop_count = _get_loop_count(node)
            inner_steps = []
            _extract_steps_from_body(node.body, inner_steps)
            if inner_steps:
                for s in inner_steps:
                    s["loop_count"] = loop_count
                    s["label"] = f"{loop_count}x {s['label']}"
                steps.extend(inner_steps)
            continue

        # If statement: mark as conditional
        if isinstance(node, ast.If):
            steps.append(
                {
                    "type": "conditional",
                    "operator": "Conditional",
                    "label": "Review\nCheck",
                    "detail": "if improved → re-generate",
                    "loop_count": None,
                }
            )
            continue

        # Assignment with await: operator call
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            value = node.value if isinstance(node, ast.Assign) else node.value
            call_node = _unwrap_await(value)
            if call_node and isinstance(call_node, ast.Call):
                op_info = _classify_operator_call(call_node, node)
                if op_info:
                    steps.append(op_info)
            continue

        # Expression statement with await (no assignment)
        if isinstance(node, ast.Expr):
            call_node = _unwrap_await(node.value)
            if call_node and isinstance(call_node, ast.Call):
                op_info = _classify_operator_call(call_node, node)
                if op_info:
                    steps.append(op_info)


def _unwrap_await(node) -> Optional[ast.AST]:
    """Unwrap Await and Subscript nodes to get the underlying call."""
    if isinstance(node, ast.Await):
        return node.value
    return None


def _get_loop_count(for_node: ast.For) -> int:
    """Extract loop count from `for _ in range(N):`."""
    if isinstance(for_node.iter, ast.Call):
        func = for_node.iter.func
        if isinstance(func, ast.Name) and func.id == "range":
            if for_node.iter.args:
                arg = for_node.iter.args[0]
                if isinstance(arg, ast.Constant):
                    return arg.value
    return 0


def _classify_operator_call(call_node: ast.Call, parent_node) -> Optional[Dict]:
    """Classify an operator call into a step dict."""
    func = call_node.func
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name):
        return None
    if func.value.id != "self":
        return None

    attr = func.attr
    op_map = {
        "custom": ("Custom", "Custom"),
        "answer_gen": ("AnswerGenerate", "Answer\nGenerate"),
        "ensemble": ("ScEnsemble", "Sc\nEnsemble"),
    }

    if attr not in op_map:
        return None

    operator, label = op_map[attr]

    # Check if this is a review call (custom with REVIEW_PROMPT or review-like input)
    detail = ""
    if attr == "custom":
        for kw in call_node.keywords:
            if kw.arg == "instruction":
                src = ast.dump(kw.value)
                if "REVIEW" in src:
                    operator = "Review"
                    label = "Self\nReview"
                    detail = "Expert review"

    return {
        "type": "operator",
        "operator": operator,
        "label": label,
        "detail": detail,
        "loop_count": None,
    }


def _draw_workflow_figure(steps: List[Dict], title: str = "") -> go.Figure:
    """Draw a vertical flowchart from a list of steps."""
    fig = go.Figure()

    if not steps:
        return fig

    n = len(steps)
    y_positions = [i * Y_STEP for i in range(n)]

    # Draw edges (arrows between consecutive nodes)
    for i in range(n - 1):
        fig.add_annotation(
            x=X_CENTER,
            y=-y_positions[i] - NODE_HEIGHT / 2,
            ax=X_CENTER,
            ay=-y_positions[i + 1] + NODE_HEIGHT / 2,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="#757575",
        )

    # Draw nodes
    for i, step in enumerate(steps):
        y = -y_positions[i]
        color = OP_COLORS.get(step["operator"], "#757575")

        # Rectangle shape
        fig.add_shape(
            type="rect",
            x0=X_CENTER - NODE_WIDTH / 2,
            y0=y - NODE_HEIGHT / 2,
            x1=X_CENTER + NODE_WIDTH / 2,
            y1=y + NODE_HEIGHT / 2,
            line=dict(color=color, width=2),
            fillcolor=color,
            opacity=0.15,
            layer="below",
        )

        # Border with full opacity
        fig.add_shape(
            type="rect",
            x0=X_CENTER - NODE_WIDTH / 2,
            y0=y - NODE_HEIGHT / 2,
            x1=X_CENTER + NODE_WIDTH / 2,
            y1=y + NODE_HEIGHT / 2,
            line=dict(color=color, width=2),
            fillcolor="rgba(0,0,0,0)",
        )

        # Node label
        fig.add_annotation(
            x=X_CENTER,
            y=y,
            text=f"<b>{step['label']}</b>",
            showarrow=False,
            font=dict(size=12, color=color),
        )

    total_height = max(250, n * 80 + 60)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_white",
        height=total_height,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[X_CENTER - 2, X_CENTER + 2],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def display_workflow_diagrams(
    loader: AFlowDataLoader,
    dataset: str,
    rounds: List[int],
) -> None:
    """Display side-by-side workflow flow diagrams for selected rounds."""
    if not rounds:
        st.info("No rounds to display.")
        return

    # Parse all workflows
    workflows = {}
    for r in rounds:
        code = loader.load_round_graph(dataset, r)
        if code:
            steps = _parse_workflow_steps(code)
            if steps:
                workflows[r] = steps

    if not workflows:
        st.info("No workflow diagrams to display.")
        return

    # Display in columns (max 3 per row)
    round_list = sorted(workflows.keys())
    for row_start in range(0, len(round_list), 3):
        row_rounds = round_list[row_start : row_start + 3]
        cols = st.columns(len(row_rounds))
        for col, r in zip(cols, row_rounds):
            with col:
                fig = _draw_workflow_figure(
                    workflows[r], title=f"Round {r}"
                )
                st.plotly_chart(fig, use_container_width=True)
