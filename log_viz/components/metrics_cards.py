"""Summary metric cards for the AFlow dashboard."""

from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st


def _derive_run_id(df: pd.DataFrame) -> str:
    """Derive a yyyymmdd-hhmm run ID from the first round's timestamp."""
    if "time" not in df.columns:
        return "N/A"
    first_time = df.iloc[0]["time"]
    try:
        if isinstance(first_time, pd.Timestamp):
            return first_time.strftime("%Y%m%d-%H%M")
        dt = datetime.fromisoformat(str(first_time))
        return dt.strftime("%Y%m%d-%H%M")
    except (ValueError, TypeError):
        return "N/A"


def display_metrics_cards(df: pd.DataFrame) -> None:
    """Display summary metrics in card format."""
    if df.empty:
        st.warning("No results available.")
        return

    best_score = df["score"].max()
    baseline_score = df.iloc[0]["score"]
    best_round = int(df.loc[df["score"].idxmax(), "round"])
    improvement = best_score - baseline_score
    run_id = _derive_run_id(df)
    source = df["source"].iloc[0] if "source" in df.columns else "val"

    # Row 1: Core metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Run ID", run_id)
    c2.metric("Best Score", f"{best_score:.1%}")
    c3.metric("Baseline (R1)", f"{baseline_score:.1%}")
    c4.metric(
        "Improvement",
        f"+{improvement:.1%}",
        delta=(
            f"{improvement / baseline_score:.0%} relative"
            if baseline_score > 0
            else None
        ),
    )

    # Row 2: Additional metrics
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Rounds", len(df))
    c6.metric("Best Round", f"Round {best_round}")

    # Total cost
    total_cost = df["total_cost"].sum() if "total_cost" in df.columns else 0
    c7.metric("Total Cost", f"${total_cost:.4f}")

    c8.metric("Split", source.upper())
