"""Summary metric cards for the AFlow dashboard."""

from typing import Optional

import pandas as pd
import streamlit as st


def display_metrics_cards(
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
) -> None:
    """Display summary metrics in card format."""
    if val_df.empty:
        st.warning("No validation results available.")
        return

    best_score = val_df["score"].max()
    baseline_score = val_df.iloc[0]["score"]
    best_round = int(val_df.loc[val_df["score"].idxmax(), "round"])
    improvement = best_score - baseline_score

    # Row 1: Core metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rounds", len(val_df))
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
    c5.metric("Best Round", f"Round {best_round}")
    c6.metric("Mean Score", f"{val_df['score'].mean():.1%}")
    c7.metric("Std Dev", f"{val_df['score'].std():.4f}")

    if test_df is not None and not test_df.empty:
        test_best = test_df["score"].max()
        gap = best_score - test_best
        c8.metric(
            "Test Score (Best)",
            f"{test_best:.1%}",
            delta=f"-{gap:.1%} gap" if gap > 0 else f"+{-gap:.1%}",
            delta_color="inverse",
        )
    else:
        c8.metric("Test Score", "N/A")
