"""Workflow code evolution viewer."""

from typing import Dict, List, Optional

import streamlit as st

from data_loader import AFlowDataLoader


def display_workflow_evolution(
    loader: AFlowDataLoader,
    dataset: str,
    rounds: List[int],
) -> None:
    """Display workflow code (graph.py) evolution across rounds in tabs."""
    if not rounds:
        st.info("No rounds to display.")
        return

    tabs = st.tabs([f"Round {r}" for r in rounds])

    for tab, round_num in zip(tabs, rounds):
        with tab:
            # Show modification that led to this round
            exp = loader.load_round_experience(dataset, round_num)
            if exp:
                status = "Improved" if exp.get("succeed") else "Regressed"
                delta = exp.get("after", 0) - exp.get("before", 0)
                sign = "+" if delta >= 0 else ""
                st.info(
                    f"**From Round {exp.get('father_node', '?')}** | "
                    f"{status} ({sign}{delta:.1%}) | "
                    f"{exp.get('modification', '')}"
                )

            # Show graph.py code
            code = loader.load_round_graph(dataset, round_num)
            if code:
                st.code(code, language="python", line_numbers=True)
            else:
                st.warning(f"No graph.py found for round {round_num}")
