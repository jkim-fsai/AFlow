"""Workflow code evolution viewer."""

from typing import List

import streamlit as st

from data_loader import AFlowDataLoader


def display_workflow_code_single(
    loader: AFlowDataLoader,
    dataset: str,
    round_num: int,
) -> None:
    """Display workflow code (graph.py) for a single round."""
    code = loader.load_round_graph(dataset, round_num)
    if code:
        st.code(code, language="python", line_numbers=True)
    else:
        st.warning(f"No graph.py found for round {round_num}")
