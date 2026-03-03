"""Prompt evolution viewer."""

from typing import List

import streamlit as st

from data_loader import AFlowDataLoader


def display_prompt_evolution(
    loader: AFlowDataLoader,
    dataset: str,
    rounds: List[int],
) -> None:
    """Display prompt templates across rounds in expanders."""
    if not rounds:
        st.info("No rounds to display.")
        return

    for round_num in rounds:
        prompt = loader.load_round_prompt(dataset, round_num)
        if prompt is None:
            continue

        # Check if prompt has content beyond boilerplate
        has_content = any(
            line.strip() and not line.strip().startswith("#")
            for line in prompt.split("\n")
            if "XXX_PROMPT" not in line and '"""' not in line and "'''" not in line
        )

        label = f"Round {round_num}"
        if not has_content:
            label += " (empty)"

        with st.expander(label, expanded=(round_num == 1)):
            st.code(prompt, language="python")
