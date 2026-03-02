"""Sidebar controls for the AFlow dashboard."""

from typing import Any, Dict

import streamlit as st

from data_loader import AFlowDataLoader


def render_sidebar(loader: AFlowDataLoader) -> Dict[str, Any]:
    """Render sidebar with dataset selector, round filter, and refresh controls."""
    with st.sidebar:
        st.header("AFlow Dashboard")

        # Dataset selector
        datasets = loader.get_available_datasets()
        if not datasets:
            st.error("No datasets found in workspace/")
            st.stop()

        dataset = st.selectbox("Dataset", datasets, index=0)

        # Round filter
        available_rounds = loader.get_available_rounds(dataset)
        selected_rounds = st.multiselect(
            "Filter Rounds",
            available_rounds,
            default=available_rounds,
            help="Select which rounds to display",
        )

        st.divider()

        # Operator definitions
        operators = loader.load_operator_definitions(dataset)
        if operators:
            with st.expander("Available Operators"):
                for name, info in operators.items():
                    st.markdown(f"**{name}**")
                    st.caption(info.get("description", ""))
                    st.code(info.get("interface", ""), language="python")

        st.divider()

        # Auto-refresh controls
        auto_refresh = st.toggle("Auto-refresh", value=False)
        refresh_interval = st.slider(
            "Refresh interval (s)",
            min_value=1,
            max_value=10,
            value=5,
            disabled=not auto_refresh,
        )

        if st.button("Force Refresh"):
            st.cache_data.clear()
            st.rerun()

    return {
        "dataset": dataset,
        "selected_rounds": selected_rounds,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval,
    }
