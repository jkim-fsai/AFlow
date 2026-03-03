"""Validation vs Test comparison page."""

import streamlit as st

from components.sidebar import render_sidebar
from data_loader import AFlowDataLoader
from plots import create_val_vs_test_comparison
from utils.config import WORKSPACE_DIR

st.set_page_config(
    page_title="Val vs Test - AFlow",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_data_loader():
    return AFlowDataLoader(WORKSPACE_DIR)


loader = get_data_loader()
sidebar_state = render_sidebar(loader)
dataset = sidebar_state["dataset"]

st.title("Validation vs Test Comparison")
st.caption(f"Dataset: {dataset}")

val_df = loader.load_validation_results(dataset)
test_df = loader.load_test_results(dataset)

if val_df.empty:
    st.warning(f"No validation results for {dataset}.")
    st.stop()

if test_df is None or test_df.empty:
    st.warning(f"No test results for {dataset}. Run test evaluation first.")
    st.stop()

# Comparison chart
fig = create_val_vs_test_comparison(val_df, test_df)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Side-by-side tables
col1, col2 = st.columns(2)

with col1:
    st.subheader("Validation Results")
    display_val = val_df.copy()
    display_val["score"] = display_val["score"].apply(lambda x: f"{x:.1%}")
    st.dataframe(
        display_val[["round", "score", "time"]],
        use_container_width=True,
        hide_index=True,
    )

with col2:
    st.subheader("Test Results")
    display_test = test_df.copy()
    display_test["score"] = display_test["score"].apply(lambda x: f"{x:.1%}")
    st.dataframe(
        display_test[["round", "score", "time"]],
        use_container_width=True,
        hide_index=True,
    )

# Generalization analysis
st.divider()
st.subheader("Generalization Analysis")

common_rounds = sorted(set(val_df["round"].tolist()) & set(test_df["round"].tolist()))
if common_rounds:
    for r in common_rounds:
        val_score = val_df.loc[val_df["round"] == r, "score"].iloc[0]
        test_score = test_df.loc[test_df["round"] == r, "score"].iloc[0]
        gap = val_score - test_score

        col_a, col_b, col_c = st.columns(3)
        col_a.metric(f"Round {r} - Validation", f"{val_score:.1%}")
        col_b.metric(f"Round {r} - Test", f"{test_score:.1%}")
        col_c.metric(
            f"Round {r} - Gap",
            f"{gap:.1%}",
            delta=f"{gap / val_score:.0%} relative" if val_score > 0 else None,
            delta_color="inverse",
        )
