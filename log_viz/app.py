"""Main Streamlit dashboard for AFlow optimization visualization."""

import time

import streamlit as st

from components.mcts_tree import display_mcts_tree
from components.metrics_cards import display_metrics_cards
from components.predictions_table import display_predictions
from components.prompt_viewer import display_prompt_evolution
from components.sidebar import render_sidebar
from components.workflow_diagram import display_workflow_diagrams
from components.workflow_viewer import display_workflow_evolution
from data_loader import AFlowDataLoader
from plots import create_running_max_plot, create_score_progression_plot
from utils.config import WORKSPACE_DIR

# Page config
st.set_page_config(
    page_title="AFlow Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_data_loader():
    return AFlowDataLoader(WORKSPACE_DIR)


loader = get_data_loader()
sidebar_state = render_sidebar(loader)
dataset = sidebar_state["dataset"]
selected_rounds = sidebar_state["selected_rounds"]
auto_refresh = sidebar_state["auto_refresh"]
refresh_interval = sidebar_state["refresh_interval"]

# Title
st.title("AFlow Optimization Dashboard")
st.markdown("*MCTS-based agentic workflow optimization*")
st.caption(
    "AFlow: Automating Agentic Workflow Generation "
    "([Zhang et al., ICLR 2025](https://arxiv.org/abs/2410.10762))"
)

# Load data
val_df = loader.load_validation_results(dataset)
test_df = loader.load_test_results(dataset)
tree_data = loader.load_mcts_tree(dataset)

if val_df.empty:
    st.warning(f"No results found for {dataset}.")
    st.stop()

# Filter to selected rounds
filtered_df = val_df[val_df["round"].isin(selected_rounds)].copy()

# --- Metrics Cards ---
display_metrics_cards(val_df, test_df)

st.divider()

# --- Charts ---
col1, col2 = st.columns(2)
with col1:
    fig = create_score_progression_plot(filtered_df, test_df)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = create_running_max_plot(filtered_df)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- MCTS Tree ---
st.subheader("MCTS Search Tree")
experiences = {}
for r in loader.get_available_rounds(dataset):
    exp = loader.load_round_experience(dataset, r)
    if exp:
        experiences[r] = exp
display_mcts_tree(tree_data, experiences)

st.divider()

# --- Workflow Diagrams ---
st.subheader("Workflow Structure Evolution")
display_workflow_diagrams(loader, dataset, selected_rounds)

st.divider()

# --- Workflow Code ---
st.subheader("Workflow Code")
display_workflow_evolution(loader, dataset, selected_rounds)

st.divider()

# --- Prompt Evolution ---
st.subheader("Prompt Evolution")
display_prompt_evolution(loader, dataset, selected_rounds)

st.divider()

# --- Prediction Details ---
st.subheader("Prediction Details (Incorrect)")
display_predictions(loader, dataset, selected_rounds)

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
