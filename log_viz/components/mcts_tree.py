"""MCTS tree visualization component."""

from collections import deque
from typing import Dict, List, Optional, Tuple

import streamlit as st

from plots import create_mcts_tree_figure


def _build_tree(
    tree_data: Dict,
) -> Tuple[List[Dict], List[Dict]]:
    """Build node and edge lists from processed_experience.json.

    Returns (nodes, edges) ready for create_mcts_tree_figure.
    """
    # Build adjacency: parent -> [(child, success, modification, score)]
    children: Dict[int, List] = {}
    all_rounds = set()
    scores: Dict[int, float] = {}

    for parent_str, parent_data in tree_data.items():
        parent_id = int(parent_str)
        all_rounds.add(parent_id)
        scores[parent_id] = parent_data.get("score", 0)
        if parent_id not in children:
            children[parent_id] = []

        for child_str, child_info in parent_data.get("success", {}).items():
            child_id = int(child_str)
            all_rounds.add(child_id)
            scores[child_id] = child_info["score"]
            children[parent_id].append(
                (child_id, True, child_info.get("modification", ""))
            )

        for child_str, child_info in parent_data.get("failure", {}).items():
            child_id = int(child_str)
            all_rounds.add(child_id)
            scores[child_id] = child_info["score"]
            children[parent_id].append(
                (child_id, False, child_info.get("modification", ""))
            )

    if not all_rounds:
        return [], []

    # Find root (node that is never a child)
    child_set = {c[0] for cs in children.values() for c in cs}
    roots = all_rounds - child_set
    root = min(roots) if roots else min(all_rounds)

    # BFS to assign positions
    best_round = max(scores, key=scores.get) if scores else root
    positions: Dict[int, Tuple[float, float]] = {}
    level_counts: Dict[int, int] = {}
    queue = deque([(root, 0)])
    visited = {root}

    # First pass: count nodes per level
    temp_queue = deque([(root, 0)])
    temp_visited = {root}
    while temp_queue:
        node, level = temp_queue.popleft()
        level_counts[level] = level_counts.get(level, 0) + 1
        for child_id, _, _ in children.get(node, []):
            if child_id not in temp_visited:
                temp_visited.add(child_id)
                temp_queue.append((child_id, level + 1))

    # Second pass: assign positions
    level_index: Dict[int, int] = {}
    while queue:
        node, level = queue.popleft()
        idx = level_index.get(level, 0)
        total = level_counts.get(level, 1)
        x = (idx + 0.5) / total  # Center nodes within level
        positions[node] = (x, level)
        level_index[level] = idx + 1

        for child_id, _, _ in children.get(node, []):
            if child_id not in visited:
                visited.add(child_id)
                queue.append((child_id, level + 1))

    # Build output
    nodes = []
    for round_id in sorted(all_rounds):
        x, y = positions.get(round_id, (0, 0))
        score = scores.get(round_id, 0)
        nodes.append(
            {
                "id": round_id,
                "x": x,
                "y": y,
                "score": score,
                "is_best": round_id == best_round,
                "label": f"R{round_id}\n{score:.1%}",
            }
        )

    edges = []
    for parent_id, child_list in children.items():
        px, py = positions.get(parent_id, (0, 0))
        for child_id, success, modification in child_list:
            cx, cy = positions.get(child_id, (0, 0))
            edges.append(
                {
                    "x0": px,
                    "y0": py,
                    "x1": cx,
                    "y1": cy,
                    "success": success,
                    "modification": modification,
                }
            )

    return nodes, edges


def display_mcts_tree(
    tree_data: Dict,
    experiences: Optional[Dict[int, Dict]] = None,
) -> None:
    """Render the MCTS search tree."""
    if not tree_data:
        st.info("No MCTS experience data available.")
        return

    nodes, edges = _build_tree(tree_data)
    if not nodes:
        st.info("No tree nodes to display.")
        return

    fig = create_mcts_tree_figure(nodes, edges)
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    cols = st.columns(3)
    cols[0].markdown(":large_green_circle: **Green edge** = Improved over parent")
    cols[1].markdown(":red_circle: **Red edge** = Regressed from parent")
    cols[2].markdown(":large_orange_circle: **Orange node** = Best round")

    # Modification details
    if experiences:
        with st.expander("Modification Details"):
            for round_num in sorted(experiences.keys()):
                exp = experiences[round_num]
                status = "Improved" if exp.get("succeed") else "Regressed"
                st.markdown(
                    f"**Round {round_num}** (from Round {exp.get('father_node', '?')}): "
                    f"{status} ({exp.get('before', 0):.1%} -> {exp.get('after', 0):.1%})"
                )
                st.caption(exp.get("modification", ""))
