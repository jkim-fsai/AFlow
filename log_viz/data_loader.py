"""Data loading for AFlow optimization results."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from utils.config import CACHE_TTL_DATASETS, CACHE_TTL_RESULTS, WORKSPACE_DIR


class AFlowDataLoader:
    """Loads AFlow optimization data from workspace directories."""

    def __init__(self, workspace_root: Path = WORKSPACE_DIR):
        self.workspace_root = workspace_root

    def _workflows_path(self, dataset: str) -> Path:
        return self.workspace_root / dataset / "workflows"

    def _workflows_test_path(self, dataset: str) -> Path:
        return self.workspace_root / dataset / "workflows_test"

    @st.cache_data(ttl=CACHE_TTL_DATASETS)
    def get_available_datasets(_self) -> List[str]:
        """Discover datasets under workspace/ that have results."""
        datasets = []
        if not _self.workspace_root.exists():
            return datasets
        for d in sorted(_self.workspace_root.iterdir()):
            results_path = d / "workflows" / "results.json"
            if d.is_dir() and results_path.exists() and results_path.stat().st_size > 0:
                datasets.append(d.name)
        return datasets

    @st.cache_data(ttl=CACHE_TTL_RESULTS)
    def load_validation_results(_self, dataset: str) -> pd.DataFrame:
        """Load workflows/results.json as DataFrame."""
        path = _self._workflows_path(dataset) / "results.json"
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values("round").reset_index(drop=True)
        return df

    @st.cache_data(ttl=CACHE_TTL_RESULTS)
    def load_test_results(_self, dataset: str) -> Optional[pd.DataFrame]:
        """Load workflows_test/results.json if it exists."""
        path = _self._workflows_test_path(dataset) / "results.json"
        if not path.exists() or path.stat().st_size == 0:
            return None
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values("round").reset_index(drop=True)
        return df

    @st.cache_data(ttl=CACHE_TTL_RESULTS)
    def load_mcts_tree(_self, dataset: str) -> Dict:
        """Load processed_experience.json (MCTS tree)."""
        path = _self._workflows_path(dataset) / "processed_experience.json"
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def get_available_rounds(self, dataset: str) -> List[int]:
        """Discover round_N directories."""
        wf = self._workflows_path(dataset)
        rounds = []
        if not wf.exists():
            return rounds
        for d in wf.iterdir():
            m = re.match(r"round_(\d+)", d.name)
            if m and d.is_dir():
                rounds.append(int(m.group(1)))
        return sorted(rounds)

    def load_round_experience(self, dataset: str, round_num: int) -> Optional[Dict]:
        """Load round_N/experience.json."""
        path = self._workflows_path(dataset) / f"round_{round_num}" / "experience.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def load_round_graph(self, dataset: str, round_num: int) -> Optional[str]:
        """Load round_N/graph.py as string."""
        path = self._workflows_path(dataset) / f"round_{round_num}" / "graph.py"
        if not path.exists():
            return None
        return path.read_text()

    def load_round_prompt(self, dataset: str, round_num: int) -> Optional[str]:
        """Load round_N/prompt.py as string."""
        path = self._workflows_path(dataset) / f"round_{round_num}" / "prompt.py"
        if not path.exists():
            return None
        return path.read_text()

    def load_round_log(self, dataset: str, round_num: int) -> Optional[List[Dict]]:
        """Load round_N/log.json (failed/incorrect predictions)."""
        path = self._workflows_path(dataset) / f"round_{round_num}" / "log.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def load_operator_definitions(self, dataset: str) -> Optional[Dict]:
        """Load template/operator.json."""
        path = self._workflows_path(dataset) / "template" / "operator.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
