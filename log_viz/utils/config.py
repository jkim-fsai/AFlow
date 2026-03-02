"""Configuration constants for the AFlow visualization dashboard."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"

# UI Configuration
PLOT_HEIGHT = 400
MAX_PREDICTIONS_DISPLAY = 200

# Cache TTLs (seconds)
CACHE_TTL_DATASETS = 30
CACHE_TTL_RESULTS = 10

# Color scheme (TensorBoard-inspired, consistent with aa-context-optimization)
COLORS = {
    "primary": "#FF6F00",
    "secondary": "#0091EA",
    "success": "#00C853",
    "warning": "#FFD600",
    "error": "#D50000",
    "mcts_success": "#00C853",
    "mcts_failure": "#D50000",
    "mcts_node": "#0091EA",
    "mcts_best": "#FF6F00",
    "validation": "#FF6F00",
    "test": "#0091EA",
}
