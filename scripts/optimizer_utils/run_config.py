"""Run configuration capture for AFlow experiment tracking."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """Full hyperparameter snapshot for one AFlow optimization or test run."""

    # Identity
    dataset: str
    mode: str  # "Graph" or "Test"
    question_type: str  # "qa", "math", "code"

    # Models
    opt_model: Optional[str] = None
    exec_model: str
    opt_temperature: Optional[float] = None
    opt_top_p: Optional[float] = None
    exec_temperature: float
    exec_top_p: float

    # Search budget
    sample: int
    max_rounds: int
    validation_rounds: int
    check_convergence: bool
    initial_round: int

    # MCTS parameters
    mcts_alpha: float = Field(description="Softmax temperature for round selection")
    mcts_lambda: float = Field(
        description="Exploration-exploitation mix (higher = more exploration)"
    )

    # Convergence detection
    convergence_top_k: int = 3
    convergence_z: float = 0
    convergence_consecutive: int = 5

    # Retry limits
    max_retries: int = 1
    max_generation_retries: int = 20

    # Evaluation
    max_concurrent_tasks: int = 50
    log_samples: int = 3

    # Dataset sizes
    valset_size: Optional[int] = None
    testset_size: Optional[int] = None

    # Runtime metadata (filled at end of run)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    rounds_completed: Optional[int] = None
    best_score: Optional[float] = None
    baseline_score: Optional[float] = None  # Round 1 score (unoptimized baseline)
    total_cost: Optional[float] = None
    runtime_seconds: Optional[float] = None  # Wall-clock seconds from start to end
    converged: Optional[bool] = None

    # Test-mode specific
    test_rounds: Optional[List[int]] = None


def write_run_config(config: RunConfig, output_dir: str) -> Path:
    """Write run_config.json at the start of a run."""
    path = Path(output_dir) / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=4, ensure_ascii=False)
    return path


def update_run_config(output_dir: str, updates: Dict[str, Any]) -> Path:
    """Patch runtime metadata into existing run_config.json."""
    path = Path(output_dir) / "run_config.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.update(updates)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return path


def load_run_config(output_dir: str) -> Optional[Dict[str, Any]]:
    """Load run_config.json if it exists."""
    path = Path(output_dir) / "run_config.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
