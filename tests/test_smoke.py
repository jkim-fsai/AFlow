"""Smoke tests for AFlow core modules."""

import importlib


def test_optimizer_imports():
    """Verify optimizer module imports without errors."""
    mod = importlib.import_module("scripts.optimizer")
    assert hasattr(mod, "Optimizer")
    assert hasattr(mod, "GraphOptimize")


def test_experience_utils_imports():
    """Verify experience_utils module imports without errors."""
    mod = importlib.import_module("scripts.optimizer_utils.experience_utils")
    assert hasattr(mod, "ExperienceUtils")


def test_graph_optimize_model():
    """Verify GraphOptimize pydantic model has expected fields."""
    from scripts.optimizer import GraphOptimize

    fields = set(GraphOptimize.model_fields.keys())
    assert "modification" in fields
    assert "graph" in fields
    assert "prompt" in fields
    assert "short_label" in fields
