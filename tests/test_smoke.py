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


def test_strategyqa_benchmark_imports():
    """Verify StrategyQABenchmark imports and has expected methods."""
    from benchmarks.strategyqa import StrategyQABenchmark

    assert hasattr(StrategyQABenchmark, "evaluate_problem")
    assert hasattr(StrategyQABenchmark, "calculate_score")
    assert hasattr(StrategyQABenchmark, "extract_yes_no")


def test_arc_benchmark_imports():
    """Verify ARCBenchmark imports and has expected methods."""
    from benchmarks.arc import ARCBenchmark

    assert hasattr(ARCBenchmark, "evaluate_problem")
    assert hasattr(ARCBenchmark, "calculate_score")
    assert hasattr(ARCBenchmark, "extract_label")
    assert hasattr(ARCBenchmark, "format_choices")


def test_evaluator_includes_new_datasets():
    """Verify StrategyQA and ARC are registered in Evaluator."""
    from scripts.evaluator import Evaluator

    evaluator = Evaluator(eval_path="/tmp")
    assert "StrategyQA" in evaluator.dataset_configs
    assert "ARC" in evaluator.dataset_configs


def test_strategyqa_scoring():
    """Verify StrategyQA yes/no extraction and scoring."""
    from benchmarks.strategyqa import StrategyQABenchmark

    bench = StrategyQABenchmark("test", "/dev/null", "/tmp")
    assert bench.extract_yes_no("Yes, that is correct.") == "yes"
    assert bench.extract_yes_no("No") == "no"
    assert bench.extract_yes_no("The answer is yes because...") == "yes"
    score, _ = bench.calculate_score("yes", "Yes")
    assert score == 1.0
    score, _ = bench.calculate_score("no", "Yes")
    assert score == 0.0


def test_arc_scoring():
    """Verify ARC label extraction and scoring."""
    from benchmarks.arc import ARCBenchmark

    bench = ARCBenchmark("test", "/dev/null", "/tmp")
    assert bench.extract_label("A") == "A"
    assert bench.extract_label("The answer is B.") == "B"
    assert bench.extract_label("d") == "D"
    score, _ = bench.calculate_score("C", "C")
    assert score == 1.0
    score, _ = bench.calculate_score("A", "B")
    assert score == 0.0


def test_arc_format_choices():
    """Verify ARC choice formatting."""
    from benchmarks.arc import ARCBenchmark

    problem = {
        "question": "What is 1+1?",
        "choices": {"text": ["1", "2", "3"], "label": ["A", "B", "C"]},
        "answerKey": "B",
    }
    formatted = ARCBenchmark.format_choices(problem)
    assert "A. 1" in formatted
    assert "B. 2" in formatted
    assert "C. 3" in formatted
