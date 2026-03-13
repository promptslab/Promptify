"""Tests for evaluation system."""

from __future__ import annotations

import json

import pytest

from promptify.eval.evaluator import evaluate
from promptify.eval.metrics import accuracy, exact_match, f1, precision, recall
from promptify.eval.datasets import load_dataset
from promptify.core.exceptions import EvaluationError


# --- Metric unit tests ---


class TestMetrics:
    def test_precision_perfect(self):
        assert precision(["a", "b"], ["a", "b"]) == 1.0

    def test_precision_partial(self):
        assert precision(["a", "b", "c"], ["a", "b"]) == pytest.approx(2 / 3, abs=0.01)

    def test_precision_empty(self):
        assert precision([], ["a"]) == 0.0

    def test_recall_perfect(self):
        assert recall(["a", "b"], ["a", "b"]) == 1.0

    def test_recall_partial(self):
        assert recall(["a"], ["a", "b"]) == 0.5

    def test_f1_perfect(self):
        assert f1(["a", "b"], ["a", "b"]) == 1.0

    def test_f1_zero(self):
        assert f1(["x"], ["a", "b"]) == 0.0

    def test_accuracy(self):
        assert accuracy(["a", "b", "c"], ["a", "b", "x"]) == pytest.approx(2 / 3, abs=0.01)

    def test_exact_match_true(self):
        assert exact_match("hello", "hello") == 1.0

    def test_exact_match_false(self):
        assert exact_match("hello", "world") == 0.0


# --- Dataset loading ---


class TestDatasets:
    def test_load_from_list(self):
        data = [
            {"input": "text1", "expected": {"label": "pos"}},
            {"input": "text2", "expected": {"label": "neg"}},
        ]
        result = load_dataset(data)
        assert len(result) == 2

    def test_load_missing_key(self):
        with pytest.raises(ValueError, match="missing 'input'"):
            load_dataset([{"expected": "foo"}])

    def test_load_missing_expected(self):
        with pytest.raises(ValueError, match="missing 'expected'"):
            load_dataset([{"input": "text"}])


# --- Evaluator ---


class MockTask:
    """Fake task that returns predictable results."""

    def __init__(self, responses):
        self._responses = responses
        self._idx = 0

    def __call__(self, text, **kwargs):
        result = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return result


class TestEvaluator:
    def test_evaluate_exact_match(self):
        task = MockTask(["yes", "no", "yes"])
        dataset = [
            {"input": "q1", "expected": "yes"},
            {"input": "q2", "expected": "no"},
            {"input": "q3", "expected": "yes"},
        ]
        scores = evaluate(task=task, dataset=dataset, metrics=["exact_match"])
        assert scores["exact_match"] == 1.0

    def test_evaluate_partial_match(self):
        task = MockTask(["yes", "yes"])
        dataset = [
            {"input": "q1", "expected": "yes"},
            {"input": "q2", "expected": "no"},
        ]
        scores = evaluate(task=task, dataset=dataset, metrics=["exact_match"])
        assert scores["exact_match"] == 0.5

    def test_evaluate_empty_dataset(self):
        task = MockTask(["yes"])
        with pytest.raises(EvaluationError, match="empty"):
            evaluate(task=task, dataset=[], metrics=["exact_match"])

    def test_evaluate_max_samples(self):
        task = MockTask(["yes", "no", "yes"])
        dataset = [
            {"input": "q1", "expected": "yes"},
            {"input": "q2", "expected": "no"},
            {"input": "q3", "expected": "yes"},
        ]
        scores = evaluate(task=task, dataset=dataset, metrics=["exact_match"], max_samples=2)
        # Sample 0: task returns "yes", expected "yes" → match
        # Sample 1: task returns "no", expected "no" → match
        assert scores["exact_match"] == 1.0

    def test_evaluate_with_progress(self):
        progress_calls = []
        task = MockTask(["yes"])
        dataset = [{"input": "q1", "expected": "yes"}]
        evaluate(
            task=task,
            dataset=dataset,
            metrics=["exact_match"],
            progress_callback=lambda i, t: progress_calls.append((i, t)),
        )
        assert progress_calls == [(1, 1)]
