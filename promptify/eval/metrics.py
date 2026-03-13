"""Evaluation metrics for NLP tasks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Set, Tuple


def _to_set(items: Any) -> Set[str]:
    """Convert items to a set of strings for comparison."""
    if isinstance(items, (list, tuple)):
        return {str(i) for i in items}
    return {str(items)}


def precision(predicted: List[Any], expected: List[Any]) -> float:
    """Compute precision: TP / (TP + FP)."""
    if not predicted:
        return 0.0
    pred_set = _to_set(predicted)
    exp_set = _to_set(expected)
    tp = len(pred_set & exp_set)
    return tp / len(pred_set) if pred_set else 0.0


def recall(predicted: List[Any], expected: List[Any]) -> float:
    """Compute recall: TP / (TP + FN)."""
    if not expected:
        return 0.0
    pred_set = _to_set(predicted)
    exp_set = _to_set(expected)
    tp = len(pred_set & exp_set)
    return tp / len(exp_set) if exp_set else 0.0


def f1(predicted: List[Any], expected: List[Any]) -> float:
    """Compute F1 score: harmonic mean of precision and recall."""
    p = precision(predicted, expected)
    r = recall(predicted, expected)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def accuracy(predicted: List[Any], expected: List[Any]) -> float:
    """Compute accuracy: fraction of correct predictions."""
    if not predicted or len(predicted) != len(expected):
        return 0.0
    correct = sum(1 for p, e in zip(predicted, expected) if str(p) == str(e))
    return correct / len(predicted)


def exact_match(predicted: str, expected: str) -> float:
    """Exact string match (1.0 or 0.0)."""
    return 1.0 if str(predicted).strip() == str(expected).strip() else 0.0


def rouge(predicted: str, expected: str) -> Dict[str, float]:
    """Compute ROUGE scores. Requires rouge-score package."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge-score is required for ROUGE metrics. "
            "Install with: pip install promptify[eval]"
        )
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(str(expected), str(predicted))
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


METRIC_REGISTRY: Dict[str, Any] = {
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "accuracy": accuracy,
    "exact_match": exact_match,
    "rouge": rouge,
}
