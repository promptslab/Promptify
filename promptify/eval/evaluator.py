"""Evaluator — runs a task over a dataset and computes metrics."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel

from promptify.core.exceptions import EvaluationError
from promptify.eval.metrics import METRIC_REGISTRY

logger = logging.getLogger("promptify")


def _extract_comparable(result: Any) -> Any:
    """Extract comparable data from a Pydantic model or raw value."""
    if isinstance(result, BaseModel):
        return result.model_dump()
    return result


def _compare_for_metric(
    metric_name: str,
    predicted: Any,
    expected: Any,
) -> float:
    """Run a single metric comparison between predicted and expected."""
    metric_fn = METRIC_REGISTRY.get(metric_name)
    if metric_fn is None:
        raise EvaluationError(f"Unknown metric: {metric_name}")

    pred = _extract_comparable(predicted)
    exp = _extract_comparable(expected)

    if metric_name in ("precision", "recall", "f1"):
        # These expect lists
        if isinstance(pred, dict):
            # Flatten dict values into a list for comparison
            pred_list = []
            for v in pred.values():
                if isinstance(v, list):
                    pred_list.extend(str(item) for item in v)
                else:
                    pred_list.append(str(v))
            pred = pred_list
        if isinstance(exp, dict):
            exp_list = []
            for v in exp.values():
                if isinstance(v, list):
                    exp_list.extend(str(item) for item in v)
                else:
                    exp_list.append(str(v))
            exp = exp_list
        if not isinstance(pred, list):
            pred = [pred]
        if not isinstance(exp, list):
            exp = [exp]
        return metric_fn(pred, exp)

    if metric_name == "accuracy":
        if not isinstance(pred, list):
            pred = [pred]
        if not isinstance(exp, list):
            exp = [exp]
        return metric_fn(pred, exp)

    if metric_name == "exact_match":
        return metric_fn(str(pred), str(exp))

    if metric_name == "rouge":
        scores = metric_fn(str(pred), str(exp))
        return scores.get("rougeL", 0.0)

    return metric_fn(pred, exp)


def evaluate(
    task: Any,
    dataset: List[Dict[str, Any]],
    metrics: List[str],
    max_samples: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, float]:
    """Evaluate a task on a dataset.

    Parameters
    ----------
    task : BaseTask
        The task to evaluate.
    dataset : list of dict
        Each item has "input" (str) and "expected" (BaseModel or dict).
        For QA tasks, items may also have "question".
    metrics : list of str
        Metric names from METRIC_REGISTRY.
    max_samples : int, optional
        Limit the number of samples to evaluate.
    progress_callback : callable, optional
        Called with (current_index, total) after each sample.

    Returns
    -------
    dict of str to float
        Aggregated metric scores.
    """
    if not dataset:
        raise EvaluationError("Dataset is empty")

    samples = dataset[:max_samples] if max_samples else dataset
    total = len(samples)

    scores: Dict[str, List[float]] = {m: [] for m in metrics}

    for i, sample in enumerate(samples):
        text_input = sample.get("input", "")
        expected = sample.get("expected")
        extra_kwargs = {k: v for k, v in sample.items() if k not in ("input", "expected")}

        try:
            predicted = task(text_input, **extra_kwargs)
        except Exception as exc:
            logger.warning("Evaluation failed on sample %d: %s", i, exc)
            for m in metrics:
                scores[m].append(0.0)
            continue

        for m in metrics:
            try:
                score = _compare_for_metric(m, predicted, expected)
                scores[m].append(score)
            except Exception as exc:
                logger.warning("Metric %s failed on sample %d: %s", m, i, exc)
                scores[m].append(0.0)

        if progress_callback:
            progress_callback(i + 1, total)

    return {m: round(sum(s) / len(s), 4) if s else 0.0 for m, s in scores.items()}
