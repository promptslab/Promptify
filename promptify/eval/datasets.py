"""Dataset loading helpers for evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union


def load_dataset(source: Union[str, Path, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Load evaluation dataset from JSON file, CSV file, or list of dicts.

    Each item must have at minimum "input" and "expected" keys.

    Parameters
    ----------
    source : str, Path, or list of dict
        Path to a JSON/CSV file, or a list of dicts directly.

    Returns
    -------
    list of dict
        Standardized dataset items with "input" and "expected" keys.
    """
    if isinstance(source, list):
        return _validate(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return _validate(data)
        raise ValueError("JSON dataset must be a list of objects")

    if suffix == ".csv":
        return _load_csv(path)

    raise ValueError(f"Unsupported file format: {suffix}. Use .json or .csv")


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    """Load dataset from CSV. Expects 'input' and 'expected' columns."""
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item: Dict[str, Any] = {"input": row.get("input", "")}
            expected = row.get("expected", "")
            # Try to parse expected as JSON
            try:
                item["expected"] = json.loads(expected)
            except (json.JSONDecodeError, ValueError):
                item["expected"] = expected
            # Pass through any extra columns
            for k, v in row.items():
                if k not in ("input", "expected"):
                    item[k] = v
            items.append(item)
    return items


def _validate(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate that each item has required keys."""
    for i, item in enumerate(data):
        if "input" not in item:
            raise ValueError(f"Dataset item {i} missing 'input' key")
        if "expected" not in item:
            raise ValueError(f"Dataset item {i} missing 'expected' key")
    return data
