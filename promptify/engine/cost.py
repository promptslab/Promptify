"""Cost tracking for LLM usage."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class _CostAccumulator:
    total_cost: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, cost: float, usage: Dict[str, int]) -> None:
        with self._lock:
            self.total_cost += cost
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
            self.call_count += 1

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_cost": round(self.total_cost, 6),
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "call_count": self.call_count,
            }

    def reset(self) -> None:
        with self._lock:
            self.total_cost = 0.0
            self.total_tokens = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.call_count = 0


_accumulator = _CostAccumulator()


def track_cost(cost: float, usage: Dict[str, int]) -> None:
    """Record cost and usage from a single LLM call."""
    _accumulator.add(cost, usage)


def get_cost_summary() -> Dict[str, Any]:
    """Return aggregated cost statistics for the session."""
    return _accumulator.summary()


def reset_cost() -> None:
    """Reset cost tracking."""
    _accumulator.reset()
