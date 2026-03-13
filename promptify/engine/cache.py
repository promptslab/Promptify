"""Simple prompt-level caching for LLM responses."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from promptify.core.config import CacheConfig


class PromptCache:
    """LRU cache for deduplicating identical prompt calls."""

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, Any] = OrderedDict()

    @staticmethod
    def _make_key(messages: List[Dict[str, str]], model: str, **kwargs: Any) -> str:
        raw = json.dumps({"messages": messages, "model": model, **kwargs}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, messages: List[Dict[str, str]], model: str, **kwargs: Any) -> Optional[Any]:
        if not self.config.enabled:
            return None
        key = self._make_key(messages, model, **kwargs)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, messages: List[Dict[str, str]], model: str, value: Any, **kwargs: Any) -> None:
        if not self.config.enabled:
            return
        key = self._make_key(messages, model, **kwargs)
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.config.maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()
