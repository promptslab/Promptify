from functools import lru_cache, cached_property

class PromptCache:
    def __init__(self, cache_size: int = 200):
        self.cache_size = cache_size
        self._cache     = lru_cache(maxsize=cache_size)(self._raw_cache)()

    @cached_property
    def cache(self):
        return self._cache

    def get(self, key):
        return self.cache.get(key)

    def add(self, key, value):
        if key not in self.cache:
            self.cache[key] = value

    def clear(self):
        self.cache.clear()

    @staticmethod
    def _raw_cache():
        return {}
