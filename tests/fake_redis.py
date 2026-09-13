"""Minimal in-memory async fake for the small slice of the redis.asyncio
API sriti/'s vendored modules actually use (get/set/ttl/scan_iter). Not a
full redis-py fake — just enough to unit-test local_vector_store.py and
reliability_store.py without a real Valkey/Redis instance running.
"""

from __future__ import annotations

import fnmatch
import time


class FakeRedis:
    def __init__(self):
        self._data: dict[str, str] = {}
        self._expires: dict[str, float] = {}

    def _expired(self, key: str) -> bool:
        exp = self._expires.get(key)
        return exp is not None and exp <= time.time()

    async def get(self, key: str) -> str | None:
        if key not in self._data or self._expired(key):
            self._data.pop(key, None)
            self._expires.pop(key, None)
            return None
        return self._data[key]

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._data[key] = value
        if ex is not None:
            self._expires[key] = time.time() + ex
        else:
            self._expires.pop(key, None)

    async def ttl(self, key: str) -> int:
        if key not in self._data:
            return -2
        exp = self._expires.get(key)
        if exp is None:
            return -1
        remaining = exp - time.time()
        return int(remaining) if remaining > 0 else -2

    async def scan_iter(self, match: str = "*"):
        for key in list(self._data.keys()):
            if self._expired(key):
                continue
            if fnmatch.fnmatch(key, match):
                yield key
