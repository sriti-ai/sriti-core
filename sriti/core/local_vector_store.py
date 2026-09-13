"""Local, in-process vector store — replaces Redis Stack's RediSearch
module (`FT.CREATE`/`FT.SEARCH`), used by Sriti's
cascade/case_memory.py and cache/semantic_cache.py for HNSW KNN lookups.

RediSearch is a Redis Stack module — not part of vanilla Redis, and not
part of Valkey (an open-source, BSD-licensed Redis replacement). RediSearch
is itself RSALv2-licensed.

At single-node data volumes (hundreds to low-thousands of
case-memory / semantic-cache records, rather than tens of millions),
numpy cosine similarity is fast, lightweight, and eliminates external
module dependencies.

Records persist to Redis/Valkey as plain JSON strings via SET/GET (no RedisJSON
module needed either) so the index survives a process restart; `load()`
rebuilds the in-memory numpy array from Redis/Valkey on startup. Both
case_memory and semantic_cache were already designed fail-open and
tolerant of losing their index on restart ("cache is volatile; cold restart = organic rebuild").
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import numpy as np
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


@dataclass
class _Record:
    key: str
    vector: np.ndarray
    metadata: dict
    expires_at: float | None = None


class LocalVectorStore:
    """One instance per logical index (case memory, semantic cache) — pass
    a distinct key_prefix for each so their Redis/Valkey keys and in-memory
    indexes never collide."""

    def __init__(self, redis_client: Redis, key_prefix: str):
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._records: list[_Record] = []

    async def load(self) -> None:
        """Rebuild the in-memory index from Redis/Valkey. Fail-open: an error
        here leaves the store empty rather than raising — matches the
        original "cold restart = organic rebuild" behavior."""
        records: list[_Record] = []
        try:
            async for raw_key in self._redis.scan_iter(match=f"{self._key_prefix}*"):
                key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
                raw = await self._redis.get(key)
                if raw is None:
                    continue
                ttl = await self._redis.ttl(key)  # -1 = no expiry, -2 = gone (race)
                expires_at = time.time() + ttl if ttl and ttl > 0 else None
                doc = json.loads(raw)
                vector = np.array(doc.pop("_vector"), dtype=np.float32)
                records.append(_Record(key=key, vector=vector, metadata=doc, expires_at=expires_at))
        except Exception:
            logger.warning(
                "LocalVectorStore(%s): load failed (fail-open, starting empty)",
                self._key_prefix,
                exc_info=True,
            )
            records = []
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    async def add(
        self,
        key: str,
        vector: np.ndarray,
        metadata: dict,
        ttl_seconds: int | None = None,
    ) -> None:
        """Add a record to the in-memory index and persist it to Redis/Valkey.
        Fail-open on the Redis/Valkey write — the in-memory index still gets the
        record either way, matching the original's fire-and-forget writes."""
        vec = vector.astype(np.float32)
        expires_at = time.time() + ttl_seconds if ttl_seconds is not None else None
        self._records.append(_Record(key=key, vector=vec, metadata=metadata, expires_at=expires_at))

        try:
            payload = dict(metadata)
            payload["_vector"] = vec.tolist()
            raw = json.dumps(payload)
            if ttl_seconds is not None:
                await self._redis.set(key, raw, ex=ttl_seconds)
            else:
                await self._redis.set(key, raw)
        except Exception:
            logger.warning(
                "LocalVectorStore(%s): persist failed (fail-open, kept in memory only)",
                self._key_prefix,
                exc_info=True,
            )

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[tuple[float, dict]]:
        """Brute-force cosine similarity — vectors are expected pre-
        normalized on insert, so this is a plain dot product. Returns
        (similarity, metadata) pairs, highest similarity first. `filters`
        does an exact-match pre-filter on metadata fields (the equivalent
        of RediSearch TAG filters in the original)."""
        now = time.time()
        candidates = [r for r in self._records if r.expires_at is None or r.expires_at > now]
        if filters:
            candidates = [
                r for r in candidates if all(r.metadata.get(k) == v for k, v in filters.items())
            ]
        if not candidates:
            return []

        query = query_vector.astype(np.float32)
        scored = [(float(np.dot(query, r.vector)), r.metadata) for r in candidates]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:k]
