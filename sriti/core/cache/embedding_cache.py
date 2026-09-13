from __future__ import annotations

import hashlib
import json
import logging

import redis.asyncio as aioredis

from sriti.core.settings import settings

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None
_DEFAULT_TTL = 86400  # 24 hours


def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.Redis.from_url(settings.redis_url, decode_responses=False)
    return _redis


def _cache_key(model: str, input_text: str) -> str:
    """SHA-256 hash of (model, input_text) for cache key."""
    data = json.dumps({"model": model, "input": input_text}, sort_keys=True)
    return f"emb:{hashlib.sha256(data.encode()).hexdigest()}"


async def get_cached_embedding(model: str, input_text: str) -> list[float] | None:
    """Lookup cached embedding vector. Returns None on miss or error."""
    try:
        r = _get_redis()
        key = _cache_key(model, input_text)
        data = await r.get(key)
        if data is None:
            return None
        return json.loads(data)
    except Exception:
        logger.debug("Embedding cache lookup failed (fail-open)")
        return None


async def store_embedding(
    model: str,
    input_text: str,
    embedding: list[float],
    ttl: int = _DEFAULT_TTL,
) -> None:
    """Store embedding vector in cache. Fire-and-forget, fail-open."""
    try:
        r = _get_redis()
        key = _cache_key(model, input_text)
        await r.set(key, json.dumps(embedding), ex=ttl)
    except Exception:
        logger.debug("Embedding cache store failed (fail-open)")


async def get_cached_embeddings_batch(
    model: str,
    inputs: list[str],
) -> dict[int, list[float]]:
    """Batch lookup. Returns {index: embedding} for cache hits only."""
    results = {}
    try:
        r = _get_redis()
        keys = [_cache_key(model, text) for text in inputs]
        values = await r.mget(keys)
        for i, val in enumerate(values):
            if val is not None:
                results[i] = json.loads(val)
    except Exception:
        logger.debug("Embedding cache batch lookup failed (fail-open)")
    return results


async def store_embeddings_batch(
    model: str,
    inputs: list[str],
    embeddings: list[list[float]],
    ttl: int = _DEFAULT_TTL,
) -> None:
    """Batch store. Fire-and-forget, fail-open."""
    try:
        r = _get_redis()
        pipe = r.pipeline()
        for text, emb in zip(inputs, embeddings):
            key = _cache_key(model, text)
            pipe.set(key, json.dumps(emb), ex=ttl)
        await pipe.execute()
    except Exception:
        logger.debug("Embedding cache batch store failed (fail-open)")
