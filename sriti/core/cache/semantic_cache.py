"""Semantic cache with task-aware similarity thresholds and optional
AES-GCM encryption at rest.

Public API: SemanticCache.warm_up/get/set/schedule_set, init_cache,
get_cache. KNN lookup runs on sriti.core.local_vector_store.LocalVectorStore
(brute-force numpy cosine similarity) instead of Redis Stack's RediSearch
HNSW index —
see local_vector_store.py's module docstring for why. No `tenant_id`
multi-tenancy concept for a per-box deployment, but the parameter is kept
(always "default" in practice) so this module's call sites stay unchanged
from the original — cheaper than reworking every caller's signature for a
distinction that costs nothing to leave in place.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from redis.asyncio import Redis

from sriti.core.cache.encryption import decrypt as _decrypt_content, encrypt as _encrypt_content
from sriti.core.cascade import classifier
from sriti.core.cascade.constants import (
    ERROR_AS_CONTENT_PATTERNS as _ERROR_PATTERNS,
    HEDGING_PATTERNS as _HEDGING_PATTERNS,
)
from sriti.core.cascade.multimodal import content_to_str
from sriti.core.local_vector_store import LocalVectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task types that should never be cached — variable output for similar inputs
# ---------------------------------------------------------------------------
_UNCACHEABLE_TASKS = frozenset({
    "reasoning", "math",
    "data_analysis", "document_review", "tool_use",
})

# Research-backed per-task-type defaults (arXiv:2510.26835, arXiv:2411.05276)
# Overridden at startup from policy.yaml cache.task_thresholds
_DEFAULT_THRESHOLDS: dict[str, float] = {
    "conversation":     0.75,
    "customer_support": 0.78,
    "creative":         0.85,
    "qa":               0.90,
    "rewriting":        0.90,
    "code":             0.92,
    "classification":   0.92,
    "translation":      0.93,
    "rag_retrieval":    0.95,
    "summarization":            0.96,
    "instruction_following":    0.88,
}
_FALLBACK_THRESHOLD = 0.90  # used for any task type not in the map

_KEY_PREFIX = "sriti:cache:"


# ---------------------------------------------------------------------------
# Return type for a cache hit
# ---------------------------------------------------------------------------

@dataclass
class CacheHit:
    content: str
    model_used: str
    model_tier: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    finish_reason: str | None
    similarity_score: float
    task_type: str


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

class SemanticCache:
    """Single-business semantic cache backed by LocalVectorStore (see
    module docstring). Thread-safety: all public methods are async and
    safe to call concurrently."""

    def __init__(
        self,
        redis_client: Redis,
        thresholds: dict[str, float],
        ttl_seconds: int,
        agent_cacheable_steps: frozenset[str] = frozenset(),
    ) -> None:
        self._redis = redis_client
        self._thresholds = thresholds
        self._ttl = ttl_seconds
        self._agent_cacheable_steps = agent_cacheable_steps
        self._enabled = False
        self._store = LocalVectorStore(redis_client, key_prefix=_KEY_PREFIX)
        # Track fire-and-forget tasks to prevent GC before completion
        self._bg_tasks: set[asyncio.Task] = set()

    async def warm_up(self) -> None:
        """Load the local vector index from Valkey. Called once from
        startup. Fail-open (LocalVectorStore.load() already is)."""
        await self._store.load()
        self._enabled = True
        logger.info(
            "Semantic cache: ready (%d records loaded, ttl=%ds).", len(self._store), self._ttl
        )

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    async def get(
        self,
        messages: list[dict],
        task_type: str,
        tenant_id: str = "default",
        similarity_offset: float = 0.0,
        tool_names: list[str] | None = None,
        step_type: str | None = None,
    ) -> CacheHit | None:
        """
        Look up a cached response for the given messages and task type.

        Returns CacheHit if a response is found above the per-task threshold.
        Returns None on cache miss or any error (fail-open).

        The embedding computation (ONNX/CPU) runs in a thread pool to avoid
        blocking the event loop.
        """
        if not self._enabled:
            return None
        if task_type in _UNCACHEABLE_TASKS and step_type not in self._agent_cacheable_steps:
            return None

        model = classifier.get_model()
        if model is None:
            return None

        cache_text = _build_cache_text(messages)
        if not cache_text.strip():
            return None

        conv_fp = _build_conv_fingerprint(messages, tool_names=tool_names)
        threshold = self._thresholds.get(task_type, _FALLBACK_THRESHOLD)
        # Apply similarity offset: negative = more hits (low_latency), positive = stricter (throughput)
        threshold = max(0.0, min(1.0, threshold + similarity_offset))

        try:
            vecs = await asyncio.to_thread(lambda: list(model.embed([cache_text])))
            vec = np.array(vecs[0], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            results = self._store.search(
                vec, k=1, filters={"tenant_id": tenant_id, "conv_fp": conv_fp}
            )
            if not results:
                return None

            similarity, meta = results[0]
            if similarity < threshold:
                logger.debug(
                    "Cache miss: similarity=%.4f < threshold=%.4f task=%s tenant=%s",
                    similarity, threshold, task_type, tenant_id,
                )
                return None

            logger.info(
                "Cache HIT: similarity=%.4f threshold=%.4f task=%s tenant=%s",
                similarity, threshold, task_type, tenant_id,
            )
            return CacheHit(
                content=_decrypt_content(str(meta.get("content", ""))),
                model_used=str(meta.get("model_used", "")),
                model_tier=int(meta.get("model_tier", 0) or 0),
                prompt_tokens=int(meta.get("prompt_tokens", 0) or 0),
                completion_tokens=int(meta.get("completion_tokens", 0) or 0),
                cost_usd=float(meta.get("cost_usd", 0.0) or 0.0),
                finish_reason=meta.get("finish_reason") or None,
                similarity_score=round(similarity, 6),
                task_type=str(meta.get("task_type", task_type) or task_type),
            )

        except Exception as exc:
            logger.warning("Semantic cache GET error (fail-open): %s", exc)
            return None

    async def set(
        self,
        messages: list[dict],
        task_type: str,
        content: str,
        model_used: str,
        model_tier: int,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        finish_reason: str | None,
        tenant_id: str = "default",
        ttl_multiplier: float = 1.0,
        tool_names: list[str] | None = None,
        step_type: str | None = None,
    ) -> None:
        """
        Store a response in the cache (fire-and-forget — never raises).

        Quality gate: only stores responses that are complete and non-hedging.
        Uncacheable task types are silently skipped.
        """
        if not self._enabled:
            return
        allow_uncacheable = step_type in self._agent_cacheable_steps
        if not _is_cacheable(content, finish_reason, task_type, allow_uncacheable=allow_uncacheable):
            return

        model = classifier.get_model()
        if model is None:
            return

        cache_text = _build_cache_text(messages)
        if not cache_text.strip():
            return

        conv_fp = _build_conv_fingerprint(messages, tool_names=tool_names)

        try:
            vecs = await asyncio.to_thread(lambda: list(model.embed([cache_text])))
            vec = np.array(vecs[0], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            metadata = {
                "tenant_id": tenant_id,
                "conv_fp": conv_fp,
                "task_type": task_type,
                "content": _encrypt_content(content),
                "model_used": model_used,
                "model_tier": model_tier,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
                "finish_reason": finish_reason or "stop",
            }
            key = f"{_KEY_PREFIX}{tenant_id}:{uuid.uuid4().hex}"
            effective_ttl = int(self._ttl * ttl_multiplier)
            await self._store.add(key, vec, metadata, ttl_seconds=effective_ttl)
            logger.debug(
                "Cached: key=%s task=%s tenant=%s tokens_out=%d",
                key, task_type, tenant_id, completion_tokens,
            )
        except Exception as exc:
            logger.warning("Semantic cache SET error (ignored): %s", exc)

    def schedule_set(self, **kwargs) -> None:
        """
        Fire-and-forget wrapper around set().
        Called after response is returned to avoid blocking the hot path.
        Stores the task reference to prevent GC before completion.
        """
        task = asyncio.create_task(self.set(**kwargs))
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)


# ---------------------------------------------------------------------------
# Stateless helpers (module-level, no self access needed)
# ---------------------------------------------------------------------------

def _build_cache_text(messages: list[dict]) -> str:
    """Embed text = system_prompt (if any) + last user message."""
    system_prompt = ""
    last_user = ""
    for msg in messages:
        if msg.get("role") == "system" and not system_prompt:
            system_prompt = content_to_str(msg.get("content", ""))
        if msg.get("role") == "user":
            last_user = content_to_str(msg.get("content", ""))
    if not last_user:
        return ""
    if system_prompt:
        return f"{system_prompt}\n{last_user}"
    return last_user


def _build_conv_fingerprint(
    messages: list[dict],
    tool_names: list[str] | None = None,
) -> str:
    """
    sha256(all prior turns excluding the last user message + tool names)[:16].

    Two conversations with different context never share cache entries even
    if the final user message is identical. Single-turn conversations share
    the fingerprint of the empty context (deterministic).
    """
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    prior = messages[:last_user_idx] if last_user_idx > 0 else []
    text = "|".join(f"{m.get('role', '')}:{content_to_str(m.get('content', ''))}" for m in prior)
    if tool_names:
        text += "|tools:" + ",".join(sorted(tool_names))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


_MIN_CACHEABLE_LENGTH = 20  # responses shorter than this are too trivial to cache


def _is_cacheable(
    content: str,
    finish_reason: str | None,
    task_type: str,
    allow_uncacheable: bool = False,
) -> bool:
    """
    Quality gate: True only if the response is complete, substantive,
    non-hedging, and error-free.
    """
    if task_type in _UNCACHEABLE_TASKS and not allow_uncacheable:
        return False
    if finish_reason != "stop":
        return False
    stripped = (content or "").strip()
    if not stripped or len(stripped) < _MIN_CACHEABLE_LENGTH:
        return False
    lower = stripped.lower()
    head = lower[:200]
    if any(phrase in head for phrase in _HEDGING_PATTERNS):
        return False
    if any(phrase in lower for phrase in _ERROR_PATTERNS):
        return False
    return True


# ---------------------------------------------------------------------------
# Module-level singleton + init
# ---------------------------------------------------------------------------

_cache: SemanticCache | None = None


def _load_cache_policy() -> tuple[dict[str, float], int, frozenset[str]]:
    """Read task_thresholds, ttl_seconds, and agent_cacheable_step_types from config/policy.yaml."""
    policy_path = Path(__file__).parents[2] / "config" / "policy.yaml"
    try:
        with open(policy_path) as f:
            data = yaml.safe_load(f)
        cache_cfg = data.get("policy", {}).get("cache", {})
        ttl = int(cache_cfg.get("ttl_seconds", 3600))
        thresholds = cache_cfg.get("task_thresholds", {})
        merged = {**_DEFAULT_THRESHOLDS, **thresholds}
        agent_steps = frozenset(cache_cfg.get("agent_cacheable_step_types", []))
        return merged, ttl, agent_steps
    except Exception as exc:
        logger.warning("Could not load cache policy (%s) — using defaults.", exc)
        return _DEFAULT_THRESHOLDS, 3600, frozenset()


async def init_cache(redis_url: str) -> None:
    """
    Initialise the module-level SemanticCache singleton.
    Must be called once at app startup (after the classifier is warmed up).
    """
    global _cache
    thresholds, ttl_seconds, agent_cacheable_steps = _load_cache_policy()
    redis_client = Redis.from_url(redis_url, decode_responses=True)
    _cache = SemanticCache(redis_client, thresholds, ttl_seconds, agent_cacheable_steps)
    await _cache.warm_up()


def get_cache() -> SemanticCache:
    """Return the initialised cache singleton. Raises if init_cache() not called."""
    if _cache is None:
        raise RuntimeError(
            "Semantic cache not initialised. Call init_cache() at startup."
        )
    return _cache
