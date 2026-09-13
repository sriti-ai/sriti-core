"""
Case Memory — Memento-inspired episodic memory for model selection.

Public API: init_case_memory, is_enabled, get_weight, get_k, store,
schedule_store, make_prompt_hash, retrieve, compute_case_bias,
compute_outcome_reward.

Two design choices worth noting:
1. KNN retrieval runs on sriti.core.local_vector_store.LocalVectorStore
   (brute-force numpy cosine similarity) instead of Redis Stack's
   RediSearch HNSW index — see local_vector_store.py's module docstring
   for why.
2. No durable backup write. Redis/Valkey is volatile — a cold restart
   rebuilds case memory organically from new executions. This is
   intentional: case memory is a warm-start optimisation, not a source
   of truth, and the overhead of a second durable store is not warranted.

Privacy: stores prompt_hash + prompt_embedding only — no raw text.
All operations are async and fail-open. Zero hot-path cost for writes
(fire-and-forget). Reads are a brute-force scan over an in-process list —
fast enough at per-box record counts (see local_vector_store.py).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from redis.asyncio import Redis

from sriti.core.local_vector_store import LocalVectorStore

logger = logging.getLogger(__name__)

_KEY_PREFIX = "sriti:cases:"

# ---------------------------------------------------------------------------
# Policy config
# ---------------------------------------------------------------------------

_POLICY_PATH = Path(__file__).parent.parent.parent / "config" / "policy.yaml"


def _load_config() -> dict[str, Any]:
    defaults = {
        "enabled": True, "k": 4, "weight": 0.15,
        "min_reward": 0.5, "max_cases_per_index": 100_000,
    }
    try:
        with open(_POLICY_PATH) as f:
            data = yaml.safe_load(f)
        cfg = data.get("policy", {}).get("reliability", {}).get("case_memory", {})
        return {**defaults, **cfg}
    except Exception:
        logger.warning("Could not load case_memory config — using defaults")
        return defaults


_config = _load_config()

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_store: LocalVectorStore | None = None
_enabled = False
_bg_tasks: set[asyncio.Task] = set()


# ---------------------------------------------------------------------------
# Outcome reward — continuous composite, not binary
# ---------------------------------------------------------------------------

def compute_outcome_reward(
    quality_score: float,
    quality_passed: bool,
    cost_usd: float,
) -> float:
    """
    Composite reward blending quality, pass/fail, and cost efficiency.

    reward = 0.4 * quality_score + 0.3 * passed + 0.3 * (1 - min(cost/0.01, 1))
    """
    cost_efficiency = 1.0 - min(cost_usd / 0.01, 1.0)
    return (
        0.4 * quality_score
        + 0.3 * float(quality_passed)
        + 0.3 * cost_efficiency
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

async def init_case_memory(redis_client: Redis) -> None:
    """
    Initialise the case memory local vector index. Called once at app
    startup. Fail-open: disables case memory on error.
    """
    global _store, _enabled

    if not _config.get("enabled", True):
        logger.info("Case memory: disabled by config")
        return

    _store = LocalVectorStore(redis_client, key_prefix=_KEY_PREFIX)
    await _store.load()
    _enabled = True
    logger.info("Case memory: local vector index ready (%d records loaded).", len(_store))


def is_enabled() -> bool:
    return _enabled


def get_weight() -> float:
    return float(_config.get("weight", 0.15))


def get_k() -> int:
    return int(_config.get("k", 4))


# ---------------------------------------------------------------------------
# Store — fire-and-forget write path
# ---------------------------------------------------------------------------

async def store(
    *,
    tenant_id: str,
    session_id: str | None,
    step_type: str | None,
    task_type: str,
    prompt_hash: str,
    embedding: NDArray[np.float32],
    model_used: str,
    quality_passed: bool,
    quality_score: float,
    cost_usd: float,
    latency_ms: float,
) -> None:
    """
    Store an execution case for fast KNN retrieval. Fail-open: errors are
    logged but never propagated.
    """
    if not _enabled or _store is None:
        return

    outcome_reward = compute_outcome_reward(quality_score, quality_passed, cost_usd)

    try:
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        metadata = {
            "tenant_id": tenant_id,
            "step_type": step_type or "*",
            "task_type": task_type,
            "model_used": model_used,
            "quality_passed": int(quality_passed),
            "outcome_reward": round(outcome_reward, 4),
            "cost_usd": round(cost_usd, 6),
            "latency_ms": round(latency_ms, 1),
        }
        import uuid
        key = f"{_KEY_PREFIX}{tenant_id}:{uuid.uuid4().hex}"
        await _store.add(key, vec, metadata)
    except Exception:
        logger.exception("Case memory: store failed (fail-open)")


def schedule_store(**kwargs: Any) -> None:
    """Fire-and-forget wrapper around store(). Zero hot-path latency."""
    task = asyncio.create_task(store(**kwargs))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


def make_prompt_hash(text: str) -> str:
    """SHA-256 hash of prompt text for dedup (no raw text stored)."""
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Retrieve — KNN lookup for model selection
# ---------------------------------------------------------------------------

async def retrieve(
    user_vec: NDArray[np.float32],
    task_type: str,
    step_type: str | None = None,
    tenant_id: str = "default",
) -> list[dict]:
    """
    Retrieve K nearest cases via local brute-force cosine KNN.

    Returns list of dicts with keys: model_used, outcome_reward, quality_passed,
    cost_usd, latency_ms. Empty list on error or when disabled.
    """
    if not _enabled or _store is None:
        return []

    k = get_k()

    try:
        vec = np.array(user_vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        filters = {"tenant_id": tenant_id, "task_type": task_type}
        if step_type:
            filters["step_type"] = step_type

        results = _store.search(vec, k=k, filters=filters)

        cases = []
        for _score, meta in results:
            cases.append({
                "model_used": meta.get("model_used", ""),
                "outcome_reward": float(meta.get("outcome_reward", 0.0)),
                "quality_passed": bool(meta.get("quality_passed", 0)),
                "cost_usd": float(meta.get("cost_usd", 0.0)),
                "latency_ms": float(meta.get("latency_ms", 0.0)),
            })
        return cases

    except Exception:
        logger.exception("Case memory: retrieve failed (fail-open)")
        return []


def compute_case_bias(cases: list[dict]) -> dict[str, float]:
    """
    Compute per-model bias from retrieved cases.

    Positive outcomes (reward > min_reward) boost the model.
    Negative outcomes penalize it.
    Returns {model_id: float} in roughly [-1.0, +1.0] range.
    """
    if not cases:
        return {}

    min_reward = float(_config.get("min_reward", 0.5))
    k = len(cases)
    bias: dict[str, float] = {}

    for case in cases:
        model = case["model_used"]
        reward = case["outcome_reward"]
        if reward > min_reward:
            bias[model] = bias.get(model, 0.0) + reward / k
        else:
            bias[model] = bias.get(model, 0.0) - (1.0 - reward) / k

    return bias
