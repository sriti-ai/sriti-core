"""Incremental, Valkey-backed reliability-profile aggregation for per-model
outcome tracking (see reliability.py's module docstring for how these
profiles are consumed by the selection engine).

A multi-tenant SaaS would compute per-(model, task_type) reliability
profiles via periodic batch SQL aggregation over a shared outcomes table.
A single-node deployment has one business's outcome volume (dozens to
low-hundreds of calls/day), so an incremental exponential-moving-average
maintained directly in Valkey on every recorded outcome is both simpler
and sufficient — no batch job, no second datastore, no scheduled
aggregation task to run. This is a deliberate cost-first tradeoff: prefer
the simpler approach that fits actual load, and document the assumption
explicitly rather than leaving it as a silent design choice.

Known simplification vs. the original: no true p95 latency (no
distribution is retained, only running averages), and no temporal decay
weighting (recent outcomes aren't weighted more than old ones beyond what
the EMA's alpha already does implicitly). Acceptable for a v1 — revisit if
real per-box outcome volume shows the EMA is too noisy or stale.
"""

from __future__ import annotations

import json
import logging

from redis.asyncio import Redis

logger = logging.getLogger(__name__)

_KEY_PREFIX = "sriti:reliability:"
_EMA_ALPHA = 0.2  # weight on the newest observation; higher = adapts faster, noisier


def _key(model: str, task_type: str) -> str:
    return f"{_KEY_PREFIX}{model}:{task_type}"


async def record_outcome(
    redis_client: Redis,
    model: str,
    task_type: str,
    quality_passed: bool,
    quality_score: float,
    latency_ms: float,
    cost_usd: float,
) -> None:
    """Update the (model, task_type) profile with one new outcome via EMA.
    Fail-open: never raises — reliability tracking must not affect the
    hot path."""
    key = _key(model, task_type)
    try:
        raw = await redis_client.get(key)
        if raw:
            profile = json.loads(raw)
            n = profile["sample_count"] + 1
            a = _EMA_ALPHA
            profile["success_rate"] = (1 - a) * profile["success_rate"] + a * float(quality_passed)
            profile["avg_quality"] = (1 - a) * profile["avg_quality"] + a * quality_score
            profile["avg_latency_ms"] = (1 - a) * profile["avg_latency_ms"] + a * latency_ms
            profile["avg_cost_usd"] = (1 - a) * profile["avg_cost_usd"] + a * cost_usd
            profile["sample_count"] = n
        else:
            profile = {
                "sample_count": 1,
                "success_rate": float(quality_passed),
                "avg_quality": quality_score,
                "avg_latency_ms": latency_ms,
                "avg_cost_usd": cost_usd,
            }
        await redis_client.set(key, json.dumps(profile))
    except Exception:
        logger.warning("reliability_store: record_outcome failed (fail-open)", exc_info=True)


async def load_all_profiles(redis_client: Redis) -> dict[tuple[str, str], dict]:
    """Scan and return every stored (model, task_type) profile. Fail-open:
    returns {} on error, which degrades callers to cold-start selection —
    the same degradation the original Postgres-backed refresh has on a DB
    error."""
    profiles: dict[tuple[str, str], dict] = {}
    try:
        async for raw_key in redis_client.scan_iter(match=f"{_KEY_PREFIX}*"):
            key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
            raw = await redis_client.get(key)
            if not raw:
                continue
            suffix = key[len(_KEY_PREFIX):]
            model, _, task_type = suffix.rpartition(":")
            profiles[(model, task_type)] = json.loads(raw)
    except Exception:
        logger.warning("reliability_store: load_all_profiles failed (fail-open)", exc_info=True)
    return profiles
