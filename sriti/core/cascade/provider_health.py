"""
Provider-level circuit breaker — rolling p95 latency tracker.

Tracks per-provider latency in minute-bucketed Redis lists. When the current
60-second p95 exceeds 2× the 60-minute baseline p95 (with at least 10 samples
in the current window), the provider is flagged DEGRADED for 120 seconds.

This is a soft signal: `filter_candidates()` deprioritises degraded providers
(moves them to the end of the candidate list) rather than dropping them, so
the cascade can still fall over to Bedrock if every other provider is out.

Fail-open: every Redis exception is swallowed. When Redis is unavailable this
module never blocks a request — it behaves as if all providers are healthy.

Key scheme:
    sriti:provider_health:{provider}:{YYYYMMDD_HHMM}  — per-minute latency list
    sriti:provider_health:{provider}:degraded          — 120s degraded latch
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables (intentionally module-level constants — no new policy.yaml knobs)
# ---------------------------------------------------------------------------

_MIN_CURRENT_SAMPLES = 10     # require this many samples in the 60s window
_CURRENT_WINDOW_MIN = 1       # "last 60 s" — one minute bucket
_BASELINE_WINDOW_MIN = 60     # "last hour" baseline
_DEGRADATION_RATIO = 2.0      # current_p95 > 2× baseline_p95 → degraded
_DEGRADATION_TTL = 120        # seconds the degraded latch is held
_BUCKET_TTL = 3700            # 1h + slack so baseline window is complete

# Fire-and-forget background tasks — strong refs to prevent GC before completion
# (same pattern as core/cache/semantic_cache.py).
_bg_tasks: set[asyncio.Task] = set()


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _bucket_key(provider: str, bucket: str) -> str:
    return f"sriti:provider_health:{provider}:{bucket}"


def _latch_key(provider: str) -> str:
    return f"sriti:provider_health:{provider}:degraded"


def _bucket_ids(now: datetime, minutes: int) -> list[str]:
    return [
        (now - timedelta(minutes=i)).strftime("%Y%m%d_%H%M")
        for i in range(minutes)
    ]


def _extract_provider(model_or_provider: str) -> str:
    """
    Accept either a provider slug ("bedrock") or a model string
    ("bedrock/anthropic.claude-3-5-sonnet-...") and return the provider slug.
    """
    if "/" in model_or_provider:
        return model_or_provider.split("/", 1)[0]
    return model_or_provider


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

async def _record(
    redis_client: aioredis.Redis,
    provider: str,
    latency_ms: float,
    success: bool,
) -> None:
    """Append a latency sample to the current minute bucket. Fail-open."""
    if not provider:
        return
    # Only successful calls count toward the latency profile — failures would
    # skew the baseline downward once providers error out quickly.
    if not success:
        return
    try:
        now = datetime.now(tz=timezone.utc)
        bucket = now.strftime("%Y%m%d_%H%M")
        key = _bucket_key(provider, bucket)
        pipe = redis_client.pipeline()
        pipe.lpush(key, f"{latency_ms:.2f}")
        pipe.expire(key, _BUCKET_TTL)
        await pipe.execute()
    except Exception:
        logger.debug("provider_health._record failed (fail-open)", exc_info=True)


def record(
    redis_client: aioredis.Redis,
    provider_or_model: str,
    latency_ms: float,
    success: bool,
) -> None:
    """
    Fire-and-forget wrapper around _record(). Safe to call from any async
    context; never blocks the hot path.
    """
    provider = _extract_provider(provider_or_model)
    try:
        task = asyncio.create_task(_record(redis_client, provider, latency_ms, success))
    except RuntimeError:
        return  # no running event loop (e.g. sync tests)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

async def _read_samples(
    redis_client: aioredis.Redis,
    provider: str,
    minutes: int,
) -> list[float]:
    """Read all latency samples in the last `minutes` minute-buckets."""
    try:
        now = datetime.now(tz=timezone.utc)
        keys = [_bucket_key(provider, b) for b in _bucket_ids(now, minutes)]
        pipe = redis_client.pipeline()
        for k in keys:
            pipe.lrange(k, 0, -1)
        results = await pipe.execute()
    except Exception:
        logger.debug("provider_health._read_samples failed (fail-open)", exc_info=True)
        return []

    samples: list[float] = []
    for lst in results or []:
        for val in lst or []:
            try:
                samples.append(float(val))
            except (ValueError, TypeError):
                continue
    return samples


def _p95(samples: list[float]) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = max(0, int(round(0.95 * (len(s) - 1))))
    return float(s[idx])


async def is_degraded(redis_client: aioredis.Redis, provider_or_model: str) -> bool:
    """
    Return True if this provider is currently flagged DEGRADED.

    Fail-open: any Redis error returns False (provider considered healthy).
    """
    provider = _extract_provider(provider_or_model)
    if not provider:
        return False

    # Latch: once degraded, stay degraded for _DEGRADATION_TTL seconds.
    try:
        latched = await redis_client.get(_latch_key(provider))
        if latched:
            return True
    except Exception:
        return False

    current = await _read_samples(redis_client, provider, _CURRENT_WINDOW_MIN)
    if len(current) < _MIN_CURRENT_SAMPLES:
        return False
    baseline = await _read_samples(redis_client, provider, _BASELINE_WINDOW_MIN)
    if len(baseline) < _MIN_CURRENT_SAMPLES:
        return False

    curr_p95 = _p95(current)
    base_p95 = _p95(baseline)
    if base_p95 <= 0:
        return False

    degraded = curr_p95 > _DEGRADATION_RATIO * base_p95
    if degraded:
        logger.warning(
            "Provider %s marked DEGRADED: current_p95=%.0fms baseline_p95=%.0fms",
            provider, curr_p95, base_p95,
        )
        try:
            await redis_client.set(_latch_key(provider), "1", ex=_DEGRADATION_TTL)
        except Exception:
            pass
    return degraded


async def filter_candidates(
    redis_client: aioredis.Redis,
    candidates: list[Any],
) -> list[Any]:
    """
    Soft-deprioritise candidates whose provider is currently degraded.

    Accepts either a list of model strings ("groq/llama-...") or anything with
    a `.provider` attribute. Never drops candidates — degraded ones are simply
    moved to the end of the list so the cascade can still fall through to them
    if healthy providers are exhausted.

    Fail-open: any Redis failure returns the input list unchanged.
    """
    if not candidates:
        return candidates

    def _prov(c: Any) -> str:
        if hasattr(c, "provider"):
            return getattr(c, "provider", "") or ""
        if isinstance(c, str):
            return _extract_provider(c)
        return ""

    try:
        providers = {_prov(c) for c in candidates if _prov(c)}
        degraded_set: set[str] = set()
        for p in providers:
            if await is_degraded(redis_client, p):
                degraded_set.add(p)
    except Exception:
        return candidates

    if not degraded_set:
        return candidates

    healthy = [c for c in candidates if _prov(c) not in degraded_set]
    degraded = [c for c in candidates if _prov(c) in degraded_set]
    return healthy + degraded
