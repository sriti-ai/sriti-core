"""
Reliability profiles engine — adaptive model selection via outcome learning.

Selection logic (get_best_model, _cold_start_selection, get_top_models,
rank_candidates) reads the module-level `_profiles`/`_global_profiles`
caches, regardless of what refreshes them.

Design notes:

1. Profile persistence: `_maybe_refresh_profiles()` and outcome recording
   go through reliability_store.py (Valkey, incremental EMA) instead of
   a batch-aggregation database. See reliability_store.py's module
   docstring for the rationale.
2. Single-tenant: this engine assumes a single-tenant deployment (one
   business per instance). `tenant_id` parameters exist but always
   default to "default"; multi-tenant blending is not implemented.
3. No scheduled aggregation job: reliability_store's EMA updates
   incrementally on each outcome, so no background job is needed.

All operations are async and fail-open. Zero added latency on the hot path.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from redis.asyncio import Redis

from sriti.core.cascade import reliability_store
from sriti.core.cascade.policy import (
    TierConfig,
    get_explore_rate,
    get_min_samples,
    get_model_for_complexity,
    get_tier,
)
from sriti.core.cascade.complexity import complexity_to_discrete

logger = logging.getLogger(__name__)

# Minimum success rate — models below this are filtered out
_MIN_SUCCESS_RATE = 0.5

# In-memory cache refresh interval (seconds)
_CACHE_TTL_S = 60.0


@dataclass
class Outcome:
    """Data class for a single request outcome."""
    model: str
    task_type: str
    complexity_score: float
    quality_passed: bool
    quality_score: float
    latency_ms: float
    cost_usd: float
    escalated: bool
    tier_num: int
    step_type: str | None = None
    session_id: str | None = None
    tenant_id: str = "default"


@dataclass
class _Profile:
    """In-memory representation of a reliability profile."""
    tenant_id: str
    model: str
    task_type: str
    sample_count: int
    success_rate: float
    avg_quality: float
    avg_latency_ms: float
    avg_cost_usd: float
    p95_latency_ms: float = 0.0  # not tracked — see reliability_store.py docstring


# Single-tenant profiles: keyed by (tenant_id, model, task_type). tenant_id
# is always "default" on a per-box deployment, kept for signature parity.
_profiles: dict[tuple[str, str, str], _Profile] = {}
# Kept for rank_candidates()'s fallback path; always empty (no tenant
# blending — see module docstring).
_global_profiles: dict[tuple[str, str], _Profile] = {}
_profiles_last_refresh: float = 0.0
_profiles_lock: asyncio.Lock | None = None  # lazily initialized

_redis_client: Redis | None = None


def init_reliability(redis_client: Redis) -> None:
    """Register the Valkey client reliability_store reads/writes through.
    Call once at app startup, alongside init_case_memory."""
    global _redis_client
    _redis_client = redis_client


# ---------------------------------------------------------------------------
# Tier model helpers
# ---------------------------------------------------------------------------

def _get_available_models_for_tier(tier_num: int) -> list[str]:
    """
    Get all available models for a tier from the policy config.
    Returns the base model + all complexity_models.
    """
    tier: TierConfig = get_tier(tier_num)
    models = {tier.model}
    for _, model in tier.complexity_models:
        models.add(model)
    for model in tier.additional_models:
        models.add(model)
    return list(models)


# ---------------------------------------------------------------------------
# Profile refresh
# ---------------------------------------------------------------------------

async def _maybe_refresh_profiles() -> None:
    """Refresh in-memory profiles from Valkey if stale (>60s). Serialized via lock."""
    global _profiles, _profiles_last_refresh, _profiles_lock
    if _redis_client is None:
        return

    now = time.monotonic()
    if now - _profiles_last_refresh < _CACHE_TTL_S:
        return

    if _profiles_lock is None:
        _profiles_lock = asyncio.Lock()

    async with _profiles_lock:
        if time.monotonic() - _profiles_last_refresh < _CACHE_TTL_S:
            return

        try:
            raw_profiles = await reliability_store.load_all_profiles(_redis_client)
            new_profiles: dict[tuple[str, str, str], _Profile] = {}
            for (model, task_type), data in raw_profiles.items():
                new_profiles[("default", model, task_type)] = _Profile(
                    tenant_id="default",
                    model=model,
                    task_type=task_type,
                    sample_count=data["sample_count"],
                    success_rate=data["success_rate"],
                    avg_quality=data["avg_quality"],
                    avg_latency_ms=data["avg_latency_ms"],
                    avg_cost_usd=data["avg_cost_usd"],
                    p95_latency_ms=data["avg_latency_ms"],
                )
            _profiles = new_profiles
            _profiles_last_refresh = now
        except Exception:
            logger.exception("Failed to refresh reliability profiles (fail-open)")


# ---------------------------------------------------------------------------
# Model selection — enhanced with case memory + expert bank
# ---------------------------------------------------------------------------

def _cold_start_selection(
    available_models: list[str],
    task_type: str,
    tier_num: int,
    complexity_score: float,
    tenant_id: str = "default",
) -> str:
    """
    Cold-start model selection — maximise variety until we learn what works.

    Picks the model with the fewest samples across all available models in
    the tier. Ties are broken randomly so traffic spreads evenly.
    """
    if not available_models:
        discrete = complexity_to_discrete(complexity_score)
        return get_model_for_complexity(tier_num, discrete)

    scored: list[tuple[str, int]] = []
    for model in available_models:
        profile = _profiles.get((tenant_id, model, task_type))
        count = profile.sample_count if profile else 0
        scored.append((model, count))

    min_count = min(c for _, c in scored)
    least_sampled = [m for m, c in scored if c == min_count]
    selected = random.choice(least_sampled)

    logger.debug(
        "Cold-start: tier%d task=%s tenant=%s → %s (samples=%d, candidates=%d)",
        tier_num, task_type, tenant_id, selected, min_count, len(available_models),
    )
    return selected


async def get_best_model(
    task_type: str,
    complexity_score: float,
    tier_num: int,
    available_models: list[str],
    *,
    step_type: str | None = None,
    user_vec: NDArray[np.float32] | None = None,
    tenant_id: str = "default",
    prefer_models: list[str] | None = None,
    avoid_models: list[str] | None = None,
    hard_avoid_models: list[str] | None = None,
) -> tuple[str, bool]:
    """
    Select the best model for a (task_type, complexity, tier) combination.

    Returns (model_id, is_exploration):
    - is_exploration=True when the model was selected via epsilon-greedy exploration
    - is_exploration=False when selected via Pareto cost-quality tradeoff

    case_bias (case memory) and affinity (expert bank) are secondary
    signals that break ties; Pareto remains dominant. Falls back to
    cold-start when insufficient data.
    """
    await _maybe_refresh_profiles()

    min_samples = get_min_samples()
    explore_rate = get_explore_rate()

    # ── Exploration: route to under-sampled model (epsilon-greedy) ─────
    if explore_rate > 0 and random.random() < explore_rate and len(available_models) > 1:
        weights = []
        for model in available_models:
            profile = _profiles.get((tenant_id, model, task_type))
            count = profile.sample_count if profile else 0
            weights.append(1.0 / (1.0 + count))
        total = sum(weights)
        weights = [w / total for w in weights]
        selected = random.choices(available_models, weights=weights, k=1)[0]
        sample_count = 0
        profile = _profiles.get((tenant_id, selected, task_type))
        if profile:
            sample_count = profile.sample_count
        logger.debug(
            "Explore: tier%d task=%s tenant=%s → %s (samples=%d)",
            tier_num, task_type, tenant_id, selected, sample_count,
        )
        return selected, True

    # ── Exploit: Pareto selection ───────────────────────────────────────
    candidates: list[_Profile] = []
    for model in available_models:
        key = (tenant_id, model, task_type)
        profile = _profiles.get(key)
        if profile and profile.sample_count >= min_samples:
            candidates.append(profile)

    if not candidates:
        return _cold_start_selection(available_models, task_type, tier_num, complexity_score, tenant_id), False

    candidates = [c for c in candidates if c.success_rate >= _MIN_SUCCESS_RATE]
    if not candidates:
        return _cold_start_selection(available_models, task_type, tier_num, complexity_score, tenant_id), False

    # ── Case bias (async, ~3-5ms) ───────────────────────────────────────
    case_bias: dict[str, float] = {}
    case_weight = 0.0
    try:
        from sriti.core.cascade import case_memory
        if case_memory.is_enabled() and user_vec is not None:
            cases = await case_memory.retrieve(
                user_vec, task_type, step_type=step_type, tenant_id=tenant_id,
            )
            case_bias = case_memory.compute_case_bias(cases)
            case_weight = case_memory.get_weight()
    except Exception:
        logger.debug("Case memory lookup failed (fail-open)", exc_info=True)

    # ── Expert bank affinity ─────────────────────────────────────────────
    affinity_weight = 0.0
    try:
        from sriti.core.cascade import expert_bank
        if expert_bank.is_enabled():
            affinity_weight = expert_bank.get_weight()
    except Exception:
        logger.debug("Expert bank lookup failed (fail-open)", exc_info=True)

    # ── Pareto selection — normalize cost and quality across candidates ─
    qualities = [c.avg_quality for c in candidates]
    costs = [c.avg_cost_usd for c in candidates]

    q_min, q_max = min(qualities), max(qualities)
    c_min, c_max = min(costs), max(costs)
    q_range = q_max - q_min if q_max > q_min else 1.0
    c_range = c_max - c_min if c_max > c_min else 1.0

    quality_weight = 0.3 + 0.6 * complexity_score
    cost_w = 1.0 - quality_weight

    best_model = candidates[0].model
    best_fitness = float("-inf")

    for c in candidates:
        norm_quality = (c.avg_quality - q_min) / q_range
        norm_cost = (c.avg_cost_usd - c_min) / c_range

        pareto = (quality_weight * norm_quality - cost_w * norm_cost) * c.success_rate
        case_term = case_weight * case_bias.get(c.model, 0.0)

        affinity = _DEFAULT_AFFINITY
        try:
            if affinity_weight > 0:
                from sriti.core.cascade import expert_bank
                affinity = expert_bank.get_affinity(c.model, step_type, tenant_id=tenant_id)
        except Exception:
            pass
        affinity_term = affinity_weight * affinity

        memory_term = 0.0
        if prefer_models and c.model in prefer_models:
            memory_term += 0.15
        if avoid_models and c.model in avoid_models:
            memory_term -= 0.20
        if hard_avoid_models and c.model in hard_avoid_models:
            memory_term -= 0.50

        success_bonus = 0.01 * c.success_rate

        fitness = pareto + case_term + affinity_term + memory_term + success_bonus
        if fitness > best_fitness:
            best_fitness = fitness
            best_model = c.model

    return best_model, False


_DEFAULT_AFFINITY = 0.5


# ---------------------------------------------------------------------------
# Top-N model selection (used by aggregation)
# ---------------------------------------------------------------------------

def get_top_models(
    task_type: str,
    complexity: float,
    tier_num: int,
    available_models: list[str],
    n: int = 2,
    *,
    exclude_models: set[str] | None = None,
    step_type: str | None = None,
    user_vec: NDArray[np.float32] | None = None,
    tenant_id: str = "default",
    prefer_models: list[str] | None = None,
    avoid_models: list[str] | None = None,
    hard_avoid_models: list[str] | None = None,
) -> list[tuple[str, float]]:
    """
    Return top N models by Pareto fitness score (synchronous, uses cached profiles).

    Unlike get_best_model, this does NOT do exploration or cold-start fallback.
    Returns empty list if no candidates have sufficient profile data.
    """
    min_samples = get_min_samples()
    exclude = exclude_models or set()

    filtered = [m for m in available_models if m not in exclude]
    if not filtered:
        return []

    candidates: list[_Profile] = []
    for model in filtered:
        key = (tenant_id, model, task_type)
        profile = _profiles.get(key)
        if profile and profile.sample_count >= min_samples and profile.success_rate >= _MIN_SUCCESS_RATE:
            candidates.append(profile)

    if not candidates:
        return [(m, 0.0) for m in filtered[:n]]

    qualities = [c.avg_quality for c in candidates]
    costs = [c.avg_cost_usd for c in candidates]

    q_min, q_max = min(qualities), max(qualities)
    c_min, c_max = min(costs), max(costs)
    q_range = q_max - q_min if q_max > q_min else 1.0
    c_range = c_max - c_min if c_max > c_min else 1.0

    quality_weight = 0.3 + 0.6 * complexity
    cost_w = 1.0 - quality_weight

    scored: list[tuple[str, float]] = []
    for c in candidates:
        norm_quality = (c.avg_quality - q_min) / q_range
        norm_cost = (c.avg_cost_usd - c_min) / c_range
        fitness = (quality_weight * norm_quality - cost_w * norm_cost) * c.success_rate
        fitness += 0.01 * c.success_rate
        if prefer_models and c.model in prefer_models:
            fitness += 0.15
        if avoid_models and c.model in avoid_models:
            fitness -= 0.20
        if hard_avoid_models and c.model in hard_avoid_models:
            fitness -= 0.50
        scored.append((c.model, fitness))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


# ---------------------------------------------------------------------------
# Candidate ranking (used by policy tier resolution — unbiased, no provider bias)
# ---------------------------------------------------------------------------

_LATENCY_CLASS_RANK: dict[str, int] = {"fast": 0, "medium": 1, "slow": 2}


def rank_candidates(candidates: list[Any]) -> list[Any]:
    """
    Rank a list of ModelInfo candidates reliability-first (no provider bias).

    Uses cached reliability profiles when available; otherwise falls back to a
    cold-start sort by (typical_latency_class, cost_per_1m_input ascending).
    Safe to call at module import time — will not touch Valkey.
    Returns a new list sorted best → worst. Never raises.
    """
    if not candidates:
        return []

    min_samples = get_min_samples()

    def _cold_start_key(m: Any) -> tuple[int, float]:
        cls = getattr(m, "typical_latency_class", "medium") or "medium"
        return (_LATENCY_CLASS_RANK.get(cls, 1), float(getattr(m, "cost_per_1m_input", 0.0)))

    if not _profiles:
        return sorted(candidates, key=_cold_start_key)

    def _profile_score(m: Any) -> float | None:
        total_success = 0.0
        total_quality = 0.0
        n = 0
        for (_tid, model, _task), p in _profiles.items():
            if model != m.model_id or p.sample_count < min_samples:
                continue
            total_success += p.success_rate
            total_quality += p.avg_quality
            n += 1
        if n == 0:
            return None
        avg_success = total_success / n
        avg_quality = total_quality / n
        return (0.6 * avg_quality + 0.4 * avg_success) - 1e-6 * float(
            getattr(m, "cost_per_1m_input", 0.0)
        )

    scored: list[tuple[float, tuple[int, float], Any]] = []
    unprofiled: list[Any] = []
    for m in candidates:
        s = _profile_score(m)
        if s is None:
            unprofiled.append(m)
        else:
            scored.append((-s, _cold_start_key(m), m))

    scored.sort(key=lambda x: (x[0], x[1]))
    return [t[2] for t in scored] + sorted(unprofiled, key=_cold_start_key)


# ---------------------------------------------------------------------------
# Outcome recording
# ---------------------------------------------------------------------------

def schedule_record_outcome(outcome: Outcome) -> None:
    """Fire-and-forget: record an outcome into the Valkey-backed profile store."""
    try:
        asyncio.create_task(_record_outcome(outcome))
    except RuntimeError:
        pass  # no running event loop (e.g. tests, scripts)


async def _record_outcome(outcome: Outcome) -> None:
    """Update the (model, task_type) reliability profile. Fail-open."""
    if _redis_client is None:
        return
    try:
        await reliability_store.record_outcome(
            _redis_client,
            model=outcome.model,
            task_type=outcome.task_type,
            quality_passed=outcome.quality_passed,
            quality_score=outcome.quality_score,
            latency_ms=outcome.latency_ms,
            cost_usd=outcome.cost_usd,
        )
    except Exception:
        logger.exception("Failed to record outcome (fail-open)")
