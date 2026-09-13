from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
import redis.asyncio as aioredis

from sriti.core.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TierConfig — immutable descriptor for a model tier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierConfig:
    tier_num: int                    # 1, 2, or 3
    model: str                       # resolved model string for current environment
    quality_threshold: float | None  # None = no quality check (tier1/frontier)
    max_latency_ms: int
    name: str                        # "tier1_frontier", "tier2_balanced", "tier3_fast"
    # Complexity-aware model selection within this tier.
    # Tuple of (complexity_level, model_string) sorted ascending by complexity_level.
    # Empty when no complexity_models are configured or env var override is active.
    # complexity_level: 0=simple, 1=moderate, 2=complex
    complexity_models: tuple[tuple[int, str], ...] = ()
    # Additional models available for reliability-based selection (not default routing).
    additional_models: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# YAML loading — singleton, loaded once at import time
# ---------------------------------------------------------------------------

_POLICY_PATH = Path(__file__).parent.parent.parent / "config" / "policy.yaml"


def _load_policy() -> dict[str, Any]:
    with open(_POLICY_PATH) as f:
        data = yaml.safe_load(f)
    return data["policy"]


_policy: dict[str, Any] = _load_policy()


# ---------------------------------------------------------------------------
# Tier resolution — env-aware model selection
# ---------------------------------------------------------------------------

# Typical-latency-class ordering used at tier-resolution time (import) when no
# reliability profiles exist yet. Kept inline here to avoid a circular import
# with core.cascade.reliability — which itself imports from this module.
_LATENCY_CLASS_RANK: dict[str, int] = {"fast": 0, "medium": 1, "slow": 2}


def _cold_start_rank(models: list) -> list:
    """
    Unbiased reliability-first cold-start sort: (typical_latency_class, cost).

    At module import time there are no profiles loaded yet, so policy tier
    resolution uses this cheap deterministic ordering. At request time, the
    full reliability engine in reliability.py handles ranking with live
    profile data.
    """
    def _key(m) -> tuple[int, float]:
        cls = getattr(m, "typical_latency_class", "medium") or "medium"
        return (_LATENCY_CLASS_RANK.get(cls, 1), float(getattr(m, "cost_per_1m_input", 0.0)))
    return sorted(models, key=_key)


def _resolve_tiers() -> list[TierConfig]:
    """
    Build TierConfigs by querying the model catalog.

    For each tier:
    1. If TIERN_MODEL env var is set → use it (single model override)
    2. Otherwise → query catalog and rank candidates reliability-first with
       the inline _cold_start_rank (latency-class, cost). When the legacy
       feature flag `provider_preference_enabled` is true, preserve the old
       Bedrock-first ordering for rollback safety.
    """
    from sriti.core.catalog.registry import get_models_for_tier

    _overrides = {
        1: settings.tier1_model,
        2: settings.tier2_model,
        3: settings.tier3_model,
    }

    # Fallback models when catalog is empty for a tier
    _fallbacks = {1: "gpt-4o-mini", 2: "gpt-4o-mini", 3: "gpt-4o-mini"}

    # Legacy provider preference (feature-flagged for instant rollback).
    # Default off: reliability ranking is the primary sort.
    _pref_enabled = bool(_policy.get("provider_preference_enabled", False))
    _preferred_providers = _policy.get("provider_preference", []) or []

    def _legacy_provider_sort(models: list) -> list:
        """Legacy Bedrock-first sort. Kept only for the feature-flag rollback path."""
        preferred = [m for m in models if m.provider in _preferred_providers]
        others = [m for m in models if m.provider not in _preferred_providers]
        preferred.sort(key=lambda m: m.cost_per_1m_input)
        others.sort(key=lambda m: m.cost_per_1m_input)
        return preferred + others

    tiers = []
    for i, tier_data in enumerate(_policy["tiers"], start=1):
        override = _overrides.get(i) or None  # treat "" as None

        if override:
            # Explicit override — single model, no complexity routing
            model = override
            complexity_models: tuple[tuple[int, str], ...] = ()
            additional_models: tuple[str, ...] = ()
        else:
            # Catalog-driven selection
            candidates = get_models_for_tier(i)
            if not candidates:
                # Fallback if catalog has no models for this tier
                model = _fallbacks.get(i, "gpt-4o-mini")
                complexity_models = ()
                additional_models = ()
            else:
                if _pref_enabled and _preferred_providers:
                    candidates = _legacy_provider_sort(candidates)
                else:
                    # Reliability-first cold-start ranking (latency class → cost).
                    candidates = _cold_start_rank(candidates)

                # Primary model: first in preference order (cheapest preferred provider)
                model = candidates[0].model_id

                # Build complexity_models: split into 3 cost buckets
                # 0=cheapest (preferred provider), 1=mid, 2=most capable (expensive)
                n = len(candidates)
                if n >= 3:
                    complexity_models = (
                        (0, candidates[0].model_id),           # cheapest preferred
                        (1, candidates[n // 2].model_id),      # median
                        (2, candidates[-1].model_id),          # most capable
                    )
                elif n == 2:
                    complexity_models = (
                        (0, candidates[0].model_id),
                        (2, candidates[1].model_id),
                    )
                else:
                    complexity_models = ()

                # Additional models: all remaining candidates for reliability engine
                primary_ids = {cm[1] for cm in complexity_models} | {model}
                additional_models = tuple(
                    m.model_id for m in candidates if m.model_id not in primary_ids
                )

        threshold = tier_data["quality_threshold"]
        tiers.append(TierConfig(
            tier_num=i,
            model=model,
            quality_threshold=float(threshold) if threshold is not None else None,
            max_latency_ms=tier_data.get("max_latency_ms", 30000),
            name=tier_data["name"],
            complexity_models=tuple(complexity_models),
            additional_models=tuple(additional_models),
        ))

    return tiers


# Tier resolution is deferred to the end of this module (see bottom of file)
# so that rank_candidates (imported lazily from sriti.core.cascade.reliability) can
# reach back into this module's getter functions without a circular import.


# ---------------------------------------------------------------------------
# Task routing config
# ---------------------------------------------------------------------------

def _build_task_tier_map() -> dict[str, int]:
    routing = _policy["task_routing"]
    result: dict[str, int] = {}
    for task in routing.get("tier3_tasks", []):
        result[task] = 3
    for task in routing.get("tier2_tasks", []):
        result[task] = 2
    return result


_task_tier_map: dict[str, int] = _build_task_tier_map()
_default_tier_num: int = _policy["task_routing"]["default_tier"]
_quality_check_mode: str = _policy["quality_check"]["mode"]
_frontier_cap: float = _policy["budget"]["frontier_traffic_cap"]

# Reliability engine settings
_reliability_cfg: dict = _policy.get("reliability", {})
_explore_rate: float = _reliability_cfg.get("explore_rate", 0.1)
_min_samples: int = _reliability_cfg.get("min_samples", 20)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_initial_tier(task_type: str) -> TierConfig:
    """
    Return the starting TierConfig for a given task type.
    Falls back to default_tier (2) for "unknown" or unrecognised task types.
    """
    tier_num = _task_tier_map.get(task_type, _default_tier_num)
    return _tier_map[tier_num]


def get_next_tier(current_tier_num: int) -> TierConfig | None:
    """
    Return the next escalation tier, or None if already at tier1 (frontier).
    Cascade direction: 3 → 2 → 1 (cheapest to most expensive).
    """
    return _tier_map.get(current_tier_num - 1)


def get_tier(tier_num: int) -> TierConfig:
    """Return a specific tier by number. Raises KeyError if not found."""
    return _tier_map[tier_num]


def get_quality_check_mode() -> str:
    """Return 'low' or 'medium'."""
    return _quality_check_mode


def get_explore_rate() -> float:
    """Return the exploration rate for reliability engine (epsilon-greedy)."""
    return _explore_rate


def get_min_samples() -> int:
    """Return the minimum sample count before trusting a model's profile."""
    return _min_samples


def is_latency_slo_enforced() -> bool:
    """Return whether tier.max_latency_ms is enforced as a hard SLO on each dispatch."""
    return bool(_policy.get("latency_slo_enforcement", True))


def get_model_for_complexity(tier_num: int, complexity: int) -> str:
    """
    Return the model string to use for a given tier and complexity score.

    complexity: 0=simple, 1=moderate, 2=complex

    If the tier has no complexity_models (env override or local env), returns
    tier.model directly.

    Selection: find the entry with the highest complexity_level that is <=
    the requested complexity. This ensures that:
    - complexity=0 uses the cheapest/simplest model
    - complexity=2 uses the most capable model in the tier
    - complexity=1 selects a middle-tier model when one is defined
    """
    tier = _tier_map[tier_num]
    if not tier.complexity_models:
        return tier.model
    best = tier.model
    for c, m in tier.complexity_models:  # sorted ascending by complexity
        if c <= complexity:
            best = m
    return best


# ---------------------------------------------------------------------------
# Aggregation config
# ---------------------------------------------------------------------------

_aggregation_cfg: dict = _reliability_cfg.get("aggregation", {})


def get_aggregation_config() -> dict:
    """Return the aggregation sub-config from policy.yaml reliability section."""
    return {
        "enabled": _aggregation_cfg.get("enabled", False),
        "max_parallel_models": _aggregation_cfg.get("max_parallel_models", 2),
        "synthesis_method": _aggregation_cfg.get("synthesis_method", "best_pick"),
        "eligible_tiers": _aggregation_cfg.get("eligible_tiers", [2, 3]),
        "min_tier_models": _aggregation_cfg.get("min_tier_models", 2),
    }


# ---------------------------------------------------------------------------
# Redis frontier traffic cap
# ---------------------------------------------------------------------------

async def check_and_record_frontier(
    redis_client: aioredis.Redis,
    tenant_id: str | None = None,
    cap_override: float | None = None,
) -> bool:
    """
    Check whether a frontier (tier1) request is permitted under the traffic cap.

    Returns True if the request is allowed, False if the cap is exceeded.

    Uses minute-bucket counters with 120s TTL:
      sriti:total:{YYYYMMDD_HHMM}  — total requests this minute
      sriti:frontier:{YYYYMMDD_HHMM}  — frontier requests this minute

    The cap check is: (frontier + 1) / (total + 1) <= frontier_cap

    Fail-open: if Redis is unavailable, the request is allowed through.
    The frontier cap is a soft guardrail, not a hard security gate.
    """
    scope = tenant_id or "global"
    cap = cap_override if cap_override is not None else _frontier_cap
    now = datetime.now(tz=timezone.utc)
    bucket = now.strftime("%Y%m%d_%H%M")
    total_key = f"sriti:{scope}:total:{bucket}"
    frontier_key = f"sriti:{scope}:frontier:{bucket}"
    ttl = 120  # seconds

    # Minimum traffic before cap is meaningful. Below this, always allow frontier.
    _MIN_TRAFFIC = 20

    try:
        total_raw, frontier_raw = await redis_client.mget(total_key, frontier_key)
        total = int(total_raw or 0)
        frontier = int(frontier_raw or 0)

        if total < _MIN_TRAFFIC:
            # Not enough data to enforce cap — allow through and count
            pipe = redis_client.pipeline()
            pipe.incr(total_key)
            pipe.expire(total_key, ttl)
            pipe.incr(frontier_key)
            pipe.expire(frontier_key, ttl)
            await pipe.execute()
            return True

        projected_ratio = (frontier + 1) / (total + 1)
        if projected_ratio > cap:
            logger.info(
                "Frontier cap hit: frontier=%d total=%d ratio=%.3f cap=%.3f scope=%s",
                frontier, total, projected_ratio, cap, scope,
            )
            # Increment total only — request happens at tier2, still counts
            pipe = redis_client.pipeline()
            pipe.incr(total_key)
            pipe.expire(total_key, ttl)
            await pipe.execute()
            return False

        # Allowed — increment both counters atomically
        pipe = redis_client.pipeline()
        pipe.incr(total_key)
        pipe.expire(total_key, ttl)
        pipe.incr(frontier_key)
        pipe.expire(frontier_key, ttl)
        await pipe.execute()
        return True

    except Exception:
        logger.warning("Redis unavailable for frontier cap check — allowing frontier (fail-open)")
        return True


# ---------------------------------------------------------------------------
# Deferred tier resolution
# ---------------------------------------------------------------------------
# Runs at the very end of module-import so that _resolve_tiers() can safely
# import from sriti.core.cascade.reliability (which itself imports getters from this
# module at top-level). Any code further up that references _tier_map is only
# invoked at request time, after module import has completed.

_tiers: list[TierConfig] = _resolve_tiers()
_tier_map: dict[int, TierConfig] = {t.tier_num: t for t in _tiers}

for _t in _tiers:
    logger.info(
        "Tier %d (%s): primary=%s complexity_models=%d additional=%d",
        _t.tier_num, _t.name, _t.model,
        len(_t.complexity_models), len(_t.additional_models),
    )
