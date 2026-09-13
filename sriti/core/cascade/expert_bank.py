"""
Expert Bank — step-type affinity scores for model selection (MEXA-inspired).

Per-model, per-step_type affinity in [0.0, 1.0]. Seeded from model catalog
heuristics (strengths + latency_class), updated from production outcomes.

All operations are fail-open. Zero hot-path latency — affinities are
pre-computed in memory and refreshed by background aggregation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from sriti.core.catalog.registry import ModelInfo, get_all_models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic seeding rules — derived from ModelInfo.strengths + latency_class
# ---------------------------------------------------------------------------

_STRENGTH_BOOSTS: dict[str, dict[str, float]] = {
    "reasoning":      {"planning": 0.9, "verification": 0.8, "synthesis": 0.7},
    "code":           {"tool_use": 0.9, "generation": 0.8},
    "multilingual":   {"translation": 0.9},
    "summarization":  {"synthesis": 0.8},
    "extraction":     {"retrieval": 0.7},
    "math":           {"verification": 0.8, "planning": 0.7},
    "creative":       {"generation": 0.9},
    "agentic":        {"planning": 0.8, "tool_use": 0.8},
}

_LATENCY_BOOSTS: dict[str, dict[str, float]] = {
    "fast":   {"retrieval": 0.9, "generation": 0.7},
    "slow":   {"planning": 0.8, "verification": 0.7},
    "medium": {},
}

# All recognised step types
STEP_TYPES: tuple[str, ...] = (
    "planning", "retrieval", "synthesis", "verification",
    "tool_use", "generation", "translation",
)

_DEFAULT_AFFINITY = 0.5  # neutral — used for unknown step types or models

# ---------------------------------------------------------------------------
# Policy config loader
# ---------------------------------------------------------------------------

_POLICY_PATH = Path(__file__).parent.parent.parent / "config" / "policy.yaml"


def _load_expert_bank_config() -> dict[str, Any]:
    """Load expert_bank config from policy.yaml. Returns defaults on failure."""
    defaults = {"enabled": True, "weight": 0.10, "update_interval_s": 600, "blend_ratio": 0.7}
    try:
        with open(_POLICY_PATH) as f:
            data = yaml.safe_load(f)
        cfg = data.get("policy", {}).get("reliability", {}).get("expert_bank", {})
        return {**defaults, **cfg}
    except Exception:
        logger.warning("Could not load expert_bank config — using defaults")
        return defaults


_config = _load_expert_bank_config()

# ---------------------------------------------------------------------------
# In-memory affinity store
# ---------------------------------------------------------------------------

# Heuristic-seeded affinities (computed once at startup, never mutated by production data)
_heuristic_affinities: dict[str, dict[str, float]] = {}

# Tenant-scoped affinities: {tenant_id: {model_id: {step_type: float}}}
_tenant_affinities: dict[str, dict[str, dict[str, float]]] = {}


def _seed_heuristic(model: ModelInfo) -> dict[str, float]:
    """Compute initial affinities for a model from its catalog metadata."""
    scores: dict[str, float] = {st: _DEFAULT_AFFINITY for st in STEP_TYPES}

    # Apply strength-based boosts
    for strength in model.strengths:
        boosts = _STRENGTH_BOOSTS.get(strength, {})
        for step_type, boost in boosts.items():
            if step_type in scores:
                scores[step_type] = max(scores[step_type], boost)

    # Apply latency-based boosts
    latency_boosts = _LATENCY_BOOSTS.get(model.typical_latency_class, {})
    for step_type, boost in latency_boosts.items():
        if step_type in scores:
            scores[step_type] = max(scores[step_type], boost)

    return scores


def seed_all() -> None:
    """Seed heuristic affinities for all active models in the catalog. Call at startup."""
    global _heuristic_affinities
    models = get_all_models(status="active")
    new_affinities: dict[str, dict[str, float]] = {}
    for model in models:
        new_affinities[model.model_id] = _seed_heuristic(model)
    _heuristic_affinities = new_affinities
    # Note: _tenant_affinities intentionally NOT reset here — learned production
    # data survives app restarts until next background aggregation refreshes it.
    logger.info("Expert bank: seeded heuristic affinities for %d models", len(new_affinities))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_affinity(model_id: str, step_type: str | None, tenant_id: str = "default") -> float:
    """
    Return the affinity score for a model + step_type pair, scoped by tenant.

    Lookup order:
    1. Tenant-specific affinity (from production data)
    2. Heuristic affinity (from model catalog)
    3. Default (0.5)

    When step_type is None (standalone request without workflow context),
    returns DEFAULT_AFFINITY (0.5) — constant across all models, cancels out.
    """
    if step_type is None:
        return _DEFAULT_AFFINITY

    # Try tenant-specific first
    tenant_models = _tenant_affinities.get(tenant_id)
    if tenant_models:
        model_affinities = tenant_models.get(model_id)
        if model_affinities and step_type in model_affinities:
            return model_affinities[step_type]

    # Fall back to heuristic (never another tenant's data)
    heuristic = _heuristic_affinities.get(model_id)
    if heuristic:
        return heuristic.get(step_type, _DEFAULT_AFFINITY)

    return _DEFAULT_AFFINITY


def get_weight() -> float:
    """Return the fitness function weight for affinity from policy config."""
    return float(_config.get("weight", 0.10))


def is_enabled() -> bool:
    """Return whether expert bank is enabled in policy config."""
    return bool(_config.get("enabled", True))


def get_blend_ratio() -> float:
    """Return production/heuristic blend ratio (default 0.7 = 70% production)."""
    return float(_config.get("blend_ratio", 0.7))


# ---------------------------------------------------------------------------
# Production update — called from background aggregation
# ---------------------------------------------------------------------------

def update_from_production(
    stats: dict[tuple[str, str], tuple[float, int]],
    tenant_id: str = "default",
) -> None:
    """
    Update tenant-scoped affinities by blending production success rates with heuristic seeds.

    Args:
        stats: {(model_id, step_type): (weighted_success_rate, sample_count)}
        tenant_id: Tenant whose affinities to update.
    """
    if not is_enabled():
        return

    blend = get_blend_ratio()
    min_samples = 20
    updated = 0

    # Ensure tenant dict exists
    if tenant_id not in _tenant_affinities:
        _tenant_affinities[tenant_id] = {}

    for (model_id, step_type), (success_rate, sample_count) in stats.items():
        heuristic = _seed_heuristic_for_model_step(model_id, step_type)

        if sample_count >= min_samples:
            new_affinity = blend * success_rate + (1.0 - blend) * heuristic
        else:
            data_weight = blend * (sample_count / min_samples)
            new_affinity = data_weight * success_rate + (1.0 - data_weight) * heuristic

        if model_id not in _tenant_affinities[tenant_id]:
            _tenant_affinities[tenant_id][model_id] = {}

        _tenant_affinities[tenant_id][model_id][step_type] = max(0.0, min(1.0, new_affinity))
        updated += 1

    if updated > 0:
        logger.info(
            "Expert bank: updated %d affinities for tenant=%s from production",
            updated, tenant_id,
        )


def _seed_heuristic_for_model_step(model_id: str, step_type: str) -> float:
    """Get the original heuristic-seeded affinity for a single (model, step_type)."""
    from sriti.core.catalog.registry import get_model
    model_info = get_model(model_id)
    if model_info is None:
        return _DEFAULT_AFFINITY
    heuristic_scores = _seed_heuristic(model_info)
    return heuristic_scores.get(step_type, _DEFAULT_AFFINITY)
