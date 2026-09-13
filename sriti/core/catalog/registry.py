"""
Catalog loader — loads config/models.yaml once at import time.

Provides query functions for model lookup, filtering, and cost retrieval.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CATALOG_PATH = Path(__file__).parent.parent.parent / "config" / "models.yaml"


@dataclass(frozen=True)
class ModelInfo:
    """Immutable descriptor for a single model in the catalog."""

    model_id: str
    provider: str
    display_name: str
    context_window: int
    max_output_tokens: int
    cost_per_1m_input: float
    cost_per_1m_output: float
    default_tier: int
    status: str  # "active", "preview", or "stub"

    # ── Capability flags ─────────────────────────────────────────────────
    supports_streaming: bool = True
    supports_system_message: bool = True
    supports_tool_use: bool = False
    supports_parallel_tool_calls: bool = False
    supports_json_mode: bool = False
    supports_vision: bool = False
    supports_video: bool = False
    supports_thinking: bool = False

    # ── Prompt caching costs (None = not supported) ──────────────────────
    cache_write_cost_per_1m: float | None = None
    cache_read_cost_per_1m: float | None = None

    # ── Routing hints ────────────────────────────────────────────────────
    typical_latency_class: str = "medium"  # fast | medium | slow
    strengths: tuple[str, ...] = ()        # semantic tags for routing


def _load_catalog() -> dict[str, ModelInfo]:
    """Load models.yaml into a dict keyed by model_id."""
    if not _CATALOG_PATH.exists():
        logger.warning("models.yaml not found at %s — catalog empty", _CATALOG_PATH)
        return {}

    with open(_CATALOG_PATH) as f:
        data = yaml.safe_load(f) or {}

    catalog: dict[str, ModelInfo] = {}
    for model_id, entry in data.get("models", {}).items():
        catalog[model_id] = ModelInfo(
            model_id=model_id,
            provider=entry["provider"],
            display_name=entry["display_name"],
            context_window=entry["context_window"],
            max_output_tokens=entry.get("max_output_tokens", 4096),
            cost_per_1m_input=float(entry["cost_per_1m_input"]),
            cost_per_1m_output=float(entry["cost_per_1m_output"]),
            default_tier=int(entry["default_tier"]),
            status=entry.get("status", "active"),
            supports_streaming=entry.get("supports_streaming", True),
            supports_system_message=entry.get("supports_system_message", True),
            supports_tool_use=entry.get("supports_tool_use", False),
            supports_parallel_tool_calls=entry.get("supports_parallel_tool_calls", False),
            supports_json_mode=entry.get("supports_json_mode", False),
            supports_vision=entry.get("supports_vision", False),
            supports_video=entry.get("supports_video", False),
            supports_thinking=entry.get("supports_thinking", False),
            cache_write_cost_per_1m=entry.get("cache_write_cost_per_1m"),
            cache_read_cost_per_1m=entry.get("cache_read_cost_per_1m"),
            typical_latency_class=entry.get("typical_latency_class", "medium"),
            strengths=tuple(entry.get("strengths", ())),
        )

    logger.info("Loaded %d models from catalog (%d active, %d preview, %d stub)",
                len(catalog),
                sum(1 for m in catalog.values() if m.status == "active"),
                sum(1 for m in catalog.values() if m.status == "preview"),
                sum(1 for m in catalog.values() if m.status == "stub"))
    return catalog


_catalog: dict[str, ModelInfo] = _load_catalog()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model(model_id: str) -> ModelInfo | None:
    """Return a ModelInfo by its litellm model string, or None if not found."""
    return _catalog.get(model_id)


def get_all_models(status: str = "active") -> list[ModelInfo]:
    """Return all models with a given status. Default: active only."""
    return [m for m in _catalog.values() if m.status == status]


def get_models_by_provider(provider: str) -> list[ModelInfo]:
    """Return all active models for a given provider slug."""
    return [m for m in _catalog.values()
            if m.provider == provider and m.status == "active"]


def get_models_for_tier(
    tier_num: int,
    providers: tuple[str, ...] = (),
) -> list[ModelInfo]:
    """
    Return active models for a given default_tier.

    If providers is non-empty, filter to only those providers.
    """
    results = [m for m in _catalog.values()
               if m.default_tier == tier_num and m.status == "active"]
    if providers:
        results = [m for m in results if m.provider in providers]
    return results


def get_model_cost(model_id: str) -> tuple[float, float]:
    """
    Return (cost_per_1m_input, cost_per_1m_output) for a model.

    Returns (0.0, 0.0) if the model is not in the catalog.
    """
    model = _catalog.get(model_id)
    if model is None:
        return (0.0, 0.0)
    return (model.cost_per_1m_input, model.cost_per_1m_output)


def get_agentic_models(
    tier_num: int | None = None,
    providers: tuple[str, ...] = (),
) -> list[ModelInfo]:
    """
    Return active models suitable for agentic workflows.

    Requires: supports_tool_use=True.
    Optionally filter by tier and providers.
    """
    results = [m for m in _catalog.values()
               if m.status == "active" and m.supports_tool_use]
    if tier_num is not None:
        results = [m for m in results if m.default_tier == tier_num]
    if providers:
        results = [m for m in results if m.provider in providers]
    return results


def filter_models(
    models: list[ModelInfo] | None = None,
    *,
    min_context_window: int | None = None,
    max_cost_per_1m_input: float | None = None,
    min_output_tokens: int | None = None,
    requires_tool_use: bool = False,
    requires_vision: bool = False,
    requires_video: bool = False,
    requires_thinking: bool = False,
    requires_json_mode: bool = False,
    strengths: list[str] | None = None,
    exclude_models: list[str] | None = None,
) -> list[ModelInfo]:
    """
    Filter a list of ModelInfo by various criteria.

    If models is None, starts from all active models.
    """
    result = models if models is not None else get_all_models()

    if min_context_window is not None:
        result = [m for m in result if m.context_window >= min_context_window]

    if max_cost_per_1m_input is not None:
        result = [m for m in result if m.cost_per_1m_input <= max_cost_per_1m_input]

    if min_output_tokens is not None:
        result = [m for m in result if m.max_output_tokens >= min_output_tokens]

    if requires_tool_use:
        result = [m for m in result if m.supports_tool_use]

    if requires_vision:
        result = [m for m in result if m.supports_vision]

    if requires_video:
        result = [m for m in result if m.supports_video]

    if requires_thinking:
        result = [m for m in result if m.supports_thinking]

    if requires_json_mode:
        result = [m for m in result if m.supports_json_mode]

    if strengths:
        strength_set = set(strengths)
        result = [m for m in result if strength_set.issubset(set(m.strengths))]

    if exclude_models:
        exclude_set = set(exclude_models)
        result = [m for m in result if m.model_id not in exclude_set]

    return result
