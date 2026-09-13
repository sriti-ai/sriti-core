"""
Multi-expert aggregation — cheaper alternative to tier escalation.

When a model fails quality checks, instead of escalating to the next (more expensive)
tier, dispatch the same request to 2+ additional models *within the same tier* in
parallel, quality-check each, and pick the best. Only escalate if aggregation also fails.

Economics: 2 extra tier2 calls at ~$0.20/M tokens = $0.40 vs 1 frontier call at ~$2.50/M.
Parallel dispatch means latency = max(individual), not sum.

All operations are fail-open — aggregation failure falls through to normal escalation.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from sriti.core.cascade.reliability import Outcome, _get_available_models_for_tier
from sriti.core.execution.litellm_client import LLMResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CandidateOutcome:
    """Result from a single aggregation candidate."""
    model: str
    response: LLMResponse | None
    quality_passed: bool
    error: str | None = None


@dataclass
class AggregationResult:
    """Returned when aggregation succeeds (at least one candidate passes)."""
    response: LLMResponse
    model_used: str
    candidates_tried: int
    total_cost_usd: float
    total_latency_ms: float
    outcomes: list[Outcome] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def get_aggregation_candidates(
    tier_num: int,
    failed_model: str,
    *,
    max_parallel_models: int = 2,
    task_type: str | None = None,
    complexity: float = 0.5,
    step_type: str | None = None,
    user_vec: np.ndarray | None = None,
    tenant_id: str = "default",
    requires_json: bool = False,
    requires_tools: bool = False,
    requires_vision: bool = False,
    requires_video: bool = False,
) -> list[str]:
    """
    Get candidate models for aggregation within a tier.

    Excludes the failed_model, returns up to max_parallel_models candidates
    ranked by Pareto fitness (via get_top_models) when profiles exist,
    otherwise returns the first available models.
    """
    all_models = _get_available_models_for_tier(tier_num)

    # Filter for required capabilities
    if requires_json or requires_tools or requires_vision or requires_video:
        from sriti.core.cascade.engine import _filter_for_capabilities
        all_models = _filter_for_capabilities(
            all_models, requires_json, requires_tools,
            requires_vision, requires_video,
        )

    # Remove the model that already failed
    candidates = [m for m in all_models if m != failed_model]

    if len(candidates) <= max_parallel_models:
        return candidates

    # Try to rank by reliability profiles via get_top_models
    try:
        from sriti.core.cascade.reliability import get_top_models
        ranked = get_top_models(
            task_type=task_type or "unknown",
            complexity=complexity,
            tier_num=tier_num,
            available_models=candidates,
            n=max_parallel_models,
            exclude_models={failed_model},
            step_type=step_type,
            user_vec=user_vec,
            tenant_id=tenant_id,
        )
        if ranked:
            return [model_id for model_id, _ in ranked]
    except Exception:
        logger.debug("get_top_models failed (falling back to first N)", exc_info=True)

    return candidates[:max_parallel_models]


# ---------------------------------------------------------------------------
# Parallel dispatch
# ---------------------------------------------------------------------------

async def dispatch_parallel(
    models: list[str],
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> list[tuple[str, LLMResponse | Exception]]:
    """
    Dispatch the same request to multiple models in parallel.

    Returns list of (model_id, LLMResponse | Exception).
    Failed calls are logged and included as exceptions — fail-open.
    """
    from sriti.core.execution import litellm_client

    async def _call_one(model: str) -> tuple[str, LLMResponse | Exception]:
        try:
            resp = await litellm_client.call(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (model, resp)
        except Exception as e:
            logger.warning("Aggregation dispatch failed for %s: %s", model, e)
            return (model, e)

    results = await asyncio.gather(*[_call_one(m) for m in models])
    return list(results)


# ---------------------------------------------------------------------------
# Best response selection
# ---------------------------------------------------------------------------

def select_best_response(
    candidates: list[tuple[str, LLMResponse]],
    quality_check_fn,
) -> tuple[tuple[str, LLMResponse] | None, dict[str, bool]]:
    """
    Run quality check on each candidate and return the first passing one.

    Returns (best_or_none, quality_results) where quality_results maps
    model_id -> True/False for each candidate's actual quality check result.
    """
    quality_results: dict[str, bool] = {}
    best: tuple[str, LLMResponse] | None = None
    for model_id, response in candidates:
        try:
            passed = quality_check_fn(response.content, response.finish_reason)
            quality_results[model_id] = passed
            if passed and best is None:
                best = (model_id, response)
        except Exception:
            quality_results[model_id] = False
            logger.debug("Quality check failed for %s in aggregation", model_id, exc_info=True)
    return best, quality_results


async def select_best_response_async(
    candidates: list[tuple[str, LLMResponse]],
    quality_check_fn,
) -> tuple[tuple[str, LLMResponse] | None, dict[str, bool]]:
    """
    Async version of select_best_response for embedding-based quality checks.

    Returns (best_or_none, quality_results) where quality_results maps
    model_id -> True/False for each candidate's actual quality check result.
    """
    quality_results: dict[str, bool] = {}
    best: tuple[str, LLMResponse] | None = None
    for model_id, response in candidates:
        try:
            passed = await quality_check_fn(response.content, response.finish_reason)
            quality_results[model_id] = passed
            if passed and best is None:
                best = (model_id, response)
        except Exception:
            quality_results[model_id] = False
            logger.debug("Quality check failed for %s in aggregation", model_id, exc_info=True)
    return best, quality_results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def attempt_aggregation(
    *,
    tier_num: int,
    failed_model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    task_type: str,
    complexity: float,
    quality_check_fn,
    is_async_quality: bool = False,
    step_type: str | None = None,
    session_id: str | None = None,
    user_vec: np.ndarray | None = None,
    tenant_id: str = "default",
    requires_json: bool = False,
    requires_tools: bool = False,
    requires_vision: bool = False,
    requires_video: bool = False,
) -> AggregationResult | None:
    """
    Attempt multi-expert aggregation within the current tier.

    Returns AggregationResult if at least one candidate passes quality,
    or None if aggregation is not eligible, has too few models, or all fail.

    Entire function is wrapped in try/except — fail-open.
    """
    try:
        from sriti.core.cascade.policy import get_aggregation_config
        config = get_aggregation_config()

        if not config.get("enabled", False):
            return None

        eligible_tiers = config.get("eligible_tiers", [2, 3])
        if tier_num not in eligible_tiers:
            return None

        max_parallel = config.get("max_parallel_models", 2)
        min_models = config.get("min_tier_models", 2)

        # Get candidates
        candidates = get_aggregation_candidates(
            tier_num,
            failed_model,
            max_parallel_models=max_parallel,
            task_type=task_type,
            complexity=complexity,
            step_type=step_type,
            user_vec=user_vec,
            tenant_id=tenant_id,
            requires_json=requires_json,
            requires_tools=requires_tools,
            requires_vision=requires_vision,
            requires_video=requires_video,
        )

        if len(candidates) < min_models:
            logger.debug(
                "Aggregation skipped: tier%d has %d candidates (min=%d)",
                tier_num, len(candidates), min_models,
            )
            return None

        # Parallel dispatch
        t0 = time.perf_counter()
        dispatch_results = await dispatch_parallel(
            candidates, messages, temperature, max_tokens,
        )
        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Separate successes from failures
        successful: list[tuple[str, LLMResponse]] = []
        total_cost = 0.0
        outcomes: list[Outcome] = []

        for model_id, result in dispatch_results:
            if isinstance(result, Exception):
                outcomes.append(Outcome(
                    model=model_id,
                    task_type=task_type,
                    complexity_score=complexity,
                    quality_passed=False,
                    quality_score=0.0,
                    latency_ms=0.0,
                    cost_usd=0.0,
                    escalated=False,
                    tier_num=tier_num,
                    step_type=step_type,
                    session_id=session_id,
                    tenant_id=tenant_id,
                ))
            else:
                successful.append((model_id, result))
                total_cost += result.cost_usd

        if not successful:
            logger.debug("Aggregation: all %d candidates failed", len(candidates))
            return None

        # Select best passing response
        if is_async_quality:
            best, quality_results = await select_best_response_async(successful, quality_check_fn)
        else:
            best, quality_results = select_best_response(successful, quality_check_fn)

        # Build outcomes for all successful candidates with actual quality results
        for model_id, resp in successful:
            passed = quality_results.get(model_id, False)
            outcomes.append(Outcome(
                model=model_id,
                task_type=task_type,
                complexity_score=complexity,
                quality_passed=passed,
                quality_score=0.0,
                latency_ms=resp.inference_latency_ms,
                cost_usd=resp.cost_usd,
                escalated=False,
                tier_num=tier_num,
                step_type=step_type,
                session_id=session_id,
                tenant_id=tenant_id,
            ))

        if best is None:
            logger.info(
                "Aggregation failed: %d candidates tried, none passed quality",
                len(successful),
            )
            return AggregationResult(
                response=successful[0][1],  # not used — caller checks None
                model_used=successful[0][0],
                candidates_tried=len(candidates),
                total_cost_usd=total_cost,
                total_latency_ms=total_latency_ms,
                outcomes=outcomes,
            ) if False else None  # explicit None — all failed

        best_model, best_response = best
        logger.info(
            "Aggregation succeeded: %s passed quality (%d candidates tried)",
            best_model, len(candidates),
        )

        return AggregationResult(
            response=best_response,
            model_used=best_model,
            candidates_tried=len(candidates),
            total_cost_usd=total_cost,
            total_latency_ms=total_latency_ms,
            outcomes=outcomes,
        )

    except Exception:
        logger.exception("Aggregation attempt failed (fail-open)")
        return None
