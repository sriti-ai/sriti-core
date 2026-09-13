from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
import redis.asyncio as aioredis

from sriti.core.schemas import ChatCompletionRequest
from sriti.core.schemas import RoutingStep, SritiMetadata
from sriti.core.cascade import classifier, policy, provider_health
from sriti.core.cascade import case_memory
from sriti.core.cascade.complexity import compute_complexity
from sriti.core.cascade.constants import HEDGING_PATTERNS as _HEDGING_PATTERNS
from sriti.core.cascade.policy import TierConfig
from sriti.core.cascade.reliability import (
    Outcome,
    _get_available_models_for_tier,
    get_best_model,
    schedule_record_outcome,
)
from sriti.core.cascade.aggregation import attempt_aggregation
from sriti.core.cascade.multimodal import content_to_str, detect_media_requirements
from sriti.core.catalog.registry import get_model
from sriti.core.execution import litellm_client
from sriti.core.execution.litellm_client import LLMResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality check helpers
# ---------------------------------------------------------------------------

# Phrases that indicate a model is uncertain, refusing, or deflecting.
# A response containing any of these triggers escalation without embedding.
# Provider errors that should trigger same-tier fallback, not escalation.
# Matched by class name (same pattern as retry.py) to avoid importing litellm.
_PROVIDER_ERROR_NAMES: frozenset[str] = frozenset({
    "RateLimitError",
    "ServiceUnavailableError",
    "InternalServerError",
    "APIConnectionError",
    "Timeout",
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "APIStatusError",       # litellm wraps 5xx as APIStatusError
    "NotFoundError",        # model ID invalid/inaccessible on provider (404) — try same-tier fallback
                            # before escalating; misconfigured model shouldn't burn frontier capacity
})


def _is_provider_error(exc: Exception) -> bool:
    """Return True if the exception is a transient provider error (rate limit, 5xx, timeout)."""
    return (
        type(exc).__name__ in _PROVIDER_ERROR_NAMES
        or isinstance(exc, (ConnectionError, TimeoutError))
    )



# Minimum token estimate (word count proxy) for responses to task types that
# require substantive output. Code and data_analysis responses shorter than
# this are likely truncated or refused.
_MIN_WORDS_BY_TASK: dict[str, int] = {
    "code": 30,
    "reasoning": 40,
    "math": 20,
    "data_analysis": 30,
    "document_review": 30,
    "structured_extraction": 10,
    "tool_use": 20,
    "instruction_following": 10,
}


def _quality_check_structural(
    response_text: str,
    finish_reason: str | None,
    task_type: str,
    tier: TierConfig,
    requires_json: bool = False,
) -> bool | None:
    """
    Layer 1 structural quality check. O(1), <1ms, no ML.

    Returns:
        False  — response is structurally bad, escalate immediately (skip embedding)
        True   — response passes structural check, can skip embedding if desired
        None   — no strong signal, fall through to embedding-based check
    """
    text_lower = response_text.lower().strip()

    # 0. Empty response is NEVER valid regardless of tier — a null/empty content
    # means the model produced nothing useful (e.g. thinking models using all tokens
    # on internal reasoning). Must fail even for tier 1 so cascade tries next model.
    if not text_lower and finish_reason != "tool_calls":
        logger.debug(
            "Quality check (structural): empty response → fail tier=%d",
            tier.tier_num,
        )
        return False

    # Tier1 (frontier) never escalates on other structural signals
    if tier.quality_threshold is None:
        return True

    # 1. Truncation: model ran out of tokens — response is definitionally incomplete
    if finish_reason == "length":
        logger.debug(
            "Quality check (structural): finish_reason=length → escalate tier=%d",
            tier.tier_num,
        )
        return False

    # 2. Tool call with no text content is valid (finish_reason="tool_calls")
    if not text_lower and finish_reason == "tool_calls":
        return True

    # 2b. Empty or near-empty response (shouldn't reach here due to check above, but safety net)
    if not text_lower:
        logger.debug(
            "Quality check (structural): empty response → escalate tier=%d",
            tier.tier_num,
        )
        return False

    # 3. Hedging / refusal detection — model is signalling it cannot help.
    # Only check the first 200 chars to avoid false positives on legitimate
    # responses that contain hedging phrases mid-text (e.g. agent workflows
    # producing meal plans that say "I can't determine exact needs without...").
    # Skipped when requires_json: a hedging phrase legitimately appearing inside
    # a JSON string value isn't a refusal, and an actual refusal (non-JSON prose)
    # is already caught by the JSON-validity check in step 5 below.
    if not requires_json:
        opening = text_lower[:200]
        for phrase in _HEDGING_PATTERNS:
            if phrase in opening:
                logger.debug(
                    "Quality check (structural): hedging pattern %r in opening → escalate tier=%d",
                    phrase, tier.tier_num,
                )
                return False

    # 4. Response too short for task type. Skipped when requires_json: minimum
    # word counts are calibrated for prose (code/reasoning/data_analysis), but
    # a short, correct JSON response (e.g. `[]`, `null`, a one-item array) is
    # common and valid — word count says nothing about JSON correctness, which
    # step 5 below verifies directly.
    if not requires_json:
        min_words = _MIN_WORDS_BY_TASK.get(task_type, 0)
        if min_words > 0:
            word_count = len(text_lower.split())
            if word_count < min_words:
                logger.debug(
                    "Quality check (structural): word_count=%d < min_words=%d "
                    "task_type=%s → escalate tier=%d",
                    word_count, min_words, task_type, tier.tier_num,
                )
                return False

    # 5. JSON validation: when structured output (json_object / json_schema) was
    # requested, verify the response is actually valid JSON.  Models that
    # technically "support" JSON mode but produce malformed output (trailing
    # commas, missing delimiters) must be caught here so the cascade can
    # escalate to a more capable model.
    if requires_json and response_text.strip():
        import json as _json
        try:
            _json.loads(response_text)
        except (ValueError, TypeError):
            logger.debug(
                "Quality check (structural): invalid JSON response → escalate tier=%d",
                tier.tier_num,
            )
            return False

    # No strong structural signal — let embedding-based check decide
    return None


async def _quality_check_low(
    user_vec: np.ndarray | None,
    response_text: str,
    finish_reason: str | None,
    task_type: str,
    tier: TierConfig,
    requires_json: bool = False,
) -> tuple[bool, float]:
    """
    Low-mode quality check.

    Layer 1 (structural, O(1)): finish_reason, hedging patterns, response length, JSON validity.
    Layer 2 (semantic): cosine similarity between user message and response.

    Structural check short-circuits embedding when there is a clear signal.
    user_vec is the pre-computed L2-normalised embedding from classify() —
    reused here to avoid re-embedding the user message.

    Returns (passed, quality_score) where quality_score is the cosine similarity
    when computed, or 1.0/0.0 for structural pass/fail.
    """
    if tier.quality_threshold is None:
        return True, 1.0

    # Layer 1 — structural signals (includes JSON validation when requires_json)
    structural = _quality_check_structural(
        response_text, finish_reason, task_type, tier, requires_json=requires_json,
    )
    if structural is not None:
        return structural, 1.0 if structural else 0.0

    # If user_vec is None (e.g. task_type pre-computed without embedding),
    # fall back to structural-only — cannot do semantic check without vector.
    if user_vec is None:
        logger.debug("Quality check (low): no user_vec, structural pass → accept tier=%d", tier.tier_num)
        return True, 1.0

    # Substantive response heuristic: if the response is long enough and
    # passed structural checks, it's likely valid.  Cosine similarity between
    # a question ("Create a meal plan") and its answer (structured recipes,
    # macros, ingredients) is naturally low — penalising this causes false
    # escalation in agent workflows.  Skip embedding for responses >100 words.
    word_count = len(response_text.split())
    if word_count >= 100:
        logger.debug(
            "Quality check (low): substantial response (%d words), structural pass → accept tier=%d",
            word_count, tier.tier_num,
        )
        return True, 0.8  # reasonable default when skipping embedding

    # Layer 2 — embedding-based cosine similarity (for short responses)
    model = classifier.get_model()
    resp_vecs = await asyncio.to_thread(
        lambda: list(model.embed([response_text]))
    )
    resp_arr = np.array(resp_vecs[0], dtype=np.float32)
    resp_norm = np.linalg.norm(resp_arr)
    if resp_norm > 0:
        resp_arr = resp_arr / resp_norm
    similarity = float(np.dot(user_vec, resp_arr))
    passed = similarity >= tier.quality_threshold
    logger.debug(
        "Quality check (low): similarity=%.4f threshold=%.4f passed=%s tier=%d words=%d",
        similarity, tier.quality_threshold, passed, tier.tier_num, word_count,
    )
    return passed, similarity


async def _quality_check_medium(
    user_vec: np.ndarray | None,
    response_text: str,
    finish_reason: str | None,
    task_type: str,
    tier: TierConfig,
    requires_json: bool = False,
) -> tuple[bool, float]:
    """
    Medium-mode stub: self-grading prompt deferred to M6.
    Falls back to low-mode logic.
    """
    logger.debug("quality_check.mode=medium — using low-mode logic (M6 stub)")
    return await _quality_check_low(
        user_vec, response_text, finish_reason, task_type, tier, requires_json=requires_json,
    )


# ---------------------------------------------------------------------------
# Capability filter
# ---------------------------------------------------------------------------

def _filter_for_capabilities(
    models: list[str],
    requires_json: bool = False,
    requires_tools: bool = False,
    requires_vision: bool = False,
    requires_video: bool = False,
) -> list[str]:
    """
    Filter model list to only those supporting required capabilities.

    Deterministic hard filter — runs BEFORE any explore-exploit logic.
    Exploration (epsilon-greedy, Thompson Sampling) only sees the output.

    Fail-open: unknown models (not in catalog) pass through (unless vision/video required).
    Fail-open for json/tools: if ALL filtered out, return originals.
    Fail-CLOSED for vision/video: if ALL filtered out, return empty.
    Vision: caller escalates to next tier. Video: caller hard-rejects with 400.
    """
    if not requires_json and not requires_tools and not requires_vision and not requires_video:
        return models
    result = []
    for m in models:
        info = get_model(m)
        if info is None:
            # fail-open for json/tools; fail-closed for vision/video
            if not requires_video and not requires_vision:
                result.append(m)
            continue
        if requires_json and not info.supports_json_mode:
            continue
        if requires_tools and not info.supports_tool_use:
            continue
        if requires_vision and not info.supports_vision:
            continue
        if requires_video and not info.supports_video:
            continue
        result.append(m)

    # Fail-open for json/tools: if ALL filtered out, return originals
    # Fail-CLOSED for vision/video: return empty so caller can escalate (vision)
    # or hard-reject (video)
    if not result and not requires_video and not requires_vision:
        return models
    return result


# ---------------------------------------------------------------------------
# Main cascade entry point
# ---------------------------------------------------------------------------

async def run(
    request: ChatCompletionRequest,
    metadata: SritiMetadata,
    redis_client: aioredis.Redis,
    messages: list[dict] | None = None,
    task_type: str | None = None,
    user_vec: np.ndarray | None = None,
    classification_score: float = 0.5,
    # Tenant-aware overrides (all default to None = current behavior)
    tier_overrides: dict[int, TierConfig] | None = None,
    max_escalations: int = 99,
    quality_check_mode_override: str | None = None,
    frontier_tenant_id: str | None = None,
    frontier_cap_override: float | None = None,
    # Phase 2: workflow context for case memory + expert bank
    step_type: str | None = None,
    session_id: str | None = None,
    # Tool calling
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    # Continuous improvement: memory-informed routing adjustment
    learning_adjustment: object | None = None,
    # Bandit: override initial model selection
    bandit_model: str | None = None,
) -> tuple[LLMResponse, SritiMetadata]:
    """
    Execute the full cascade loop for a non-streaming request.

    1. Classify the task type from the last user message (skipped if pre-computed).
    2. Select the initial tier based on task type + policy.
    3. Dispatch to litellm_client, run quality check.
    4. Escalate to next tier if quality fails; stop at tier1 (frontier).
    5. Enforce frontier traffic cap via Redis before any tier1 dispatch.
    6. Return (LLMResponse, populated SritiMetadata).

    Args:
        messages:   Pre-processed messages (e.g. compressed). If None, uses request.messages.
        task_type:  Pre-computed task type — skips internal classify call.
        user_vec:   Pre-computed L2-normalised user embedding — skips re-embed.
    """
    trail: list[RoutingStep] = []

    # Detect structured output / tool use requirements
    requires_structured = (
        request.response_format is not None
        and request.response_format.get("type") in ("json_object", "json_schema")
    )
    requires_tools = tools is not None or any(
        m.get("role") == "tool" or m.get("tool_calls") is not None
        for m in (messages or [{"role": m.role, "content": m.content} for m in request.messages])
    )

    if requires_structured:
        trail.append(RoutingStep(
            step="capability_filter",
            detail=f"Structured output required ({request.response_format.get('type', 'json')})",
            value=request.response_format.get("type"),
        ))

    if task_type is None:
        # Classify internally (fallback: streaming path, tests, direct calls)
        raw = [{"role": m.role, "content": m.content} for m in request.messages]
        task_type, classification_score, user_vec = await asyncio.to_thread(
            classifier.classify, raw
        )
        logger.info("Classified task_type=%s score=%.4f", task_type, classification_score)
    else:
        logger.info("Using pre-computed task_type=%s", task_type)

    trail.append(RoutingStep(
        step="classify",
        detail=f"Classified as '{task_type}'",
        value=f"{classification_score:.2f}",
    ))

    working_messages = messages or [{"role": m.role, "content": m.content} for m in request.messages]

    # Detect multimodal content requirements (vision/video)
    media_reqs = detect_media_requirements(working_messages)
    _requires_vision = media_reqs.requires_vision
    _requires_video = media_reqs.requires_video

    if _requires_vision or _requires_video:
        trail.append(RoutingStep(
            step="capability_filter",
            detail=f"Multimodal content detected (vision={_requires_vision}, video={_requires_video})",
            value="video" if _requires_video else "vision",
        ))

    # Step 2 — Complexity scoring + initial tier selection
    complexity = compute_complexity(working_messages, task_type, classification_score)

    complexity_label = "simple" if complexity < 0.33 else ("moderate" if complexity < 0.66 else "complex")
    trail.append(RoutingStep(
        step="complexity",
        detail=f"Complexity: {complexity:.2f} ({complexity_label})",
        value=f"{complexity:.3f}",
    ))

    # Use tier_overrides if provided (tenant-specific models), else global tiers
    def _get_tier(tier_num: int) -> TierConfig:
        if tier_overrides and tier_num in tier_overrides:
            return tier_overrides[tier_num]
        return policy.get_tier(tier_num)

    # ── Extract all learning adjustment signals once ─────────────────────
    _is_shadow      = getattr(learning_adjustment, "is_shadow", False)
    _override_tier  = getattr(learning_adjustment, "override_tier", None)
    _max_tier       = getattr(learning_adjustment, "max_tier", None)
    _prefer_models  = getattr(learning_adjustment, "prefer_models", None)
    _avoid_models   = getattr(learning_adjustment, "avoid_models", None)
    _hard_avoid     = getattr(learning_adjustment, "hard_avoid_models", None)
    _req_tool_use   = getattr(learning_adjustment, "requires_tool_use", False)
    _req_json_mode  = getattr(learning_adjustment, "requires_json_mode", False)
    _applied_reason = getattr(learning_adjustment, "applied_reason", "memory_rule")
    _inject_count   = len(getattr(learning_adjustment, "memories_to_inject", []))

    # Merge learned capability requirements with caller-declared requirements
    _requires_tools_eff       = requires_tools or _req_tool_use
    _requires_structured_eff  = requires_structured or _req_json_mode

    # ── Capability-constrained model accessor ──────────────────────────────
    # Single function that returns only models capable of handling this request.
    # All downstream routing (tier selection, exploration, escalation, fallback)
    # calls this instead of raw _get_available_models_for_tier, so incapable
    # models are NEVER seen by any explore-exploit logic.
    def _get_capable_models(tier_num: int) -> list[str]:
        models = _get_available_models_for_tier(tier_num)
        return _filter_for_capabilities(
            models, _requires_structured_eff, _requires_tools_eff,
            _requires_vision, _requires_video,
        )

    def _get_initial() -> TierConfig:
        initial_num = policy.get_initial_tier(task_type).tier_num
        if learning_adjustment:
            # Always record memories were evaluated so frontend can show impact
            trail.append(RoutingStep(
                step="learning_memory",
                detail=_applied_reason,
                value=(
                    f"tier{_override_tier}" if _override_tier and not _is_shadow
                    else "shadow" if _is_shadow and _override_tier
                    else "inject_only"
                ),
                tier=_override_tier if not _is_shadow else None,
                metadata={
                    "shadow": _is_shadow,
                    "override_tier": _override_tier,
                    "max_tier": _max_tier,
                    "prefer_models": _prefer_models,
                    "avoid_models": _avoid_models,
                    "hard_avoid_models": _hard_avoid,
                    "requires_tool_use": _req_tool_use,
                    "requires_json_mode": _req_json_mode,
                    "injected_count": _inject_count,
                    "reason": _applied_reason,
                },
            ))

            if not _is_shadow and _override_tier is not None:
                return _get_tier(_override_tier)
        return _get_tier(initial_num)

    def _get_next(current_num: int) -> TierConfig | None:
        next_num = current_num - 1
        if next_num < 1:
            return None
        # Memory max_tier cap: don't escalate to a more expensive tier than allowed.
        # max_tier=3 means "stay in T3 only". In our numbering, lower = more expensive,
        # so escalation goes 3→2→1. Blocking at max_tier=2 means we never reach T1.
        if _max_tier is not None and not _is_shadow:
            if next_num < _max_tier:
                logger.info(
                    "Escalation to tier%d blocked by memory max_tier=%d",
                    next_num, _max_tier,
                )
                return None
        try:
            return _get_tier(next_num)
        except KeyError:
            return None

    _tenant_id = frontier_tenant_id or "default"

    async def _apply_health_filter(models: list[str]) -> list[str]:
        """Deprioritise models whose provider is currently flagged DEGRADED."""
        try:
            return await provider_health.filter_candidates(redis_client, models)
        except Exception:
            return models

    current_tier: TierConfig = _get_initial()
    available_models = _get_capable_models(current_tier.tier_num)

    # Auto-escalate through tiers until we find one with capable models.
    # _get_capable_models returns empty when no models in a tier match the
    # request's capability requirements (vision, video, tools, structured).
    while not available_models:
        next_tier = _get_next(current_tier.tier_num)
        if next_tier is None:
            from fastapi import HTTPException
            cap = "video" if _requires_video else "vision" if _requires_vision else "required capabilities"
            raise HTTPException(
                status_code=400,
                detail=f"Request requires {cap} but no capable model is available in any tier",
            )
        trail.append(RoutingStep(
            step="capability_skip",
            detail=f"No capable models in Tier {current_tier.tier_num} — skipping to Tier {next_tier.tier_num}",
            tier=next_tier.tier_num,
        ))
        current_tier = next_tier
        available_models = _get_capable_models(current_tier.tier_num)

    available_models = await _apply_health_filter(available_models)
    selected_model, explored = await get_best_model(
        task_type, complexity, current_tier.tier_num, available_models,
        step_type=step_type, user_vec=user_vec, tenant_id=_tenant_id,
        prefer_models=_prefer_models, avoid_models=_avoid_models, hard_avoid_models=_hard_avoid,
    )

    # Bandit override: use the bandit-selected model if provided and valid
    _bandit_applied = False
    if bandit_model and bandit_model in available_models:
        selected_model = bandit_model
        explored = False
        _bandit_applied = True
    elif bandit_model:
        # Bandit chose a model not in this tier's pool — check if it exists in any tier
        from sriti.core.catalog.registry import get_model as get_catalog_model
        cat_entry = get_catalog_model(bandit_model)
        if cat_entry:
            # Verify bandit model has required capabilities before accepting
            _bandit_capable = True
            if _requires_vision and not getattr(cat_entry, "supports_vision", False):
                _bandit_capable = False
            if _requires_video and not getattr(cat_entry, "supports_video", False):
                _bandit_capable = False
            if _bandit_capable:
                bandit_tier = _get_tier(cat_entry.default_tier)
                current_tier = bandit_tier
                selected_model = bandit_model
                explored = False
                _bandit_applied = True

    routing_reason = f"task_type:{task_type}:complexity:{complexity:.3f}"
    quality_mode = quality_check_mode_override or policy.get_quality_check_mode()
    # Request time budget — bail out gracefully before infrastructure timeouts
    # (ALB 300s, Envoy 240s, frontend proxy 180s). Must stay >= the largest
    # configured tier SLO (tier3_fast.max_latency_ms = 240s in policy.yaml,
    # for local vision inference on modest hardware) or _effective_timeout below
    # silently caps every dispatch at this budget regardless of that SLO,
    # which is exactly what a 90s budget did here previously. Kept a few
    # seconds under sriti/client.py's 280s HTTP timeout so the cascade's own
    # budget_exhausted/504 path fires first, with a clear reason, instead of
    # the client timing out first and masking it.
    _request_budget_s = 270.0
    _request_t0 = time.perf_counter()
    escalated = False
    _quality_passed: bool | None = None
    result: LLMResponse | None = None
    # Tier that actually produced `result` — tracked separately from
    # `current_tier` (which advances to the *next* tier being attempted
    # before we know whether that attempt will even succeed). Needed so a
    # tier1-unreachable fallback (below) can report metadata that matches
    # what's actually being returned, not the tier that just failed.
    _last_successful_tier: TierConfig | None = None
    total_inference_ms = 0.0
    escalation_count = 0
    _failed_models: set[str] = set()  # track provider-failed models for same-tier fallback

    if _bandit_applied:
        trail.append(RoutingStep(
            step="bandit_action",
            detail=f"Bandit selected {selected_model}",
            value=selected_model,
            tier=current_tier.tier_num,
            metadata={"source": "thompson_sampling"},
        ))

    trail.append(RoutingStep(
        step="tier_select",
        detail=f"Starting at Tier {current_tier.tier_num} ({current_tier.name})",
        tier=current_tier.tier_num,
    ))
    trail.append(RoutingStep(
        step="model_select",
        detail=f"Selected {selected_model} (explore)" if explored else f"Selected {selected_model}",
        value=selected_model,
        tier=current_tier.tier_num,
    ))

    # Step 4 — Cascade loop
    _cap_fell_back = False  # prevents tier1↔tier2 ping-pong when frontier cap is hit
    while True:
        # Budget check — stop cascading before infrastructure timeouts kill us silently
        _elapsed = time.perf_counter() - _request_t0
        if _elapsed >= _request_budget_s:
            logger.warning(
                "Request time budget exhausted (%.1fs >= %.1fs) — returning best result or raising",
                _elapsed, _request_budget_s,
            )
            if result is not None:
                trail.append(RoutingStep(
                    step="budget_exhausted",
                    detail=f"Time budget exhausted ({_elapsed:.1f}s) — returning best available result",
                    metadata={"elapsed_s": round(_elapsed, 1), "budget_s": _request_budget_s},
                ))
                break
            from fastapi import HTTPException
            raise HTTPException(
                status_code=504,
                detail=f"Request time budget exhausted ({_elapsed:.1f}s) across all cascade tiers",
            )

        # Frontier cap check — only when about to dispatch tier1 (frontier)
        if current_tier.tier_num == 1:
            allowed = await policy.check_and_record_frontier(
                redis_client,
                tenant_id=frontier_tenant_id,
                cap_override=frontier_cap_override,
            )
            if not allowed:
                logger.info("Frontier cap exceeded — falling back to tier2")
                _cap_fell_back = True
                trail.append(RoutingStep(
                    step="cap_fallback",
                    detail="Frontier cap exceeded — falling back to Tier 2",
                    tier=2,
                ))
                current_tier = _get_tier(2)
                available_models = _get_capable_models(2)
                available_models = await _apply_health_filter(available_models)
                selected_model, explored = await get_best_model(
                    task_type, complexity, 2, available_models,
                    step_type=step_type, user_vec=user_vec, tenant_id=_tenant_id,
                    prefer_models=_prefer_models, avoid_models=_avoid_models, hard_avoid_models=_hard_avoid,
                )
                routing_reason = f"task_type:{task_type}:complexity:{complexity:.3f}:cap_fallback"
                trail.append(RoutingStep(
                    step="model_select",
                    detail=f"Selected {selected_model} (explore)" if explored else f"Selected {selected_model}",
                    value=selected_model,
                    tier=2,
                ))

        logger.info(
            "Dispatching to tier%d model=%s complexity=%.3f",
            current_tier.tier_num, selected_model, complexity,
        )
        # Phase 2: enforce tier.max_latency_ms as a hard SLO on each dispatch.
        # The SLO timeout is passed directly to litellm_client.call() so it
        # applies at the HTTP level — no nested asyncio.wait_for needed.
        # Also cap at remaining budget so we don't overshoot the request deadline.
        _slo_enforced = bool(
            policy.is_latency_slo_enforced()
            and current_tier.max_latency_ms
            and current_tier.max_latency_ms > 0
        )
        _remaining_budget = _request_budget_s - (time.perf_counter() - _request_t0)
        _slo_seconds = (current_tier.max_latency_ms / 1000.0) if _slo_enforced else None
        # Use the tighter of SLO timeout and remaining budget
        _effective_timeout = min(
            _slo_seconds or _remaining_budget,
            _remaining_budget,
        )
        _dispatch_t0 = time.perf_counter()
        try:
            result = await litellm_client.call(
                model=selected_model,
                messages=working_messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                timeout=_effective_timeout,
                response_format=request.response_format,
                tools=tools,
                tool_choice=tool_choice,
            )
            _last_successful_tier = current_tier
        except Exception as call_exc:
            # Count time spent waiting on provider even when it fails —
            # otherwise timeouts/errors inflate platform_overhead_ms.
            _failed_dispatch_ms = (time.perf_counter() - _dispatch_t0) * 1000.0
            total_inference_ms += _failed_dispatch_ms
            _recorded_failure_outcome = False
            _is_timeout = (
                isinstance(call_exc, (asyncio.TimeoutError, TimeoutError))
                or type(call_exc).__name__ in ("Timeout", "TimeoutException", "ReadTimeout")
            )
            if _is_timeout:
                logger.warning(
                    "SLO/budget timeout on tier%d model=%s after %.1fs (limit=%.1fs) — treating as transient failure",
                    current_tier.tier_num, selected_model, _failed_dispatch_ms / 1000.0, _effective_timeout,
                )
                trail.append(RoutingStep(
                    step="slo_timeout",
                    detail=f"Timeout after {_effective_timeout:.1f}s on {selected_model}",
                    value=selected_model,
                    tier=current_tier.tier_num,
                    metadata={
                        "model": selected_model,
                        "slo_ms": current_tier.max_latency_ms,
                        "effective_timeout_s": round(_effective_timeout, 1),
                    },
                ))
                schedule_record_outcome(Outcome(
                    model=selected_model,
                    task_type=task_type,
                    complexity_score=complexity,
                    quality_passed=False,
                    quality_score=0.0,
                    latency_ms=_failed_dispatch_ms,
                    cost_usd=0.0,
                    escalated=True,
                    tier_num=current_tier.tier_num,
                    step_type=step_type,
                    session_id=session_id,
                    tenant_id=_tenant_id,
                ))
                _recorded_failure_outcome = True
            exc_name = type(call_exc).__name__
            is_provider_error = _is_provider_error(call_exc)

            if is_provider_error:
                # ── Provider error (rate limit, 5xx, timeout) ──────────
                # Try another model in the SAME tier before escalating.
                # A Groq rate limit doesn't mean this tier can't handle
                # the request — it means this *provider* can't right now.
                logger.warning(
                    "Provider error on tier%d model=%s: %s: %s — trying same-tier fallback",
                    current_tier.tier_num, selected_model, exc_name, call_exc,
                )
                trail.append(RoutingStep(
                    step="provider_error",
                    detail=f"Provider error ({exc_name}) on {selected_model}",
                    tier=current_tier.tier_num,
                    metadata={"model": selected_model, "reason": exc_name},
                ))

                # Get alternative models in same tier, excluding the failed one
                # Also exclude hard_avoid models from fallback pool (they failed once already
                # and memory says they're unreliable — don't retry them even in fallback)
                _failed_models.add(selected_model)
                fallback_models = [
                    m for m in _get_capable_models(current_tier.tier_num)
                    if m not in _failed_models
                    and m not in (_hard_avoid or [])
                    and m not in (_avoid_models or [])
                ]

                if fallback_models:
                    # Pick best alternative in same tier
                    selected_model, explored = await get_best_model(
                        task_type, complexity, current_tier.tier_num, fallback_models,
                        step_type=step_type, user_vec=user_vec, tenant_id=_tenant_id,
                        prefer_models=_prefer_models, avoid_models=_avoid_models, hard_avoid_models=_hard_avoid,
                    )
                    trail.append(RoutingStep(
                        step="provider_fallback",
                        detail=f"Same-tier fallback → {selected_model}",
                        value=selected_model,
                        tier=current_tier.tier_num,
                    ))
                    continue  # retry loop with new model, same tier

                # No same-tier alternatives left — fall through to escalation
                logger.info(
                    "No same-tier alternatives left on tier%d — escalating",
                    current_tier.tier_num,
                )

            else:
                logger.warning(
                    "Non-retryable error on tier%d model=%s: %s: %s",
                    current_tier.tier_num, selected_model, exc_name, call_exc,
                )

            # ── Escalate to next tier (provider error with no fallback, or non-retryable) ──
            next_tier = None if current_tier.tier_num == 1 else _get_next(current_tier.tier_num)
            if next_tier is None:
                # Nowhere left to escalate to (already at tier1, or a
                # memory max_tier cap blocks further escalation). If an
                # earlier tier already produced a real response — just not
                # one that passed the quality gate, or we never got that
                # far and are escalating past a provider outage — serve
                # that instead of hard-failing the whole request. Mirrors
                # the graceful "no next tier — return current result"
                # behavior the quality-check-failure path already has
                # further below; this is the equivalent for the
                # exception/unreachable-tier path, which previously always
                # raised even when a perfectly good fallback answer was
                # sitting right there. Found via a real demo run: a
                # missing tier1 API key turned an intermittent tier2
                # quality-gate quirk into a hard 500 instead of just
                # serving tier2's answer.
                if result is not None and _last_successful_tier is not None:
                    logger.warning(
                        "Tier%d unreachable (%s: %s) and no further tier to escalate "
                        "to — degrading to last successful tier%d result (%s) instead "
                        "of failing the request",
                        current_tier.tier_num, exc_name, call_exc,
                        _last_successful_tier.tier_num, result.model_used,
                    )
                    trail.append(RoutingStep(
                        step="tier_unreachable_degrade",
                        detail=(
                            f"Tier {current_tier.tier_num} unreachable ({exc_name}) — "
                            f"serving last successful Tier {_last_successful_tier.tier_num} "
                            "result instead of failing"
                        ),
                        tier=_last_successful_tier.tier_num,
                        metadata={"error": exc_name, "fallback_model": result.model_used},
                    ))
                    if not _recorded_failure_outcome:
                        schedule_record_outcome(Outcome(
                            model=selected_model,
                            task_type=task_type,
                            complexity_score=complexity,
                            quality_passed=False,
                            quality_score=0.0,
                            latency_ms=_failed_dispatch_ms,
                            cost_usd=0.0,
                            escalated=True,
                            tier_num=current_tier.tier_num,
                            step_type=step_type,
                            session_id=session_id,
                            tenant_id=_tenant_id,
                        ))
                        _recorded_failure_outcome = True
                    current_tier = _last_successful_tier
                    break
                raise
            _failed_tier_num = current_tier.tier_num
            trail.append(RoutingStep(
                step="escalate",
                detail=f"{'All same-tier models failed' if is_provider_error else f'Error ({exc_name})'} — escalating to Tier {next_tier.tier_num}",
                tier=next_tier.tier_num,
            ))
            current_tier = next_tier
            _failed_models.clear()  # reset for new tier
            available_models = _get_capable_models(next_tier.tier_num)
            # Skip tiers with no capable models
            _no_capable_models_degrade = False
            while not available_models:
                _skip_tier = _get_next(current_tier.tier_num)
                if _skip_tier is None:
                    # No capable models in any remaining tier. Same fallback as
                    # the tier-unreachable path above — serve the last
                    # successful tier's result instead of discarding it, if one
                    # exists. Previously this always raised, even when a good
                    # answer was already sitting in `result` (e.g. tier3
                    # succeeded, its quality gate failed, and no tier above it
                    # has a capable model).
                    if result is not None and _last_successful_tier is not None:
                        logger.warning(
                            "No capable models in any remaining tier — degrading "
                            "to last successful tier%d result (%s) instead of "
                            "failing the request",
                            _last_successful_tier.tier_num, result.model_used,
                        )
                        trail.append(RoutingStep(
                            step="tier_unreachable_degrade",
                            detail=(
                                "No capable models in any remaining tier — serving "
                                f"last successful Tier {_last_successful_tier.tier_num} "
                                "result instead of failing"
                            ),
                            tier=_last_successful_tier.tier_num,
                            metadata={"fallback_model": result.model_used},
                        ))
                        if not _recorded_failure_outcome:
                            schedule_record_outcome(Outcome(
                                model=selected_model,
                                task_type=task_type,
                                complexity_score=complexity,
                                quality_passed=False,
                                quality_score=0.0,
                                latency_ms=_failed_dispatch_ms,
                                cost_usd=0.0,
                                escalated=True,
                                tier_num=_failed_tier_num,
                                step_type=step_type,
                                session_id=session_id,
                                tenant_id=_tenant_id,
                            ))
                            _recorded_failure_outcome = True
                        current_tier = _last_successful_tier
                        _no_capable_models_degrade = True
                        break
                    raise  # no capable models in any remaining tier, nothing to fall back to
                trail.append(RoutingStep(
                    step="capability_skip",
                    detail=f"No capable models in Tier {current_tier.tier_num} — skipping to Tier {_skip_tier.tier_num}",
                    tier=_skip_tier.tier_num,
                ))
                current_tier = _skip_tier
                available_models = _get_capable_models(_skip_tier.tier_num)
            if _no_capable_models_degrade:
                break
            available_models = await _apply_health_filter(available_models)
            selected_model, explored = await get_best_model(
                task_type, complexity, current_tier.tier_num, available_models,
                step_type=step_type, user_vec=user_vec, tenant_id=_tenant_id,
                prefer_models=_prefer_models, avoid_models=_avoid_models, hard_avoid_models=_hard_avoid,
            )
            trail.append(RoutingStep(
                step="model_select",
                detail=f"Selected {selected_model} (explore)" if explored else f"Selected {selected_model}",
                value=selected_model,
                tier=current_tier.tier_num,
            ))
            escalated = True
            escalation_count += 1
            continue
        total_inference_ms += result.inference_latency_ms

        # Phase 3: fire-and-forget provider-health recording. Never blocks.
        provider_health.record(
            redis_client, selected_model, result.inference_latency_ms, success=True,
        )

        trail.append(RoutingStep(
            step="dispatch",
            detail=f"Response in {result.inference_latency_ms:,.0f}ms",
            value=f"{result.inference_latency_ms:.0f}ms",
            tier=current_tier.tier_num,
            metadata={
                "model": selected_model,
                "latency_ms": round(result.inference_latency_ms, 2),
                "cost_usd": round(result.cost_usd, 6) if result.cost_usd else 0,
                "tokens_in": result.prompt_tokens,
                "tokens_out": result.completion_tokens,
                "response_snippet": (result.content or "")[:200],
            },
        ))

        # Quality check — user_vec reused from classify(), only response is embedded
        # For multimodal requests, force structural-only: semantic similarity
        # correlates poorly when the request contains images/video.
        _effective_quality_mode = quality_mode
        if _requires_vision or _requires_video:
            _effective_quality_mode = "structural"

        _quality_score = 0.0
        if _effective_quality_mode == "structural":
            # Structural-only check (low_latency mode) — skip embedding entirely
            structural = _quality_check_structural(
                result.content, result.finish_reason, task_type, current_tier,
                requires_json=_requires_structured_eff,
            )
            passed = structural if structural is not None else True
            _quality_score = 1.0 if passed else 0.0
        elif _effective_quality_mode == "medium":
            passed, _quality_score = await _quality_check_medium(
                user_vec, result.content, result.finish_reason, task_type, current_tier,
                requires_json=_requires_structured_eff,
            )
        else:
            passed, _quality_score = await _quality_check_low(
                user_vec, result.content, result.finish_reason, task_type, current_tier,
                requires_json=_requires_structured_eff,
            )

        _quality_passed = passed

        # Record outcome (fire-and-forget — zero latency impact)
        schedule_record_outcome(Outcome(
            model=selected_model,
            task_type=task_type,
            complexity_score=complexity,
            quality_passed=passed,
            quality_score=_quality_score,
            latency_ms=result.inference_latency_ms,
            cost_usd=result.cost_usd,
            escalated=escalated,
            tier_num=current_tier.tier_num,
            step_type=step_type,
            session_id=session_id,
            tenant_id=_tenant_id,
        ))

        # Phase 2: record execution case for episodic memory (fire-and-forget)
        if user_vec is not None:
            _last_user = ""
            for m in reversed(working_messages):
                if m.get("role") == "user":
                    _last_user = content_to_str(m.get("content", ""))
                    break
            if _last_user:
                case_memory.schedule_store(
                    tenant_id=_tenant_id,
                    session_id=session_id,
                    step_type=step_type,
                    task_type=task_type,
                    prompt_hash=case_memory.make_prompt_hash(_last_user),
                    embedding=user_vec,
                    model_used=selected_model,
                    quality_passed=passed,
                    quality_score=_quality_score,
                    cost_usd=result.cost_usd,
                    latency_ms=result.inference_latency_ms,
                )

        # Already at frontier (tier1) — always return regardless of quality
        if current_tier.tier_num == 1:
            trail.append(RoutingStep(
                step="quality_pass",
                detail="Quality check passed (frontier)",
                tier=current_tier.tier_num,
                metadata={"model": selected_model, "reason": "frontier_always_pass"},
            ))
            break

        if passed:
            trail.append(RoutingStep(
                step="quality_pass",
                detail="Quality check passed",
                tier=current_tier.tier_num,
                metadata={"model": selected_model},
            ))
            break

        # Quality failed
        threshold_str = f"{current_tier.quality_threshold}" if current_tier.quality_threshold else "?"
        trail.append(RoutingStep(
            step="quality_fail",
            detail=f"Quality check failed (threshold: {threshold_str})",
            tier=current_tier.tier_num,
            metadata={
                "model": selected_model,
                "threshold": threshold_str,
                "response_snippet": (result.content or "")[:200],
            },
        ))

        # --- Multi-expert aggregation: try other models in the same tier ---
        def _structural_qc(text: str, finish: str | None) -> bool:
            s = _quality_check_structural(
                text, finish, task_type, current_tier,
                requires_json=_requires_structured_eff,
            )
            return s if s is not None else True

        async def _async_qc(text: str, finish: str | None) -> bool:
            if quality_mode == "medium":
                passed, _ = await _quality_check_medium(
                    user_vec, text, finish, task_type, current_tier,
                    requires_json=_requires_structured_eff,
                )
                return passed
            passed, _ = await _quality_check_low(
                user_vec, text, finish, task_type, current_tier,
                requires_json=_requires_structured_eff,
            )
            return passed

        is_async_qc = quality_mode != "structural"
        # Pre-fetch candidate list so we can log it even if aggregation returns None
        from sriti.core.cascade.aggregation import get_aggregation_candidates
        _agg_candidates = get_aggregation_candidates(
            current_tier.tier_num, selected_model,
            task_type=task_type, complexity=complexity,
            step_type=step_type, user_vec=user_vec, tenant_id=_tenant_id,
            requires_json=requires_structured, requires_tools=requires_tools,
            requires_vision=_requires_vision, requires_video=_requires_video,
        )
        agg_result = await attempt_aggregation(
            tier_num=current_tier.tier_num,
            failed_model=selected_model,
            messages=working_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            task_type=task_type,
            complexity=complexity,
            quality_check_fn=_async_qc if is_async_qc else _structural_qc,
            is_async_quality=is_async_qc,
            step_type=step_type,
            session_id=session_id,
            user_vec=user_vec,
            tenant_id=_tenant_id,
            requires_json=requires_structured,
            requires_tools=requires_tools,
            requires_vision=_requires_vision,
            requires_video=_requires_video,
        )
        if agg_result is not None:
            result = agg_result.response
            total_inference_ms += agg_result.total_latency_ms
            # Collect all candidate models and their pass/fail status
            agg_candidates_detail = [
                {"model": o.model, "passed": o.quality_passed}
                for o in agg_result.outcomes
            ]
            trail.append(RoutingStep(
                step="aggregation_pass",
                detail=f"Aggregation succeeded via {agg_result.model_used} ({agg_result.candidates_tried} candidates)",
                value=agg_result.model_used,
                tier=current_tier.tier_num,
                metadata={
                    "model": agg_result.model_used,
                    "candidates_tried": agg_result.candidates_tried,
                    "latency_ms": round(agg_result.total_latency_ms, 2),
                    "candidates": agg_candidates_detail,
                },
            ))
            for o in agg_result.outcomes:
                schedule_record_outcome(o)
            # Update selected_model for metadata
            selected_model = agg_result.model_used
            metadata = metadata.model_copy(update={
                "aggregation_used": True,
                "aggregation_candidates": agg_result.candidates_tried,
                "aggregation_method": "best_pick",
            })
            break

        trail.append(RoutingStep(
            step="aggregation_fail",
            detail="Aggregation failed — proceeding to escalation",
            tier=current_tier.tier_num,
            metadata={
                "candidates_tried": len(_agg_candidates),
                "candidates": [
                    {"model": m, "passed": False}
                    for m in _agg_candidates
                ],
            },
        ))

        # Max escalations reached — return current result
        if escalation_count >= max_escalations:
            logger.info(
                "Max escalations (%d) reached — returning tier%d result",
                max_escalations, current_tier.tier_num,
            )
            break

        # Quality failed — try to escalate
        # But if we already fell back from frontier cap, don't re-escalate to tier1
        # (would create a tier1→cap→tier2→fail→tier1→cap loop)
        next_tier = _get_next(current_tier.tier_num)
        if next_tier is not None and next_tier.tier_num == 1 and _cap_fell_back:
            logger.info("Skipping tier1 escalation — frontier cap already hit this request")
            next_tier = None
        if next_tier is None:
            logger.warning(
                "No next tier available — returning tier%d result", current_tier.tier_num
            )
            break

        logger.info(
            "Quality check failed on tier%d — escalating to tier%d",
            current_tier.tier_num, next_tier.tier_num,
        )
        trail.append(RoutingStep(
            step="escalate",
            detail=f"Escalating to Tier {next_tier.tier_num}",
            tier=next_tier.tier_num,
        ))
        current_tier = next_tier
        available_models = _get_capable_models(next_tier.tier_num)
        # Skip tiers with no capable models
        while not available_models:
            _skip_tier = _get_next(current_tier.tier_num)
            if _skip_tier is None:
                break  # return current result — at least we have something
            trail.append(RoutingStep(
                step="capability_skip",
                detail=f"No capable models in Tier {current_tier.tier_num} — skipping to Tier {_skip_tier.tier_num}",
                tier=_skip_tier.tier_num,
            ))
            current_tier = _skip_tier
            available_models = _get_capable_models(_skip_tier.tier_num)
        if not available_models:
            break  # exhausted all tiers
        available_models = await _apply_health_filter(available_models)
        selected_model, explored = await get_best_model(
            task_type, complexity, current_tier.tier_num, available_models,
            step_type=step_type, user_vec=user_vec, tenant_id=_tenant_id,
            prefer_models=_prefer_models, avoid_models=_avoid_models, hard_avoid_models=_hard_avoid,
        )
        trail.append(RoutingStep(
            step="model_select",
            detail=f"Selected {selected_model} (explore)" if explored else f"Selected {selected_model}",
            value=selected_model,
            tier=current_tier.tier_num,
        ))
        escalated = True
        escalation_count += 1

    # Step 5 — Populate SritiMetadata
    assert result is not None  # loop always executes at least once
    provider = result.model_used.split("/")[0] if "/" in result.model_used else "openai"

    updated_metadata = metadata.model_copy(update={
        "model_tier": current_tier.tier_num,
        "model_used": result.model_used,
        "provider": provider,
        "routing_reason": routing_reason,
        "complexity_score": complexity,
        "escalated": escalated,
        "quality_passed": _quality_passed,
        "cost_usd": result.cost_usd,
        "tokens_in": result.prompt_tokens,
        "tokens_out": result.completion_tokens,
        "inference_latency_ms": round(total_inference_ms, 2),
        "routing_trail": trail,
    })

    return result, updated_metadata
