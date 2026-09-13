"""Flat orchestrating entrypoint for one completion request: classify,
check the semantic cache, run the cascade on a miss, then fire-and-forget
cache/outcome writes.

The pipeline is intentionally a single flat function rather than a
composable DAG executor — there is exactly one fixed sequence:
classify -> compress -> cache lookup -> cascade -> cache store ->
record outcome. The added complexity of a general workflow engine is
not warranted for a fixed sequence with no branching plan overrides.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from redis.asyncio import Redis

from sriti.core import telemetry
from sriti.core.cache.semantic_cache import get_cache
from sriti.core.cascade import classifier
from sriti.core.cascade.engine import run as engine_run
from sriti.core.compression.llmlingua import maybe_compress
from sriti.core.schemas import ChatCompletionRequest, SritiMetadata

logger = logging.getLogger(__name__)

# Maps orchestration's EscalationTier values to Sriti's numeric tier
# convention (1 = frontier/most capable, 3 = local/cheapest). Kept here
# rather than in orchestration/ — orchestration only knows about the
# EscalationTier enum, not Sriti's internal tier numbering.
_TIER_NUM = {"tier_1": 1, "tier_2": 2, "tier_3": 3}


@dataclass
class _TierFloor:
    """Duck-typed `learning_adjustment` for engine.run() — only the three
    attributes the engine's closures actually read (everything else wants
    defaults via getattr(..., default)).

    `override_tier` forces the cascade to *start* there, implementing the
    escalation gate's "must reach at least this tier" floor: starting
    there already satisfies "at least", and the cascade can still escalate
    further on a quality failure exactly as it would unforced. None means
    no floor (tier_3 minimum — cascade picks its own initial tier).

    `requires_json_mode` tells the engine's structural quality check to
    validate JSON syntax and, combined with `complete()`'s
    quality_check_mode_override="structural" (set whenever this is True),
    to skip the semantic-similarity check entirely — see
    orchestration/model_client.py's `requires_json` docstring for why:
    a short structured-output prompt like detect_anomalies.py's often gets
    classified under an unrelated task_type, making similarity-to-anchor
    an unstable pass/fail signal for it.

    Second, less obvious effect: `requires_json_mode=True` also narrows
    `_get_capable_models()` to `supports_json_mode: true` models only (via
    engine.py's `_requires_structured_eff`/`_filter_for_capabilities`).
    Harmless today since every entry in models.yaml sets that flag, but the
    moment a tier without it is added, these call sites will silently skip
    that tier and escalate — a cost regression that won't be obvious from
    this docstring alone if you're not looking for it.
    """

    override_tier: int | None
    is_shadow: bool = False
    requires_json_mode: bool = False


async def complete(
    prompt: str | list[dict],
    minimum_tier: str,
    redis_client: Redis,
    cacheable: bool = True,
    requires_json: bool = False,
) -> tuple[str, SritiMetadata]:
    """Run one completion through classify -> cache -> cascade.

    `prompt` is a single user message — either plain text, or a
    list of content parts for a multimodal call (e.g., image extraction).
    `minimum_tier` is a tier value ("tier_1"/"tier_2"/"tier_3"). `cacheable=False` skips the
    semantic cache even for text-only prompts; `requires_json=True` makes
    the cascade's quality gate check the response structurally (valid
    JSON) instead of by semantic similarity to a topic anchor — see
    orchestration/model_client.py's docstring for when to set each.
    Returns (response_text, metadata).
    """
    messages = [{"role": "user", "content": prompt}]
    start = time.monotonic()

    # Multimodal requests never hit the semantic cache — the cache keys
    # off text-only content (content_to_str strips image/video parts), so
    # every vision call sharing the same prompt template would otherwise
    # collide on the exact same cache key regardless of which image was
    # actually sent. Found via a real end-to-end vision test: a second,
    # completely different photo returned the first photo's cached
    # extraction verbatim. Embedding base64 image/video data as part of
    # the cache key is nonsensical — the cache operates on semantic
    # text similarity, not image content.
    #
    # Callers can additionally opt a text-only prompt out via
    # cacheable=False — found necessary when two prompts using the same
    # template but different embedded numbers embedded similarly enough
    # to collide and return the wrong cached answer.
    has_media = any(isinstance(m.get("content"), list) for m in messages)
    skip_cache = has_media or not cacheable

    task_type, classification_score, user_vec = await asyncio.to_thread(
        classifier.classify, messages
    )

    cache = get_cache()
    cache_hit = None if skip_cache else await cache.get(messages=messages, task_type=task_type)
    if cache_hit is not None:
        latency_ms = round((time.monotonic() - start) * 1000, 2)
        telemetry.record_call(
            task_type=task_type,
            model_used=cache_hit.model_used,
            tier=cache_hit.model_tier,
            cache_hit=True,
            cost_usd=0.0,
            latency_ms=latency_ms,
            tokens_in=cache_hit.prompt_tokens,
            tokens_out=cache_hit.completion_tokens,
            escalated=False,
            routing_reason="cache_hit",
            quality_passed=None,
        )
        metadata = SritiMetadata(
            cache_hit=True,
            model_tier=cache_hit.model_tier,
            model_used=cache_hit.model_used,
            cost_usd=0.0,
            latency_ms=latency_ms,
            tokens_in=cache_hit.prompt_tokens,
            tokens_out=cache_hit.completion_tokens,
            routing_reason="cache_hit",
        )
        return cache_hit.content, metadata

    compression = await maybe_compress(messages, task_type)
    dispatch_messages = compression.compressed_messages

    tier_num = _TIER_NUM[minimum_tier]
    learning_adjustment = (
        _TierFloor(
            override_tier=tier_num if tier_num < 3 else None,
            requires_json_mode=requires_json,
        )
        if (tier_num < 3 or requires_json)
        else None
    )

    request = ChatCompletionRequest(model="auto", messages=dispatch_messages)
    result, updated_metadata = await engine_run(
        request,
        SritiMetadata(),
        redis_client,
        messages=dispatch_messages,
        task_type=task_type,
        quality_check_mode_override="structural" if requires_json else None,
        user_vec=user_vec,
        classification_score=classification_score,
        learning_adjustment=learning_adjustment,
    )

    latency_ms = round((time.monotonic() - start) * 1000, 2)
    updated_metadata = updated_metadata.model_copy(update={"latency_ms": latency_ms})

    telemetry.record_call(
        task_type=task_type,
        model_used=result.model_used,
        tier=updated_metadata.model_tier,
        cache_hit=False,
        cost_usd=result.cost_usd,
        latency_ms=latency_ms,
        tokens_in=result.prompt_tokens,
        tokens_out=result.completion_tokens,
        escalated=updated_metadata.escalated,
        routing_reason=updated_metadata.routing_reason,
        quality_passed=updated_metadata.quality_passed,
    )

    # Note: engine_run() already schedules the reliability outcome record
    # and the case-memory write internally (using its own precise internal
    # quality score, which isn't exposed on the returned metadata) — don't
    # duplicate those calls here, only the cache write is this layer's job.
    # Skipped for multimodal content or cacheable=False — see the note above.
    # Also skipped when quality_passed is explicitly False: the graceful-
    # degradation path can now return a response that failed the quality gate
    # (nowhere left to escalate to) instead of
    # raising. Caching that would serve a known-bad response to future
    # similar requests as though it had passed. quality_passed is None (not
    # evaluated, e.g. tier1) is still cacheable — only an explicit failure
    # blocks the write.
    if not skip_cache and updated_metadata.quality_passed is not False:
        cache.schedule_set(
            messages=messages,  # original, uncompressed — cache keys off intent, not dispatch form
            task_type=task_type,
            content=result.content,
            model_used=result.model_used,
            model_tier=updated_metadata.model_tier or tier_num,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
            finish_reason=result.finish_reason,
        )

    return result.content, updated_metadata
