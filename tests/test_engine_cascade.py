"""Tests for sriti/core/cascade/engine.py — the highest-consequence code
in the codebase, verified here via regression tests rather than live replay.

Two things are covered:

1. `_quality_check_structural` directly (pure function, no mocking needed) —
   requires_json=True must bypass the hedging/min-words checks (calibrated
   for prose, not JSON), or a short-but-correct JSON response like "[]" gets
   wrongly escalated to a more expensive tier.

2. `run()`'s graceful-degradation paths, via a minimal fake harness (FakeRedis
   + a scripted fake litellm_client.call + tier_overrides + a real _TierFloor
   learning_adjustment to force the starting tier). This covers two distinct
   bugs: (a) an unreachable tier discarding a perfectly good earlier result
   instead of degrading to it, and (b) a second, previously-unfixed site with
   the same shape — triggered when escalating lands on a tier with an empty
   model pool. Both cases are regression-tested, plus the "nothing to fall
   back to" path that must still raise.

Everything needed to run `run()` deterministically without a real Redis/LLM
provider is faked or forced via existing engine.py extension points
(tier_overrides, learning_adjustment, quality_check_mode_override="structural"
to avoid the embedding-based quality check's model-loading dependency,
task_type passed explicitly to skip classify()). Every Redis call this path
can reach is wrapped fail-open in its own module, so FakeRedis
(get/set/ttl/scan_iter only) is sufficient.
"""

from __future__ import annotations

import pytest

from sriti.core.cascade import engine, policy
from sriti.core.cascade.engine import _quality_check_structural
from sriti.core.cascade.policy import TierConfig
from sriti.core.execution import litellm_client
from sriti.core.execution.litellm_client import LLMResponse
from sriti.core.schemas import ChatCompletionRequest, Message, SritiMetadata
from sriti.core.service import _TierFloor
from tests.fake_redis import FakeRedis

# Not module-level pytestmark: this file mixes sync tests (the pure
# _quality_check_structural checks) with async ones (the run() cascade
# tests) — asyncio_mode=auto in pytest.ini already picks up the async ones
# without a marker, and applying the marker module-wide warns on every sync
# test ("marked with asyncio but it is not an async function").


# ---------------------------------------------------------------------------
# 1. _quality_check_structural — pure function, direct tests
# ---------------------------------------------------------------------------

_TIER2 = TierConfig(tier_num=2, model="fake/tier2", quality_threshold=0.55, max_latency_ms=15000, name="tier2")
_TIER1 = TierConfig(tier_num=1, model="fake/tier1", quality_threshold=None, max_latency_ms=30000, name="tier1")


def test_empty_response_fails_regardless_of_requires_json():
    assert _quality_check_structural("", "stop", "customer_support", _TIER2, requires_json=True) is False
    assert _quality_check_structural("", "stop", "customer_support", _TIER2, requires_json=False) is False


def test_tier1_frontier_always_passes():
    # quality_threshold is None for tier1 — no structural signal should override "always pass"
    assert _quality_check_structural("no", "stop", "code", _TIER1, requires_json=False) is True


def test_truncated_response_fails():
    assert _quality_check_structural("some text", "length", "customer_support", _TIER2) is False


def test_hedging_prose_fails_when_not_requires_json():
    text = "I'm not sure I can help with that specific request."
    assert _quality_check_structural(text, "stop", "customer_support", _TIER2, requires_json=False) is False


def test_hedging_phrase_inside_json_value_is_not_flagged_when_requires_json():
    # A hedging phrase appearing inside a JSON string value is legitimate content,
    # not a refusal — must not be caught by the prose-oriented hedging check.
    # Not-False rather than True: this function
    # returns None ("no strong signal, fall through to embedding check") when
    # nothing fails structurally — engine.py's structural-mode caller treats
    # None the same as True (`passed = structural if structural is not None
    # else True`), so the real regression signal is "did NOT return False".
    text = '{"note": "I am not sure this is a sale or a bill"}'
    assert _quality_check_structural(text, "stop", "customer_support", _TIER2, requires_json=True) is not False


def test_short_json_below_min_words_passes_when_requires_json():
    # Regression test: a short, correct
    # JSON response like "[]" (detect_anomalies' common "no anomalies" answer)
    # must not fail structurally just because it's short. task_type=code has
    # a min_words floor of 30 in _MIN_WORDS_BY_TASK — before the fix, this
    # exact case would escalate unnecessarily (or hard-fail if tier1 has no key).
    assert _quality_check_structural("[]", "stop", "code", _TIER2, requires_json=True) is not False
    assert _quality_check_structural("null", "stop", "code", _TIER2, requires_json=True) is not False


def test_short_response_below_min_words_fails_when_not_requires_json():
    # Same input, requires_json=False — the min-words floor must still apply
    # for genuine prose tasks. Confirms the fix is scoped to requires_json,
    # not a blanket removal of the check.
    assert _quality_check_structural("nope", "stop", "code", _TIER2, requires_json=False) is False


def test_invalid_json_fails_when_requires_json():
    assert _quality_check_structural("{not valid json", "stop", "code", _TIER2, requires_json=True) is False


def test_valid_json_object_passes_when_requires_json():
    assert _quality_check_structural('{"a": 1}', "stop", "code", _TIER2, requires_json=True) is not False


# ---------------------------------------------------------------------------
# 2. run() — graceful degradation, via a fake litellm_client.call
# ---------------------------------------------------------------------------

_MODELS_BY_TIER = {
    3: ["fake-tier3-model"],
    2: ["fake-tier2-model"],
    1: [],  # deliberately empty — simulates "no reachable/configured tier1 model"
}


def _fake_get_available_models_for_tier(tier_num: int) -> list[str]:
    return list(_MODELS_BY_TIER.get(tier_num, []))


def _tier_overrides() -> dict[int, TierConfig]:
    return {
        3: TierConfig(tier_num=3, model="fake-tier3-model", quality_threshold=0.70, max_latency_ms=240000, name="tier3_fast"),
        2: TierConfig(tier_num=2, model="fake-tier2-model", quality_threshold=0.55, max_latency_ms=15000, name="tier2_balanced"),
        1: TierConfig(tier_num=1, model="fake-tier1-model", quality_threshold=None, max_latency_ms=30000, name="tier1_frontier"),
    }


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="auto", messages=[Message(role="user", content="hello")])


class _NonRetryableError(Exception):
    """A class name deliberately NOT in engine._PROVIDER_ERROR_NAMES, so the
    cascade treats it as non-retryable and escalates immediately instead of
    trying same-tier fallback models first — matches how a real auth failure
    (e.g. no tier1 API key configured) behaves."""


def _ok_response(model: str, content: str = "a valid response with plenty of words in it") -> LLMResponse:
    return LLMResponse(
        content=content, model_used=model, prompt_tokens=10, completion_tokens=10,
        cost_usd=0.0, inference_latency_ms=5.0, finish_reason="stop",
    )


async def _no_aggregation(*args, **kwargs):
    return None


@pytest.fixture(autouse=True)
def _patch_model_pool(monkeypatch):
    monkeypatch.setattr(engine, "_get_available_models_for_tier", _fake_get_available_models_for_tier)
    # attempt_aggregation is a separate multi-expert-retry mechanism that
    # picks "other same-tier models" via its own get_aggregation_candidates(),
    # independent of _get_available_models_for_tier — so it isn't constrained
    # by the fake pool above and will happily dispatch to a real catalog
    # model name. These tests aren't exercising aggregation, only the
    # escalation/degrade logic below it, so disable it here to keep every
    # dispatch going through the faked litellm_client.call with a model name
    # this test controls.
    monkeypatch.setattr(engine, "attempt_aggregation", _no_aggregation)


async def _run(monkeypatch, call_fn, *, task_type="customer_support"):
    monkeypatch.setattr(litellm_client, "call", call_fn)
    force_tier3 = _TierFloor(override_tier=3, requires_json_mode=False)
    return await engine.run(
        _request(),
        SritiMetadata(),
        FakeRedis(),
        messages=[{"role": "user", "content": "hello"}],
        task_type=task_type,
        tier_overrides=_tier_overrides(),
        quality_check_mode_override="structural",
        learning_adjustment=force_tier3,
    )


async def test_normal_success_no_escalation(monkeypatch):
    async def call_fn(model, messages, **kwargs):
        return _ok_response(model)

    result, metadata = await _run(monkeypatch, call_fn)
    assert result.model_used == "fake-tier3-model"
    assert metadata.model_tier == 3
    assert metadata.escalated is False
    assert metadata.quality_passed is True


async def test_quality_failure_escalates_to_next_tier(monkeypatch):
    async def call_fn(model, messages, **kwargs):
        if model == "fake-tier3-model":
            return _ok_response(model, content="")  # empty → fails structural check
        return _ok_response(model)

    result, metadata = await _run(monkeypatch, call_fn)
    assert result.model_used == "fake-tier2-model"
    assert metadata.model_tier == 2
    assert metadata.escalated is True


async def test_tier_unreachable_at_frontier_degrades_to_last_success(monkeypatch):
    """Regression test: tier3 succeeds (but fails quality, so the cascade tries
    to escalate), tier2 also fails quality, escalates to tier1 — tier1's
    dispatch raises (e.g. no API key) and tier1 has nowhere further to
    escalate to (next_tier is None because current_tier.tier_num == 1).
    Must serve tier3's/tier2's last real result instead of raising."""
    async def call_fn(model, messages, **kwargs):
        if model == "fake-tier1-model":
            raise _NonRetryableError("no credentials configured")
        return _ok_response(model, content="")  # tier3, tier2 both fail quality → escalate

    # tier1 needs a capable model to reach the dispatch-raises branch at all
    # (as opposed to the no-capable-models branch tested separately below).
    local_models = dict(_MODELS_BY_TIER)
    local_models[1] = ["fake-tier1-model"]

    def fake_pool(tier_num: int) -> list[str]:
        return list(local_models.get(tier_num, []))

    monkeypatch.setattr(engine, "_get_available_models_for_tier", fake_pool)
    result, metadata = await _run(monkeypatch, call_fn)

    # Degraded to the last real (quality-failed but genuine) response rather
    # than raising — same as the request never having tried tier1 at all.
    assert result.model_used == "fake-tier2-model"
    assert metadata.quality_passed is False


async def test_no_capable_models_in_remaining_tier_degrades_to_last_success(monkeypatch):
    """Regression test for a second degradation bug: triggered when escalating
    (after a dispatch exception, not a quality failure) lands on a tier with
    an empty model pool and there's no further tier to skip to. tier3 succeeds
    and is the real answer sitting in `result`; tier3's quality check fails so
    the cascade escalates to tier2; tier2's dispatch raises a non-retryable
    error; escalating from tier2 lands on tier1, whose model pool is empty
    (the default _MODELS_BY_TIER fixture, unmodified) — must degrade to
    tier3's response instead of discarding it via `raise`."""
    async def call_fn(model, messages, **kwargs):
        if model == "fake-tier3-model":
            return _ok_response(model, content="")  # fails quality → escalate to tier2
        if model == "fake-tier2-model":
            raise _NonRetryableError("provider misconfigured")
        raise AssertionError(f"should never dispatch to {model}")

    result, metadata = await _run(monkeypatch, call_fn)

    assert result.model_used == "fake-tier3-model"
    assert metadata.model_tier == 3


async def test_nothing_to_fall_back_to_still_raises(monkeypatch):
    """Sanity check the other direction: when there was never a successful
    dispatch to fall back to, the cascade must still raise rather than
    silently swallow a real failure. Forces tier1 as the *starting* tier
    (not an escalation target) so there's no earlier result at all."""
    async def call_fn(model, messages, **kwargs):
        raise _NonRetryableError("no credentials configured")

    local_models = {1: ["fake-tier1-model"], 2: [], 3: []}

    def fake_pool(tier_num: int) -> list[str]:
        return list(local_models.get(tier_num, []))

    monkeypatch.setattr(engine, "_get_available_models_for_tier", fake_pool)
    monkeypatch.setattr(litellm_client, "call", call_fn)

    force_tier1 = _TierFloor(override_tier=1, requires_json_mode=False)
    with pytest.raises(_NonRetryableError):
        await engine.run(
            _request(),
            SritiMetadata(),
            FakeRedis(),
            messages=[{"role": "user", "content": "hello"}],
            task_type="customer_support",
            tier_overrides=_tier_overrides(),
            quality_check_mode_override="structural",
            learning_adjustment=force_tier1,
        )


# ---------------------------------------------------------------------------
# 3. Workstream E1 — Core cascade behavior tests
# ---------------------------------------------------------------------------

async def test_tier_routing_minimum_tier_2_never_calls_tier_3(monkeypatch):
    """E1 Behavior 1: Given a request flagged minimum_tier=2 by the rule gate,
    the cascade starts at tier 2 and never calls a tier-3 model."""
    dispatched_models = []

    async def call_fn(model, messages, **kwargs):
        dispatched_models.append(model)
        return _ok_response(model)

    monkeypatch.setattr(litellm_client, "call", call_fn)
    floor_tier2 = _TierFloor(override_tier=2, requires_json_mode=False)

    result, metadata = await engine.run(
        _request(),
        SritiMetadata(),
        FakeRedis(),
        messages=[{"role": "user", "content": "hello"}],
        task_type="customer_support",
        tier_overrides=_tier_overrides(),
        quality_check_mode_override="structural",
        learning_adjustment=floor_tier2,
    )

    assert result.model_used == "fake-tier2-model"
    assert metadata.model_tier == 2
    assert "fake-tier3-model" not in dispatched_models
    assert dispatched_models == ["fake-tier2-model"]


async def test_graceful_degradation_tier2_fails_serves_tier3_result(monkeypatch):
    """E1 Behavior 2: If tier-2 fails with an exception, the cascade serves
    the last successful tier response (tier-3) rather than raising."""
    dispatched_models = []

    async def call_fn(model, messages, **kwargs):
        dispatched_models.append(model)
        if model == "fake-tier3-model":
            # Return empty response to fail structural QC and trigger escalation
            return _ok_response(model, content="")
        if model == "fake-tier2-model":
            raise _NonRetryableError("tier 2 provider down")
        return _ok_response(model)

    monkeypatch.setattr(litellm_client, "call", call_fn)
    force_tier3 = _TierFloor(override_tier=3, requires_json_mode=False)

    result, metadata = await engine.run(
        _request(),
        SritiMetadata(),
        FakeRedis(),
        messages=[{"role": "user", "content": "hello"}],
        task_type="customer_support",
        tier_overrides=_tier_overrides(),
        quality_check_mode_override="structural",
        learning_adjustment=force_tier3,
    )

    assert "fake-tier3-model" in dispatched_models
    assert "fake-tier2-model" in dispatched_models
    assert result.model_used == "fake-tier3-model"
    assert metadata.quality_passed is False


async def test_frontier_traffic_cap_exhausted_does_not_call_tier1(monkeypatch):
    """E1 Behavior 3: If the tier-1 budget is exhausted, the cascade does
    not call tier-1."""
    from unittest.mock import AsyncMock
    dispatched_models = []

    async def call_fn(model, messages, **kwargs):
        dispatched_models.append(model)
        return _ok_response(model)

    local_models = {
        1: ["fake-tier1-model"],
        2: ["fake-tier2-model"],
        3: ["fake-tier3-model"],
    }
    monkeypatch.setattr(engine, "_get_available_models_for_tier", lambda t: list(local_models.get(t, [])))
    monkeypatch.setattr(litellm_client, "call", call_fn)
    # Mock check_and_record_frontier to return False (budget cap exhausted)
    monkeypatch.setattr(policy, "check_and_record_frontier", AsyncMock(return_value=False))

    # Force initial tier to tier 1
    force_tier1 = _TierFloor(override_tier=1, requires_json_mode=False)

    result, metadata = await engine.run(
        _request(),
        SritiMetadata(),
        FakeRedis(),
        messages=[{"role": "user", "content": "hello"}],
        task_type="customer_support",
        tier_overrides=_tier_overrides(),
        quality_check_mode_override="structural",
        learning_adjustment=force_tier1,
        frontier_cap_override=0.0,
    )

    assert result.model_used == "fake-tier2-model"
    assert metadata.model_tier == 2
    assert "fake-tier1-model" not in dispatched_models
    assert dispatched_models == ["fake-tier2-model"]


async def test_semantic_cache_hit_makes_zero_model_calls(monkeypatch):
    """E1 Behavior 4: If the semantic cache returns a hit, the cascade returns
    it without any model calls."""
    from unittest.mock import AsyncMock, MagicMock
    from sriti.core import service
    from sriti.core.cache.semantic_cache import CacheHit

    call_mock = AsyncMock()
    monkeypatch.setattr(litellm_client, "call", call_mock)

    mock_cache = MagicMock()
    cached_entry = CacheHit(
        content="cached answer",
        model_used="cached-model",
        model_tier=3,
        similarity_score=0.98,
        cost_usd=0.0,
        prompt_tokens=15,
        completion_tokens=20,
        finish_reason="stop",
        task_type="customer_support",
    )
    mock_cache.get = AsyncMock(return_value=cached_entry)
    monkeypatch.setattr(service, "get_cache", lambda: mock_cache)
    monkeypatch.setattr(service.classifier, "classify", lambda msgs: ("customer_support", 0.95, None))

    resp_text, meta = await service.complete(
        "any prompt",
        minimum_tier="tier_3",
        redis_client=FakeRedis(),
        cacheable=True,
    )

    assert resp_text == "cached answer"
    assert meta.cache_hit is True
    assert call_mock.call_count == 0


# ---------------------------------------------------------------------------
# 4. Workstream E3 — Reliability learner outcome recording on degradation
# ---------------------------------------------------------------------------

async def test_graceful_degradation_records_failure_outcome_in_reliability_learner(monkeypatch):
    """E3: When the cascade takes the graceful-degradation path (a tier fails,
    serves last successful result), an explicit failure outcome is recorded for
    the failed tier so the reliability learner knows that model/tier failed."""
    from sriti.core.cascade.reliability import Outcome

    recorded_outcomes: list[Outcome] = []
    monkeypatch.setattr(engine, "schedule_record_outcome", lambda o: recorded_outcomes.append(o))

    async def call_fn(model, messages, **kwargs):
        if model == "fake-tier3-model":
            # Return empty response to fail structural QC and trigger escalation
            return _ok_response(model, content="")
        if model == "fake-tier2-model":
            raise _NonRetryableError("tier 2 provider down")
        return _ok_response(model)

    monkeypatch.setattr(litellm_client, "call", call_fn)
    force_tier3 = _TierFloor(override_tier=3, requires_json_mode=False)

    result, metadata = await engine.run(
        _request(),
        SritiMetadata(),
        FakeRedis(),
        messages=[{"role": "user", "content": "hello"}],
        task_type="customer_support",
        tier_overrides=_tier_overrides(),
        quality_check_mode_override="structural",
        learning_adjustment=force_tier3,
    )

    assert result.model_used == "fake-tier3-model"

    # Verify that an outcome was recorded for fake-tier2-model with quality_passed=False
    failed_outcomes = [
        o for o in recorded_outcomes
        if o.model == "fake-tier2-model" and o.quality_passed is False
    ]
    assert len(failed_outcomes) == 1
    assert failed_outcomes[0].tier_num == 2
    assert failed_outcomes[0].escalated is True


async def test_graceful_degradation_no_capable_models_records_failure_outcome(monkeypatch):
    """E3: When escalating after a dispatch exception lands on a tier with no capable
    models, the failed model's failure outcome is recorded before degrading."""
    from sriti.core.cascade.reliability import Outcome

    recorded_outcomes: list[Outcome] = []
    monkeypatch.setattr(engine, "schedule_record_outcome", lambda o: recorded_outcomes.append(o))

    async def call_fn(model, messages, **kwargs):
        if model == "fake-tier3-model":
            return _ok_response(model, content="")
        if model == "fake-tier2-model":
            raise _NonRetryableError("tier 2 provider down")
        raise AssertionError(f"unexpected dispatch to {model}")

    result, metadata = await _run(monkeypatch, call_fn)

    assert result.model_used == "fake-tier3-model"
    failed_outcomes = [
        o for o in recorded_outcomes
        if o.model == "fake-tier2-model" and o.quality_passed is False
    ]
    assert len(failed_outcomes) == 1
    assert failed_outcomes[0].tier_num == 2
