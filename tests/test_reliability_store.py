from __future__ import annotations

import pytest

from sriti.core.cascade import reliability_store
from tests.fake_redis import FakeRedis

pytestmark = pytest.mark.asyncio


async def test_first_outcome_seeds_profile_exactly():
    redis_client = FakeRedis()
    await reliability_store.record_outcome(
        redis_client, model="m1", task_type="qa",
        quality_passed=True, quality_score=0.9, latency_ms=500.0, cost_usd=0.001,
    )
    profiles = await reliability_store.load_all_profiles(redis_client)
    profile = profiles[("m1", "qa")]
    assert profile["sample_count"] == 1
    assert profile["success_rate"] == 1.0
    assert profile["avg_quality"] == 0.9


async def test_second_outcome_updates_via_ema_not_overwrite():
    redis_client = FakeRedis()
    await reliability_store.record_outcome(
        redis_client, model="m1", task_type="qa",
        quality_passed=True, quality_score=1.0, latency_ms=500.0, cost_usd=0.001,
    )
    await reliability_store.record_outcome(
        redis_client, model="m1", task_type="qa",
        quality_passed=False, quality_score=0.0, latency_ms=500.0, cost_usd=0.001,
    )
    profiles = await reliability_store.load_all_profiles(redis_client)
    profile = profiles[("m1", "qa")]
    assert profile["sample_count"] == 2
    # EMA alpha=0.2: 0.8*1.0 + 0.2*0.0 = 0.8, not a flat overwrite to 0.0/0.5
    assert 0.79 < profile["success_rate"] < 0.81


async def test_models_with_colons_in_name_round_trip_correctly():
    redis_client = FakeRedis()
    await reliability_store.record_outcome(
        redis_client, model="ollama_chat/qwen2.5:7b", task_type="summarization",
        quality_passed=True, quality_score=0.8, latency_ms=1200.0, cost_usd=0.0,
    )
    profiles = await reliability_store.load_all_profiles(redis_client)
    assert ("ollama_chat/qwen2.5:7b", "summarization") in profiles


async def test_load_all_profiles_is_fail_open_on_error():
    class BrokenRedis(FakeRedis):
        async def scan_iter(self, match: str = "*"):
            raise ConnectionError("boom")
            yield  # pragma: no cover

    profiles = await reliability_store.load_all_profiles(BrokenRedis())
    assert profiles == {}
