import pytest
from sriti.client import SritiModelClient, ModelResponse


def test_model_response_instantiation():
    resp = ModelResponse(
        text="Hello world",
        tier_used="tier_3",
        cost_usd=0.0,
        latency_ms=12.5,
        routing_reason="cache_hit",
    )
    assert resp.text == "Hello world"
    assert resp.tier_used == "tier_3"
    assert resp.cost_usd == 0.0
    assert resp.routing_reason == "cache_hit"


def test_client_init():
    client = SritiModelClient(base_url="http://localhost:8100", timeout_s=30.0)
    assert client._base_url == "http://localhost:8100"
    assert client._timeout_s == 30.0
