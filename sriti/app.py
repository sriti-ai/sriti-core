"""Thin FastAPI wrapper around sriti/core/service.py.

Exposes two endpoints: a health check and one completion endpoint.
This is intentionally a minimal surface — no admin routes, no auth
middleware, no multi-tenant scaffolding. sriti/client.py talks to this
app over localhost HTTP and is the only expected caller in a standard
single-node deployment.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from redis.asyncio import Redis

from sriti.core.cache.semantic_cache import init_cache
from sriti.core.cascade import case_memory, classifier, expert_bank, reliability
from sriti.core.schemas import EscalationTierLiteral
from sriti.core.service import complete as run_complete
from sriti.core.settings import settings

logger = logging.getLogger(__name__)

_redis_client: Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis_client
    logger.info("Sriti: starting up (loading classifier, warming caches)...")

    classifier.warm_up()

    _redis_client = Redis.from_url(settings.redis_url, decode_responses=False)

    await init_cache(settings.redis_url)
    await case_memory.init_case_memory(_redis_client)
    reliability.init_reliability(_redis_client)
    expert_bank.seed_all()

    logger.info("Sriti: ready.")
    yield
    logger.info("Sriti: shutting down.")
    if _redis_client is not None:
        await _redis_client.close()


app = FastAPI(title="Sriti (per-box)", lifespan=lifespan)


class CompleteRequest(BaseModel):
    prompt: str | list[dict]
    minimum_tier: EscalationTierLiteral = "tier_3"
    cacheable: bool = True
    requires_json: bool = False


class CompleteResponse(BaseModel):
    text: str
    tier_used: EscalationTierLiteral
    cost_usd: float
    latency_ms: float
    routing_reason: str | None = None


_TIER_NUM_TO_NAME = {1: "tier_1", 2: "tier_2", 3: "tier_3"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/complete", response_model=CompleteResponse)
async def complete_endpoint(body: CompleteRequest) -> CompleteResponse:
    assert _redis_client is not None, "startup did not run — Redis client unset"
    text, metadata = await run_complete(
        body.prompt,
        body.minimum_tier,
        _redis_client,
        cacheable=body.cacheable,
        requires_json=body.requires_json,
    )
    return CompleteResponse(
        text=text,
        tier_used=_TIER_NUM_TO_NAME.get(metadata.model_tier or 3, "tier_3"),
        cost_usd=metadata.cost_usd,
        latency_ms=metadata.latency_ms,
        routing_reason=metadata.routing_reason,
    )
