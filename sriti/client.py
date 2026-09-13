"""SritiClient — Python client for interacting with the Sriti service over HTTP.

Provides an async interface for invoking model completions through Sriti's
intelligent multi-tier cascading, semantic caching, and reliability-driven routing.
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from sriti.core.schemas import EscalationTierLiteral


class ModelResponse(BaseModel):
    """Response returned by model completions."""
    text: str
    tier_used: EscalationTierLiteral | str
    cost_usd: float | None = None
    latency_ms: float | None = None
    routing_reason: str | None = None


class SritiModelClient:
    """Async client for the Sriti service."""

    def __init__(self, base_url: str = "http://localhost:8100", timeout_s: float = 280.0):
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def complete(
        self,
        prompt: str | list[dict],
        minimum_tier: str | EscalationTierLiteral = "tier_3",
        cacheable: bool = True,
        requires_json: bool = False,
    ) -> ModelResponse:
        """Route prompt through Sriti's intelligent cascade."""
        tier_value = minimum_tier.value if hasattr(minimum_tier, "value") else str(minimum_tier)

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(
                f"{self._base_url}/complete",
                json={
                    "prompt": prompt,
                    "minimum_tier": tier_value,
                    "cacheable": cacheable,
                    "requires_json": requires_json,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return ModelResponse(
            text=data["text"],
            tier_used=data["tier_used"],
            cost_usd=data.get("cost_usd"),
            latency_ms=data.get("latency_ms"),
            routing_reason=data.get("routing_reason"),
        )
