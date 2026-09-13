"""Structured per-call telemetry: cost, latency, model used, and routing
reason emitted as structured JSON log lines.

A single-node deployment has no cross-tenant analytics need and no ops
team querying a traces table — a log line per call, tailed or shipped to
whatever log aggregation is available, is sufficient. Swap this for a
real sink (SQLite file, a metrics push) if querying historical telemetry
becomes a concrete requirement — this file is the one seam to change.
"""

from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger("sriti.telemetry")


def record_call(
    *,
    task_type: str,
    model_used: str,
    tier: int | None,
    cache_hit: bool,
    cost_usd: float,
    latency_ms: float,
    tokens_in: int,
    tokens_out: int,
    escalated: bool,
    routing_reason: str | None,
    quality_passed: bool | None,
) -> None:
    """One structured JSON log line per completed call. Never raises —
    telemetry must not affect the hot path."""
    try:
        record = {
            "ts": time.time(),
            "task_type": task_type,
            "model_used": model_used,
            "tier": tier,
            "cache_hit": cache_hit,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "escalated": escalated,
            "routing_reason": routing_reason,
            "quality_passed": quality_passed,
        }
        logger.info(json.dumps(record))
    except Exception:
        logger.warning("telemetry: record_call failed (fail-open)", exc_info=True)
