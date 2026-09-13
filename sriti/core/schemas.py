"""Request/response schemas for the Sriti completion API.

Pure data shapes — no auth or tenant coupling. cascade/engine.py depends
on ChatCompletionRequest, RoutingStep, and SritiMetadata directly.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Supported escalation tier literals: tier_1 (frontier), tier_2 (balanced), tier_3 (local)
EscalationTierLiteral = Literal["tier_1", "tier_2", "tier_3"]

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: str
    content: str | list[dict] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message] = Field(..., min_length=1, max_length=256)
    temperature: float = 0.7
    max_tokens: Optional[int] = Field(default=None, le=16384)
    stream: bool = False
    response_format: dict | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages must contain at least one message")
        return v


class CompressRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=256)
    target_ratio: float = Field(default=0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class RoutingStep(BaseModel):
    step: str  # "classify", "complexity", "tier_select", "model_select",
    # "dispatch", "quality_pass", "quality_fail", "escalate",
    # "cache_hit", "cap_fallback", "bypass"
    detail: str  # Human-readable description
    value: Optional[str] = None  # Key metric (score, model name, latency)
    tier: Optional[int] = None
    metadata: Optional[dict] = None  # Per-step data: cost, tokens, response_snippet, quality_score, latency_ms


class SritiMetadata(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_saved: int = 0
    compression_ratio: float = 0.0
    compression_skipped_reason: Optional[str] = None
    cache_hit: bool = False
    similarity_score: Optional[float] = None
    model_tier: Optional[int] = None
    model_used: Optional[str] = None
    provider: Optional[str] = None
    routing_reason: Optional[str] = None
    complexity_score: float = 0.0  # 0.0-1.0 continuous complexity score
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    platform_overhead_ms: float = 0.0
    classify_ms: float = 0.0
    cache_lookup_ms: float = 0.0
    compression_ms: float = 0.0
    memory_lookup_ms: float = 0.0
    quality_check_ms: float = 0.0
    escalated: bool = False
    quality_passed: Optional[bool] = None
    aggregation_used: bool = False
    aggregation_candidates: int = 0
    aggregation_method: Optional[str] = None  # "best_pick"
    routing_trail: list[RoutingStep] = []
    # Unused on a per-box deployment (no traces table / workflow session
    # store here) — left in place because engine.py and telemetry.py
    # populate/pass them through unconditionally; harmless if always None.
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    step_index: Optional[int] = None
    step_type: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage = ChatCompletionUsage()
    sriti_metadata: SritiMetadata = SritiMetadata()


class CompressResponse(BaseModel):
    original_messages: list[dict]
    compressed_messages: list[dict]
    tokens_saved: int
    compression_ratio: float
    quality_score: float
