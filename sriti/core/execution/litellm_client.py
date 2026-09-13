from __future__ import annotations

# ==============================================================================
# THE ONLY FILE IN THIS CODEBASE THAT IMPORTS LITELLM.
#
# Containment rule: `import litellm` must never appear outside this file.
# If we replace litellm (see SRI-47), only this file changes.
# The interface below (LLMResponse, call, stream) stays identical.
# ==============================================================================

import logging
import os
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import litellm

from sriti.core.settings import settings
from sriti.core.execution.retry import with_retry
from sriti.core.execution.streaming import sse_format

logger = logging.getLogger(__name__)

# Suppress litellm's verbose output — both the flag and the underlying loggers.
# suppress_debug_info alone is not enough; litellm resets it internally on some paths.
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)

# Drop unsupported params silently instead of raising UnsupportedParamsError.
# Some Bedrock models (DeepSeek V3.2, Qwen, etc.) don't accept temperature,
# top_p, or other OpenAI-compat params. Dropping them is always preferable to
# failing the call — the model just uses its provider default.
litellm.drop_params = True


def _inject_provider_config() -> None:
    """
    Push credentials from settings into the env vars litellm reads.
    Called once at module import time.
    """
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.anthropic_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    if settings.google_api_key:
        os.environ.setdefault("GEMINI_API_KEY", settings.google_api_key)
    if settings.groq_api_key:
        os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
    # Phase 2 stubs
    if settings.fireworks_api_key:
        os.environ.setdefault("FIREWORKS_AI_API_KEY", settings.fireworks_api_key)
    if settings.xai_api_key:
        os.environ.setdefault("XAI_API_KEY", settings.xai_api_key)
    if settings.perplexity_api_key:
        os.environ.setdefault("PERPLEXITYAI_API_KEY", settings.perplexity_api_key)
    if settings.nebius_api_key:
        os.environ.setdefault("NEBIUS_API_KEY", settings.nebius_api_key)
    if settings.aws_access_key_id:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.aws_access_key_id)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.aws_secret_access_key)
        os.environ.setdefault("AWS_REGION_NAME", settings.aws_region)


_inject_provider_config()


@dataclass
class LLMResponse:
    """Provider-agnostic response. Nothing above this layer sees litellm types."""

    content: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    inference_latency_ms: float = 0.0
    finish_reason: str | None = None
    response_id: str | None = None
    created: int | None = None
    tool_calls: list[dict] | None = None


def _build_kwargs(
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    stream: bool,
    timeout: float | None = None,
    response_format: dict | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> dict:
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    # Request timeout — prevents hanging on slow providers (Bedrock cold starts, etc.)
    # Default: 90s for both streaming and non-streaming.
    # Frontier models with large payloads (GPT-4o + JSON) routinely need 60-80s.
    if timeout is not None:
        kwargs["timeout"] = timeout
    else:
        kwargs["timeout"] = 90.0

    # Bedrock cross-region inference profiles (CRIPs): route requests across
    # multiple US regions for higher availability. Using us.* on a model WITHOUT
    # a CRIP causes 400 "invalid model identifier".
    #
    # This is an EXACT set of model IDs with active CRIPs, sourced from
    # `aws bedrock list-inference-profiles --query '...[?status==ACTIVE]'`.
    # Prefix-matching by provider is unsafe — not all models from a provider
    # have CRIPs (e.g. deepseek.r1 has one but deepseek.v3.2 does not).
    #
    # Models work fine WITHOUT us.* prefix (single-region routing), so omitting
    # a model here is safe — it just loses cross-region load balancing.
    _BEDROCK_HAS_CRIP: frozenset[str] = frozenset({
        # anthropic (Claude)
        "anthropic.claude-opus-4-6-v1",
        "anthropic.claude-sonnet-4-6",
        "anthropic.claude-opus-4-5-20251101-v1:0",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.claude-opus-4-20250514-v1:0",
        "anthropic.claude-sonnet-4-20250514-v1:0",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-3-5-haiku-20241022-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        # meta (Llama)
        "meta.llama4-maverick-17b-instruct-v1:0",
        "meta.llama4-scout-17b-instruct-v1:0",
        "meta.llama3-3-70b-instruct-v1:0",
        "meta.llama3-2-90b-instruct-v1:0",
        "meta.llama3-2-11b-instruct-v1:0",
        # amazon (Nova)
        "amazon.nova-premier-v1:0",
        "amazon.nova-pro-v1:0",
        "amazon.nova-lite-v1:0",
        "amazon.nova-micro-v1:0",
        # deepseek (R1 only — V3.2 has no CRIP)
        "deepseek.r1-v1:0",
        # mistral (Pixtral Large only)
        "mistral.pixtral-large-2502-v1:0",
    })
    if model.startswith("bedrock/") and not model.startswith("bedrock/us."):
        _, model_part = model.split("/", 1)
        if model_part in _BEDROCK_HAS_CRIP:
            kwargs["model"] = f"bedrock/us.{model_part}"
    return kwargs


def _extract_cost(response: litellm.ModelResponse, original_model: str) -> float:
    # litellm strips the provider prefix from response.model
    # (e.g. "groq/llama-3.1-8b-instant" → "llama-3.1-8b-instant"), which breaks
    # the pricing lookup. Always pass the original model string explicitly.
    try:
        return litellm.completion_cost(
            completion_response=response,
            model=original_model,
        ) or 0.0
    except Exception:
        return 0.0


async def call(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_retries: int = 3,
    response_format: dict | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> LLMResponse:
    """
    Single async LLM call with automatic retry on transient errors.

    Args:
        model: LiteLLM model string, e.g. "gpt-4o-mini", "groq/llama-3.1-8b-instant",
               "bedrock/meta.llama3-8b-instruct-v1:0"
        messages: OpenAI-format message list [{"role": ..., "content": ...}]
        timeout: Request timeout in seconds. Default: 90s.
        max_retries: Max retry attempts. Use 1 for bypass/passthrough (no retry).
    """
    kwargs = _build_kwargs(
        model, messages, temperature, max_tokens, stream=False,
        timeout=timeout, response_format=response_format,
        tools=tools, tool_choice=tool_choice,
    )

    t0 = time.perf_counter()

    async def _do_call() -> litellm.ModelResponse:
        return await litellm.acompletion(**kwargs)

    response: litellm.ModelResponse = await with_retry(_do_call, max_attempts=max_retries)
    inference_latency_ms = (time.perf_counter() - t0) * 1000.0

    usage = response.usage or {}
    msg = response.choices[0].message

    # Extract tool_calls from the response message
    raw_tool_calls = getattr(msg, "tool_calls", None)
    tool_calls_out = None
    if raw_tool_calls:
        tool_calls_out = [
            {
                "id": tc.id,
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in raw_tool_calls
        ]

    return LLMResponse(
        content=msg.content or "",
        model_used=model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cost_usd=_extract_cost(response, model),
        inference_latency_ms=round(inference_latency_ms, 2),
        finish_reason=response.choices[0].finish_reason,
        response_id=getattr(response, "id", None),
        created=getattr(response, "created", None) or int(time.time()),
        tool_calls=tool_calls_out,
    )


@dataclass
class EmbeddingResponse:
    """Provider-agnostic embedding response."""

    embeddings: list[list[float]]
    model_used: str
    prompt_tokens: int
    cost_usd: float
    inference_latency_ms: float = 0.0


async def embed(
    model: str,
    input: str | list[str],
) -> EmbeddingResponse:
    """Async embedding call via LiteLLM."""
    t0 = time.perf_counter()

    async def _do_embed():
        return await litellm.aembedding(model=model, input=input)

    response = await with_retry(_do_embed)
    inference_latency_ms = (time.perf_counter() - t0) * 1000.0

    data = response.data or []
    embeddings = [item["embedding"] for item in data]
    usage = response.usage or {}
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0

    try:
        cost = litellm.completion_cost(completion_response=response, model=model) or 0.0
    except Exception:
        cost = 0.0

    return EmbeddingResponse(
        embeddings=embeddings,
        model_used=model,
        prompt_tokens=prompt_tokens,
        cost_usd=cost,
        inference_latency_ms=round(inference_latency_ms, 2),
    )


async def stream(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float | None = None,
    on_complete=None,
    response_format: dict | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> AsyncGenerator[str, None]:
    """
    Streaming LLM call. Returns an async generator of SSE-formatted strings
    ready to be passed directly into FastAPI's StreamingResponse.
    """
    kwargs = _build_kwargs(
        model, messages, temperature, max_tokens, stream=True,
        timeout=timeout, response_format=response_format,
        tools=tools, tool_choice=tool_choice,
    )
    async def _do_stream():
        return await litellm.acompletion(**kwargs)

    litellm_stream = await with_retry(_do_stream, max_attempts=3)
    return sse_format(litellm_stream, model=model, on_complete=on_complete)


async def raw_stream(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> AsyncGenerator:
    """
    Returns the raw litellm async stream (not SSE-formatted).
    Use this when you need to apply custom formatting (e.g. Responses API events).
    """
    kwargs = _build_kwargs(
        model, messages, temperature, max_tokens, stream=True,
        timeout=timeout, tools=tools, tool_choice=tool_choice,
    )

    async def _do_stream():
        return await litellm.acompletion(**kwargs)

    return await with_retry(_do_stream, max_attempts=3)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text using litellm's token counter. Fail-safe: returns len//4."""
    try:
        return litellm.token_counter(model=model, text=text)
    except Exception:
        return len(text) // 4
