from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class StreamMetrics:
    """Accumulated metrics from a completed stream."""
    chunks: int = 0
    content_length: int = 0
    start_ts: float = field(default_factory=time.time)
    end_ts: float = 0.0

    @property
    def latency_ms(self) -> float:
        return round((self.end_ts - self.start_ts) * 1000.0, 2) if self.end_ts else 0.0


async def sse_format(
    litellm_stream: AsyncGenerator,
    model: str,
    response_id: str | None = None,
    on_complete: Callable[[StreamMetrics], None] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Format a litellm async stream into OpenAI-compatible SSE strings.

    Yields strings like:  data: {...}\n\n
    Terminates with:      data: [DONE]\n\n

    When on_complete is provided, it is called after the stream finishes
    with accumulated StreamMetrics (chunk count, content length, latency).

    This module has zero litellm imports — it only handles formatting.
    The stream is passed in from litellm_client.py.
    """
    rid = response_id or f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    metrics = StreamMetrics(start_ts=time.time())

    try:
        async for chunk in litellm_stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None) or ""
            metrics.chunks += 1
            metrics.content_length += len(content)

            delta_dict: dict = {
                "role": getattr(delta, "role", None),
                "content": content,
            }

            # Include tool_calls in delta when present (streaming tool calling)
            raw_tool_calls = getattr(delta, "tool_calls", None)
            if raw_tool_calls:
                delta_dict["tool_calls"] = [
                    {
                        "index": getattr(tc, "index", i),
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", None),
                        "function": {
                            "name": getattr(tc.function, "name", None) if tc.function else None,
                            "arguments": getattr(tc.function, "arguments", None) if tc.function else None,
                        },
                    }
                    for i, tc in enumerate(raw_tool_calls)
                ]

            payload = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta_dict,
                        "finish_reason": chunk.choices[0].finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"
    finally:
        # Always fire on_complete — even on client disconnect or error.
        # Code after the last yield in an async generator may never execute
        # if the consumer stops iterating, but finally blocks always run.
        metrics.end_ts = metrics.end_ts or time.time()
        if on_complete is not None:
            try:
                on_complete(metrics)
            except Exception:
                import logging
                logging.getLogger(__name__).debug(
                    "on_complete callback failed (ignored)", exc_info=True,
                )

    yield "data: [DONE]\n\n"


async def sse_format_responses(
    litellm_stream: AsyncGenerator,
    model: str,
    response_id: str | None = None,
    on_complete: Callable[[StreamMetrics], None] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Format a litellm async stream into OpenAI Responses API SSE events.

    Event sequence:
      response.created → response.output_item.added → response.output_text.delta* →
      response.output_text.done → response.completed
    """
    rid = response_id or f"resp-{uuid.uuid4().hex}"
    created = int(time.time())
    metrics = StreamMetrics(start_ts=time.time())
    full_text = ""

    # response.created
    yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': {'id': rid, 'object': 'response', 'created_at': created, 'model': model, 'output': []}})}\n\n"

    # response.output_item.added
    yield f"event: response.output_item.added\ndata: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item': {'type': 'message', 'role': 'assistant', 'content': []}})}\n\n"

    try:
        async for chunk in litellm_stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None) or ""
            metrics.chunks += 1
            metrics.content_length += len(content)
            full_text += content

            if content:
                yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'output_index': 0, 'content_index': 0, 'delta': content})}\n\n"
    finally:
        metrics.end_ts = metrics.end_ts or time.time()
        if on_complete is not None:
            try:
                on_complete(metrics)
            except Exception:
                import logging
                logging.getLogger(__name__).debug(
                    "on_complete callback failed (ignored)", exc_info=True,
                )

    # response.output_text.done
    yield f"event: response.output_text.done\ndata: {json.dumps({'type': 'response.output_text.done', 'output_index': 0, 'content_index': 0, 'text': full_text})}\n\n"

    # response.completed
    yield f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': {'id': rid, 'object': 'response', 'created_at': created, 'model': model, 'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': full_text}]}]}})}\n\n"
