from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Matched by exception class name so this module has zero litellm / provider imports.
# Any exception whose type name is in this set will be retried.
_RETRYABLE_NAMES = frozenset({
    "RateLimitError",
    "ServiceUnavailableError",
    "InternalServerError",
    "APIConnectionError",
    "APIStatusError",           # litellm wraps 5xx (incl. 504) as APIStatusError
    "Timeout",
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
})


def _is_retryable(exc: Exception) -> bool:
    return (
        type(exc).__name__ in _RETRYABLE_NAMES
        or isinstance(exc, (ConnectionError, TimeoutError))
    )


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
) -> T:
    """
    Retry an async callable with exponential backoff and jitter.

    Retries on transient provider errors (rate limits, 5xx, connectivity).
    Raises immediately on non-retryable errors (auth, bad request).
    """
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = min(
                    base_delay * (2 ** attempt) + random.uniform(0, 0.5),
                    max_delay,
                )
                logger.warning(
                    "Retrying LLM call (attempt %d/%d) after %.2fs — %s: %s",
                    attempt + 1,
                    max_attempts,
                    delay,
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]
