from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sriti.core.execution.litellm_client import count_tokens as _litellm_count_tokens

from sriti.core.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Policy — eligible task types loaded from policy.yaml
# ---------------------------------------------------------------------------

_POLICY_PATH = Path(__file__).parent.parent.parent / "config" / "policy.yaml"


def _load_eligible_task_types() -> list[str]:
    try:
        with open(_POLICY_PATH) as f:
            data = yaml.safe_load(f)
        return data["policy"]["compression"]["eligible_task_types"]
    except Exception:
        return ["customer_support", "rag_retrieval", "classification", "summarization"]


def _load_compression_quality_threshold() -> float:
    try:
        with open(_POLICY_PATH) as f:
            data = yaml.safe_load(f)
        return float(data["policy"]["compression"]["quality_threshold"])
    except Exception:
        return 0.92


_ELIGIBLE_TASK_TYPES: list[str] = _load_eligible_task_types()
_COMPRESSION_QUALITY_THRESHOLD: float = _load_compression_quality_threshold()

# ---------------------------------------------------------------------------
# CompressionResult
# ---------------------------------------------------------------------------


@dataclass
class CompressionResult:
    compressed_messages: list[dict]
    original_token_count: int
    compressed_token_count: int
    tokens_saved: int
    compression_ratio: float       # fraction reduced: 0.30 = 30% fewer tokens
    raw_compressed_text: str       # for quality gate input
    skipped: bool = False
    skip_reason: str | None = None


# ---------------------------------------------------------------------------
# Module-level lazy compressor (local mode only)
# ---------------------------------------------------------------------------

_compressor: Any = None  # PromptCompressor instance, loaded on first call


def _get_compressor() -> Any:
    global _compressor
    if _compressor is None:
        from llmlingua import PromptCompressor  # type: ignore[import]
        logger.info("Loading LLMLingua-2 model (first call)...")
        _compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
        logger.info("LLMLingua-2 model loaded.")
    return _compressor


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    try:
        return _litellm_count_tokens(text)
    except Exception:
        return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Message segmentation
# ---------------------------------------------------------------------------


def _split_messages(
    messages: list[dict],
) -> tuple[dict | None, list[dict], dict | None]:
    """
    Split messages into (system, history, last_user).

    system   = first message if role == "system", else None
    last_user = last message where role == "user"
    history  = everything in between (prior turns, excludes last_user)
    """
    if not messages:
        return None, [], None

    system: dict | None = None
    remaining = list(messages)

    if remaining and remaining[0].get("role") == "system":
        system = remaining[0]
        remaining = remaining[1:]

    # Find last user message
    last_user_idx: int | None = None
    for i in range(len(remaining) - 1, -1, -1):
        if remaining[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        return system, [], None

    last_user = remaining[last_user_idx]
    history = remaining[:last_user_idx]

    return system, history, last_user


# ---------------------------------------------------------------------------
# Serialization / reconstruction
# ---------------------------------------------------------------------------

_ROLE_TAG_RE = re.compile(r"\[(USER|ASSISTANT)\]")


def _serialize_history(history: list[dict]) -> str:
    """Serialize history messages to a string with role tags."""
    parts = []
    for msg in history:
        role = msg.get("role", "user").upper()
        if role == "USER":
            tag = "[USER]"
        else:
            tag = "[ASSISTANT]"
        parts.append(f"{tag} {msg.get('content', '')}")
    return "\n".join(parts)


def _reconstruct_messages(compressed_text: str, history: list[dict]) -> list[dict]:
    """
    Split compressed text on role tags and rebuild message list.
    Falls back to single user message if parsing fails.
    """
    segments = _ROLE_TAG_RE.split(compressed_text)
    # segments alternates: [pre_text, ROLE, content, ROLE, content, ...]
    result: list[dict] = []
    i = 1  # skip any leading text before first tag
    while i < len(segments) - 1:
        role_tag = segments[i]
        content = segments[i + 1].strip()
        if role_tag == "USER":
            result.append({"role": "user", "content": content})
        elif role_tag == "ASSISTANT":
            result.append({"role": "assistant", "content": content})
        i += 2

    if not result:
        # Fallback: return compressed text as a single user message
        return [{"role": "user", "content": compressed_text.strip()}]

    return result


# ---------------------------------------------------------------------------
# Core compress() — three execution modes
# ---------------------------------------------------------------------------


async def compress(
    messages: list[dict],
    target_ratio: float | None = None,
) -> CompressionResult:
    """
    Compress conversation history in messages using LLMLingua-2.

    Always returns a CompressionResult. Never raises — fail-open.
    """
    if target_ratio is None:
        target_ratio = settings.compression_target_ratio

    try:
        return await _compress_inner(messages, target_ratio)
    except Exception as exc:
        logger.warning("Compression error (fail-open): %s", exc)
        return CompressionResult(
            compressed_messages=messages,
            original_token_count=0,
            compressed_token_count=0,
            tokens_saved=0,
            compression_ratio=0.0,
            raw_compressed_text="",
            skipped=True,
            skip_reason="compression_error",
        )


async def _compress_inner(
    messages: list[dict],
    target_ratio: float,
) -> CompressionResult:
    mode = settings.compression_mode

    if mode == "disabled":
        return CompressionResult(
            compressed_messages=messages,
            original_token_count=0,
            compressed_token_count=0,
            tokens_saved=0,
            compression_ratio=0.0,
            raw_compressed_text="",
            skipped=True,
            skip_reason="compression_disabled",
        )

    system, history, last_user = _split_messages(messages)

    if not history:
        return CompressionResult(
            compressed_messages=messages,
            original_token_count=0,
            compressed_token_count=0,
            tokens_saved=0,
            compression_ratio=0.0,
            raw_compressed_text="",
            skipped=True,
            skip_reason="no_compressible_history",
        )

    serialized = _serialize_history(history)
    original_tokens = _count_tokens(serialized)

    if mode == "local":
        compressed_text = await asyncio.to_thread(
            _run_local_compression, serialized, original_tokens, target_ratio
        )
    elif mode == "remote":
        compressed_text = await _run_remote_compression(serialized, target_ratio)
    else:
        return CompressionResult(
            compressed_messages=messages,
            original_token_count=0,
            compressed_token_count=0,
            tokens_saved=0,
            compression_ratio=0.0,
            raw_compressed_text="",
            skipped=True,
            skip_reason="compression_disabled",
        )

    compressed_tokens = _count_tokens(compressed_text)
    tokens_saved = max(0, original_tokens - compressed_tokens)
    ratio = round(tokens_saved / original_tokens, 4) if original_tokens > 0 else 0.0

    reconstructed_history = _reconstruct_messages(compressed_text, history)
    rebuilt: list[dict] = []
    if system:
        rebuilt.append(system)
    rebuilt.extend(reconstructed_history)
    if last_user:
        rebuilt.append(last_user)

    return CompressionResult(
        compressed_messages=rebuilt,
        original_token_count=original_tokens,
        compressed_token_count=compressed_tokens,
        tokens_saved=tokens_saved,
        compression_ratio=ratio,
        raw_compressed_text=compressed_text,
        skipped=False,
    )


def _run_local_compression(
    serialized_history: str,
    original_tokens: int,
    target_ratio: float,
) -> str:
    """Synchronous — runs in asyncio.to_thread."""
    compressor = _get_compressor()
    target_token = int(original_tokens * target_ratio)
    result = compressor.compress_prompt(
        context=[serialized_history],
        target_token=max(1, target_token),
    )
    return result["compressed_prompt"]


async def _run_remote_compression(
    serialized_history: str,
    target_ratio: float,
) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(
            f"{settings.compression_service_url}/compress",
            json={"text": serialized_history, "target_ratio": target_ratio},
        )
        resp.raise_for_status()
        return resp.json()["compressed_text"]


# ---------------------------------------------------------------------------
# maybe_compress() — eligibility check + compress + quality gate
# ---------------------------------------------------------------------------


async def maybe_compress(
    messages: list[dict],
    task_type: str,
    target_ratio: float | None = None,
) -> CompressionResult:
    """
    Run eligibility check, compress, and quality gate in one call.
    Always returns CompressionResult. Never raises.
    """
    if target_ratio is None:
        target_ratio = settings.compression_target_ratio

    # 1. Mode check
    if settings.compression_mode == "disabled":
        return CompressionResult(
            compressed_messages=messages,
            original_token_count=0,
            compressed_token_count=0,
            tokens_saved=0,
            compression_ratio=0.0,
            raw_compressed_text="",
            skipped=True,
            skip_reason="compression_disabled",
        )

    # 2. Multimodal content check — LLMLingua can't handle image/audio parts
    if any(isinstance(m.get("content"), list) for m in messages):
        return CompressionResult(
            compressed_messages=messages,
            original_token_count=0,
            compressed_token_count=0,
            tokens_saved=0,
            compression_ratio=0.0,
            raw_compressed_text="",
            skipped=True,
            skip_reason="multimodal_content",
        )

    # 3. Task eligibility check
    if task_type not in _ELIGIBLE_TASK_TYPES:
        return CompressionResult(
            compressed_messages=messages,
            original_token_count=0,
            compressed_token_count=0,
            tokens_saved=0,
            compression_ratio=0.0,
            raw_compressed_text="",
            skipped=True,
            skip_reason=f"task_not_eligible:{task_type}",
        )

    # 3. Compress
    result = await compress(messages, target_ratio=target_ratio)
    if result.skipped:
        return result

    # 4. Quality gate
    if result.raw_compressed_text:
        # Expansion check — if compression made it longer, skip
        if result.compressed_token_count > result.original_token_count:
            return CompressionResult(
                compressed_messages=messages,
                original_token_count=result.original_token_count,
                compressed_token_count=result.compressed_token_count,
                tokens_saved=0,
                compression_ratio=0.0,
                raw_compressed_text=result.raw_compressed_text,
                skipped=True,
                skip_reason="compression_expanded",
            )

        try:
            from sriti.core.cascade import classifier
            from sriti.core.compression.quality_gate import check as quality_check

            _, history, _ = _split_messages(messages)
            original_text = _serialize_history(history)

            model = classifier.get_model()
            passed, similarity = await asyncio.to_thread(
                quality_check,
                original_text,
                result.raw_compressed_text,
                _COMPRESSION_QUALITY_THRESHOLD,
                model,
            )

            if not passed:
                logger.debug(
                    "Compression quality gate failed: similarity=%.4f threshold=%.4f",
                    similarity, _COMPRESSION_QUALITY_THRESHOLD,
                )
                return CompressionResult(
                    compressed_messages=messages,
                    original_token_count=result.original_token_count,
                    compressed_token_count=result.compressed_token_count,
                    tokens_saved=0,
                    compression_ratio=0.0,
                    raw_compressed_text=result.raw_compressed_text,
                    skipped=True,
                    skip_reason="quality_gate_failed",
                )
        except Exception as exc:
            logger.warning("Quality gate error (fail-closed, returning original): %s", exc)
            # Fail-closed: return original messages, not potentially garbled compression
            return CompressionResult(
                compressed_messages=messages,
                original_token_count=result.original_token_count,
                compressed_token_count=result.original_token_count,
                tokens_saved=0,
                compression_ratio=0.0,
                skipped=True,
                skip_reason="quality_gate_error",
            )

    return result
