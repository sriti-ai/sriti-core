"""Multimodal content detection utility.

Detects vision and video requirements from message content arrays.
Pure utility — no litellm imports.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Video file extensions and MIME patterns
_VIDEO_PATTERNS: tuple[str, ...] = (
    "data:video/",
    ".mp4",
    ".webm",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
)

_VIDEO_MIME_RE = re.compile(r"video/", re.IGNORECASE)


@dataclass(frozen=True)
class MediaRequirements:
    """Detected media capability requirements for a request."""

    requires_vision: bool = False
    requires_video: bool = False


def detect_media_requirements(messages: list[dict]) -> MediaRequirements:
    """Detect vision/video requirements from message content arrays.

    Scans all messages for multimodal content parts (image_url type).
    If any URL/data contains a video MIME or extension, marks requires_video.

    Fail-open: returns (False, False) on any error.
    """
    try:
        requires_vision = False
        requires_video = False

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for part in content:
                if not isinstance(part, dict):
                    continue

                part_type = part.get("type", "")

                if part_type == "image_url":
                    requires_vision = True
                    # Check if the URL points to a video
                    image_url = part.get("image_url") or {}
                    url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                    url_lower = url.lower()

                    for pattern in _VIDEO_PATTERNS:
                        if pattern in url_lower:
                            requires_video = True
                            break

                    if not requires_video and _VIDEO_MIME_RE.search(url):
                        requires_video = True

        # Video implies vision
        if requires_video:
            requires_vision = True

        return MediaRequirements(
            requires_vision=requires_vision,
            requires_video=requires_video,
        )
    except Exception:
        logger.debug("Media detection failed (fail-open)")
        return MediaRequirements()


# Image formats supported by LLM providers (litellm/Bedrock/OpenAI)
_SUPPORTED_IMAGE_FORMATS = {"png", "jpeg", "jpg", "gif", "webp"}


def normalize_image_formats(messages: list[dict]) -> list[dict]:
    """Convert unsupported image formats (AVIF, TIFF, BMP) to JPEG.

    LiteLLM/Bedrock only support: png, jpeg, gif, webp.
    Common web formats like AVIF cause deterministic failures that
    exhaust all retries and cascade tiers → 502.

    Only processes base64 data URLs. Remote URLs are left as-is.
    Fail-open: returns original messages on any error.
    """
    try:
        return _normalize_images_inner(messages)
    except Exception:
        logger.debug("Image format normalization failed (fail-open)", exc_info=True)
        return messages


def _normalize_images_inner(messages: list[dict]) -> list[dict]:
    import base64
    import io

    from PIL import Image

    normalized = []
    any_converted = False

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            normalized.append(msg)
            continue

        new_parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                new_parts.append(part)
                continue

            image_url = part.get("image_url") or {}
            url = image_url.get("url", "") if isinstance(image_url, dict) else ""

            # Only convert base64 data URLs — can't re-encode remote URLs server-side
            if not url.startswith("data:image/"):
                new_parts.append(part)
                continue

            # Extract MIME type and check if conversion needed
            try:
                header, data = url.split(",", 1)
                mime = header.split(":")[1].split(";")[0].lower()
                fmt = mime.split("/")[-1]  # e.g. "avif", "png", "jpeg"

                if fmt in _SUPPORTED_IMAGE_FORMATS:
                    new_parts.append(part)
                    continue

                # Convert unsupported format to JPEG
                img_bytes = base64.b64decode(data)
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode in ("RGBA", "P", "LA"):
                    img = img.convert("RGB")

                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
                new_b64 = base64.b64encode(buf.getvalue()).decode()

                new_url = f"data:image/jpeg;base64,{new_b64}"
                new_parts.append({
                    "type": "image_url",
                    "image_url": {**image_url, "url": new_url},
                })
                any_converted = True
                logger.info("Converted %s image to JPEG for LLM compatibility", fmt)
            except Exception:
                logger.debug("Image conversion failed for one part (keeping original)", exc_info=True)
                new_parts.append(part)

        normalized.append({**msg, "content": new_parts})

    return normalized if any_converted else messages


def content_to_str(content: object) -> str:
    """Extract text from message content — handles both str and multimodal arrays.

    Used throughout the pipeline to safely extract text from message content
    that may be a string, a list of content parts, or None.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""
