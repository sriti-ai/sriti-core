from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Sentinel file — stores the artifact path (local or S3) of the last deployed head.
# Written by mark_active(), read by warm_up() on startup.
_ACTIVE_HEAD_PATH = Path(__file__).resolve().parent.parent.parent / "artifacts" / "classifier" / "_active_head.txt"

# ---------------------------------------------------------------------------
# Model — loaded once at startup via warm_up(), never per-request
# fastembed uses ONNX runtime — no PyTorch dependency, works on all platforms
# ---------------------------------------------------------------------------
_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model: TextEmbedding | None = None

# One representative sentence per task type. Chosen to be maximally distinct
# in embedding space so cosine similarity produces a clear winner.
_TASK_ANCHORS: dict[str, str] = {
    "customer_support":      "Help me with my order, account issue, or refund request.",
    "rag_retrieval":         "Based on the provided documents and context, answer the question.",
    "classification":        "Classify or categorize the following item into one of these labels.",
    "summarization":         "Summarize the key points of the following text or document.",
    "structured_extraction": "Extract structured fields such as names, dates, and amounts from this text.",
    "qa":                    "Answer this factual question based on the provided context or document.",
    "translation":           "Translate the following text into the target language.",
    "conversation":          "Let us have a casual chat and discuss whatever is on your mind.",
    "rewriting":             "Rewrite or paraphrase the following text to improve clarity or tone.",
    "reasoning":             "Think through this step by step, analyze the logic, or solve this problem.",
    "math":                  "Solve this mathematical equation or calculate the numerical result.",
    "code":                  "Implement this algorithm in Python, fix this bug, or review this code.",
    "creative":              "Write a creative story, poem, or imaginative piece on this topic.",
    "data_analysis":         "Analyze this dataset and identify trends, patterns, or insights.",
    "document_review":       "Review this contract, report, or document and provide feedback.",
    "tool_use":              "Call a function, use an external tool, or invoke an API to complete this task.",
    "instruction_following": "Follow these specific instructions, rules, and formatting requirements exactly.",
}

# Pre-computed L2-normalised anchor embeddings — populated in warm_up()
_anchor_embeddings: dict[str, NDArray[np.float32]] = {}

# Trained logistic regression head (optional — loaded via reload_head())
# When present, replaces anchor-based cosine similarity with W @ vec + b
_head_W: NDArray[np.float32] | None = None  # (384, N_classes)
_head_b: NDArray[np.float32] | None = None  # (N_classes,)
_head_labels: list[str] | None = None       # class labels in order
_use_trained_head: bool = False

# Trained centroid anchors (optional — loaded alongside head from .npz)
# When present, replaces hardcoded _TASK_ANCHORS for cosine similarity fallback
_trained_anchors: dict[str, NDArray[np.float32]] | None = None


def warm_up() -> None:
    """
    Load the fastembed model and pre-compute anchor embeddings.
    Call once from the FastAPI startup event — never on the hot path.
    Subsequent calls are no-ops.
    """
    global _model
    if _model is not None:
        return
    logger.info("Loading classifier model (%s)...", _MODEL_NAME)
    _model = TextEmbedding(model_name=_MODEL_NAME)
    # Trigger actual ONNX model load (fastembed constructor is lazy)
    list(_model.embed(["warmup"]))

    # Pre-compute normalised anchor embeddings for all task types
    anchor_sentences = list(_TASK_ANCHORS.values())
    raw = list(_model.embed(anchor_sentences))
    for key, vec in zip(_TASK_ANCHORS.keys(), raw):
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        _anchor_embeddings[key] = arr / norm if norm > 0 else arr

    logger.info("Classifier model ready. %d task anchors pre-computed.", len(_anchor_embeddings))

    # Auto-load last deployed head (survives container restarts)
    if _ACTIVE_HEAD_PATH.exists():
        saved_path = _ACTIVE_HEAD_PATH.read_text().strip()
        if saved_path:
            logger.info("Auto-loading classifier head from sentinel: %s", saved_path)
            reload_head(saved_path)


def get_model() -> TextEmbedding:
    """Return the loaded model. Raises if warm_up() was never called."""
    if _model is None:
        raise RuntimeError(
            "Classifier model not loaded. Ensure warm_up() is called at startup."
        )
    return _model


def _download_artifact(artifact_path: str) -> str:
    """Resolve artifact path — download from S3 if needed, return local path."""
    if not artifact_path.startswith("s3://"):
        return artifact_path

    import boto3
    from sriti.core.settings import settings

    # Parse s3://bucket/key
    parts = artifact_path[5:].split("/", 1)
    bucket, key = parts[0], parts[1]

    local_path = Path(tempfile.gettempdir()) / "sriti_classifier" / Path(key).name
    local_path.parent.mkdir(parents=True, exist_ok=True)

    client = boto3.client("s3", region_name=settings.aws_region)
    client.download_file(bucket, key, str(local_path))
    logger.info("Downloaded classifier artifact from s3://%s/%s", bucket, key)
    return str(local_path)


def mark_active(artifact_path: str) -> None:
    """Record the active classifier head path so warm_up() can auto-load on restart."""
    _ACTIVE_HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ACTIVE_HEAD_PATH.write_text(artifact_path)
    logger.info("Marked active classifier head: %s", artifact_path)


def reload_head(artifact_path: str) -> None:
    """Hot-reload a trained classifier head from a .npz file.

    Called by the deploy endpoint. Falls back to anchor-based classification
    if the file is missing or corrupt. Handles S3 paths transparently.
    """
    global _head_W, _head_b, _head_labels, _use_trained_head, _trained_anchors

    try:
        local_path = _download_artifact(artifact_path)
        data = np.load(local_path, allow_pickle=False)
        W = data["W"]  # (384, N_classes)
        b = data["b"]  # (N_classes,)
        # labels stored as S-type array — convert to list of str
        labels_arr = np.load(local_path, allow_pickle=True)["labels"]
        labels = [str(lbl) for lbl in labels_arr]

        _head_W = W.astype(np.float32)
        _head_b = b.astype(np.float32)
        _head_labels = labels
        _use_trained_head = True

        # Load trained centroid anchors if present
        if "centroids" in data and "centroid_labels" in data:
            centroid_labels_arr = np.load(local_path, allow_pickle=True)["centroid_labels"]
            centroid_labels = [str(lbl) for lbl in centroid_labels_arr]
            centroids = data["centroids"].astype(np.float32)
            _trained_anchors = {lbl: centroids[i] for i, lbl in enumerate(centroid_labels)}
            logger.info("Loaded %d trained centroid anchors", len(_trained_anchors))
        else:
            _trained_anchors = None

        logger.info("Classifier head loaded: %d classes from %s", len(labels), artifact_path)
    except Exception:
        logger.warning("Failed to load classifier head from %s — falling back to anchors", artifact_path, exc_info=True)
        _head_W = None
        _head_b = None
        _head_labels = None
        _use_trained_head = False
        _trained_anchors = None


def _trained_head_classify(vec: NDArray[np.float32]) -> tuple[str, float]:
    """Classify using the trained logistic regression head. <0.1ms."""
    logits = vec @ _head_W + _head_b  # type: ignore[operator]
    # Softmax for probability
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    best_idx = int(np.argmax(probs))
    return _head_labels[best_idx], float(probs[best_idx])  # type: ignore[index]


def _best_anchor_match(vec: NDArray[np.float32]) -> tuple[str, float]:
    """Return (task_type, score) for the best matching anchor.

    Uses trained centroid anchors if available, otherwise hardcoded anchors.
    """
    anchors = _trained_anchors if _trained_anchors else _anchor_embeddings
    best_task = "unknown"
    best_score = 0.0
    for task_type, anchor_vec in anchors.items():
        score = float(np.dot(vec, anchor_vec))
        if score > best_score:
            best_score = score
            best_task = task_type
    return best_task, best_score


def _extract_text(content) -> str:
    """Extract text from str or multimodal content list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def classify(messages: list[dict]) -> tuple[str, float, NDArray[np.float32]]:
    """
    Classify the task type from conversation messages.

    Single-turn: embeds the last user message alone.
    Multi-turn: embeds a context-enriched string (system + first user + last user)
    to preserve task-type signal from conversation history.

    In both cases, user_vec is always the last-user-message embedding alone —
    it's reused downstream for quality checks where conversation context would hurt.

    Args:
        messages: OpenAI-format list [{"role": ..., "content": ...}]

    Returns:
        (task_type, similarity_score, user_vec)
        user_vec is the L2-normalised embedding of the last user message.
        Callers can reuse it for quality_check to avoid re-embedding the same text.
        Returns ("unknown", score, vec) if best score < 0.2.

    Note: synchronous (ONNX/CPU-bound). Callers should use asyncio.to_thread.
    """
    if not _anchor_embeddings:
        raise RuntimeError(
            "Anchor embeddings not computed. Ensure warm_up() was called at startup."
        )

    # Extract message parts for context-aware classification
    system_text = ""
    first_user_text = ""
    last_user_text = ""

    for msg in messages:
        if msg.get("role") == "system" and not system_text:
            system_text = _extract_text(msg.get("content", ""))
        if msg.get("role") == "user" and not first_user_text:
            first_user_text = _extract_text(msg.get("content", ""))

    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_text = _extract_text(msg.get("content", ""))
            break

    if not last_user_text.strip():
        from sriti.core.cascade.constants import EMBEDDING_DIM
        zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        return ("unknown", 0.0, zero)

    model = get_model()

    # Embed last user message (always needed — reused downstream for quality checks)
    vecs = list(model.embed([last_user_text]))
    user_vec = np.array(vecs[0], dtype=np.float32)
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm

    # Multi-turn: classify with context-enriched text (system + first user + last user)
    # Single-turn: classify with last user message alone
    is_multi_turn = bool(first_user_text) and first_user_text != last_user_text

    # Use trained head if available, otherwise fall back to anchor similarity
    if _use_trained_head:
        best_task, best_score = _trained_head_classify(user_vec)
    elif is_multi_turn:
        context_parts = [p for p in [system_text, first_user_text, last_user_text] if p.strip()]
        context_text = "\n".join(context_parts)

        ctx_vecs = list(model.embed([context_text]))
        ctx_vec = np.array(ctx_vecs[0], dtype=np.float32)
        ctx_norm = np.linalg.norm(ctx_vec)
        if ctx_norm > 0:
            ctx_vec = ctx_vec / ctx_norm

        best_task, best_score = _best_anchor_match(ctx_vec)
    else:
        best_task, best_score = _best_anchor_match(user_vec)

    if best_score < 0.2:
        return ("unknown", best_score, user_vec)

    return (best_task, best_score, user_vec)
