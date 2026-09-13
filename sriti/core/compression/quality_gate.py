from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding


def check(
    original: str,
    compressed: str,
    threshold: float,
    model: TextEmbedding,
) -> tuple[bool, float]:
    """
    Cosine similarity quality gate.

    Embeds both strings using the classifier's fastembed model (no new model loaded),
    then computes cosine similarity between L2-normalised vectors.

    Args:
        original:   Original (pre-compression) text.
        compressed: Compressed text output from LLMLingua-2.
        threshold:  Minimum cosine similarity to pass.
        model:      fastembed TextEmbedding — reuse classifier's model, no reload.

    Returns:
        (passed, similarity) where passed = similarity >= threshold.

    Note: synchronous (CPU-bound). Callers should use asyncio.to_thread.
    """
    if not compressed:
        return False, 0.0

    vecs = list(model.embed([original, compressed]))

    va = np.array(vecs[0], dtype=np.float32)
    vb = np.array(vecs[1], dtype=np.float32)

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)

    if norm_a > 0:
        va = va / norm_a
    if norm_b > 0:
        vb = vb / norm_b

    similarity = float(np.dot(va, vb))
    return similarity >= threshold, similarity
