"""
Probabilistic complexity scorer.

Replaces the discrete 0/1/2 _compute_complexity() with a continuous 0.0-1.0 score
based on four weighted signals: prompt length, structural signals, task type base,
and classification ambiguity.

complexity_to_discrete() maps back to 0/1/2 for cold-start fallback to YAML lookups.
"""
from __future__ import annotations

import math

# Task types that are inherently more demanding regardless of prompt length.
_COMPLEX_TASK_TYPES: frozenset[str] = frozenset({
    "reasoning", "math", "code", "data_analysis", "document_review", "tool_use",
})

# Structural signal patterns
_CODE_SIGNALS: tuple[str, ...] = ("```", "def ", "class ", "function(", "select ", "import ")
_MULTI_STEP_SIGNALS: tuple[str, ...] = ("step 1", "1.", "first, ", "then,", "finally,")
_STRUCTURED_SIGNALS: tuple[str, ...] = ("json", "schema", "output format", "structured")

# Signal weights (must sum to 1.0)
_W_LENGTH = 0.25
_W_STRUCTURAL = 0.25
_W_TASK_TYPE = 0.30
_W_AMBIGUITY = 0.20


def _sigmoid(x: float, midpoint: float, steepness: float = 0.01) -> float:
    """Logistic sigmoid centred on midpoint. Returns 0.0-1.0."""
    z = steepness * (x - midpoint)
    # Clamp to avoid overflow
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _length_signal(word_count: int) -> float:
    """Sigmoid around 300 words. Short prompts ~0.0, long prompts ~1.0."""
    return _sigmoid(word_count, midpoint=300, steepness=0.01)


def _structural_signal(text_lower: str, task_type: str) -> float:
    """Detect code markers, multi-step indicators, structured output requests."""
    signals = 0.0
    total = 0.0

    has_code = any(s in text_lower for s in _CODE_SIGNALS)
    has_multi_step = any(s in text_lower for s in _MULTI_STEP_SIGNALS)
    has_structured = any(s in text_lower for s in _STRUCTURED_SIGNALS)
    has_many_numbers = sum(c.isdigit() for c in text_lower) > 15

    if has_code:
        signals += 1.0
    total += 1.0

    if has_multi_step:
        signals += 1.0
    total += 1.0

    if has_structured:
        signals += 1.0
    total += 1.0

    if has_many_numbers and task_type in {"math", "data_analysis"}:
        signals += 1.0
    total += 1.0

    return signals / total if total > 0 else 0.0


def _task_type_signal(task_type: str) -> float:
    """Complex task types start at 0.6, simple at 0.1."""
    if task_type in _COMPLEX_TASK_TYPES:
        return 0.6
    return 0.1


def _ambiguity_signal(classification_score: float) -> float:
    """
    Low classifier confidence = harder request.
    classification_score is the cosine similarity from classifier.classify().
    High score (>0.8) → low ambiguity → low complexity contribution.
    Low score (<0.4) → high ambiguity → high complexity contribution.
    """
    # Invert and clamp: perfect confidence (1.0) → 0.0, no confidence (0.0) → 1.0
    return max(0.0, min(1.0, 1.0 - classification_score))


def compute_complexity(
    messages: list[dict],
    task_type: str,
    classification_score: float = 0.5,
) -> float:
    """
    Compute a continuous complexity score in [0.0, 1.0].

    Four weighted signals:
    - Prompt length (0.25): sigmoid around 300 words
    - Structural signals (0.25): code markers, multi-step, structured output
    - Task type base (0.30): complex types start higher
    - Classification ambiguity (0.20): low classifier confidence = harder

    Args:
        messages: OpenAI-format message list
        task_type: classified task type
        classification_score: cosine similarity from classifier.classify()

    Returns:
        Float in [0.0, 1.0] where 0.0 = trivially simple, 1.0 = maximally complex
    """
    def _to_str(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        return ""

    user_text = " ".join(
        _to_str(m.get("content", "")) for m in messages if m.get("role") == "user"
    )
    word_count = len(user_text.split())
    text_lower = user_text.lower()

    length = _length_signal(word_count)
    structural = _structural_signal(text_lower, task_type)
    task_base = _task_type_signal(task_type)
    ambiguity = _ambiguity_signal(classification_score)

    score = (
        _W_LENGTH * length
        + _W_STRUCTURAL * structural
        + _W_TASK_TYPE * task_base
        + _W_AMBIGUITY * ambiguity
    )

    return max(0.0, min(1.0, score))


def complexity_to_discrete(score: float) -> int:
    """
    Map continuous complexity score to discrete 0/1/2 for cold-start
    fallback to existing YAML policy lookups.

    [0.0, 0.33) → 0 (simple)
    [0.33, 0.66) → 1 (moderate)
    [0.66, 1.0] → 2 (complex)
    """
    if score < 0.33:
        return 0
    if score < 0.66:
        return 1
    return 2
