"""Shared constants for the cascade engine and related components."""

# Embedding model dimension (BAAI/bge-small-en-v1.5 = 384)
# Used by: classifier, semantic_cache, case_memory
EMBEDDING_DIM = 384

# Hedging phrases that indicate a model is refusing or deflecting.
# Only matched against the FIRST 200 characters — legitimate responses
# may contain "i can't" mid-text (e.g. "I can't recommend X without knowing Y")
# but actual refusals lead with these phrases.
# Used by: cascade engine (quality check), semantic cache (response validation)
HEDGING_PATTERNS: tuple[str, ...] = (
    "i'm not sure",
    "i am not sure",
    "i don't know",
    "i do not know",
    "i cannot",
    "i can't",
    "i'm unable to",
    "i am unable to",
    "i don't have access",
    "i do not have access",
    "as an ai language model",
    "as an ai,",
    "as an ai assistant",
    "i'm just an ai",
    "i was not able to",
    "i wasn't able to",
)

# Phrases that indicate an error was returned as response content.
# Checked against the full (lowered) content.
# Used by: semantic cache (response validation — never cache error content)
ERROR_AS_CONTENT_PATTERNS: tuple[str, ...] = (
    "rate limit exceeded",
    "rate_limit_exceeded",
    "internal server error",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "too many requests",
    "quota exceeded",
    "model is overloaded",
    "request timed out",
    "connection refused",
    "upstream error",
)
