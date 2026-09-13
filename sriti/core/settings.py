"""Settings for the Sriti instance. Only the fields the core modules
actually read are included — no database_url/jwt_secret/multi-tenant
fields, since there is no Postgres dependency and no auth layer in a
single-node deployment.

tier1/2/3_model are left empty by default — empty means catalog-driven
selection from config/models.yaml's `default_tier` field, same convention
as the original. Set an explicit model string here only to override the
catalog for a specific tier.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: str = "local"

    # Infrastructure
    redis_url: str = "redis://localhost:6379"

    # Model config — empty = catalog-driven selection from config/models.yaml
    tier1_model: str = ""
    tier2_model: str = ""
    tier3_model: str = ""

    # API keys — only the provider(s) actually in use for a given box need
    # to be set; litellm_client only forwards whichever are non-empty.
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""
    fireworks_api_key: str = ""
    xai_api_key: str = ""
    perplexity_api_key: str = ""
    nebius_api_key: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"

    cache_encryption_key: str = ""

    compression_mode: str = "local"  # "local" | "remote" | "disabled"
    compression_service_url: str = "http://localhost:8001"  # remote mode only, unused by default
    compression_target_ratio: float = 0.5  # fraction of tokens to RETAIN

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
