from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SENTINELML_", extra="ignore")

    app_name: str = "SentinelML"
    environment: str = "development"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    cors_allow_origins: list[str] = ["http://localhost:5173"]
    model_registry_path: str = "../ml/models/artifacts"


@lru_cache
def get_settings() -> Settings:
    return Settings()
