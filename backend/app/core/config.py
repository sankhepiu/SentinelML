from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/
_BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SENTINELML_", extra="ignore")

    app_name: str = "SentinelML"
    environment: str = "development"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    cors_allow_origins: list[str] = ["http://localhost:5173"]
    model_registry_path: str = "../ml/models/artifacts"

    @property
    def resolved_model_registry_path(self) -> Path:
        """`model_registry_path`, anchored to `backend/` if relative.

        A bare relative path would otherwise resolve against whatever the
        process's current working directory happens to be at startup,
        which is fragile -- this makes model loading work the same way
        regardless of where `uvicorn`/`sentinel serve` was invoked from.
        """
        path = Path(self.model_registry_path)
        return path if path.is_absolute() else (_BACKEND_DIR / path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
