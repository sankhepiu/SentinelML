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
    # Regex alternative to `cors_allow_origins` -- handy for platforms like
    # Vercel that mint a new preview-deployment origin per branch/PR
    # (e.g. r"^https://sentinelml.*\.vercel\.app$"), which a static list
    # can't cover. Either or both may be set; FastAPI's CORSMiddleware
    # allows an origin if it matches the list OR the regex.
    cors_allow_origin_regex: str | None = None
    model_registry_path: str = "../ml/models/artifacts"
    # Pin a specific trained-model version (e.g. "v2") instead of always
    # resolving to the latest one under model_registry_path -- useful for
    # rolling back, or for deployments that want an explicit, auditable
    # version rather than "whatever was trained most recently."
    model_version: str | None = None

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
