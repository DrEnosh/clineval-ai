"""Application configuration.

All configuration is read from environment variables (optionally via a local
.env file, see .env.example). Nothing here is hard-coded: there are no
default secrets, and no default value that would silently enable a real LLM
call. Instantiate `Settings()` (or use the module-level `settings` singleton)
rather than reading `os.environ` elsewhere in the codebase.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root, resolved from this file's location so paths work regardless
# of the process's current working directory (e.g. under pytest or uvicorn).
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Every variable is namespaced (CLINEVAL_DATABASE_URL, CLINEVAL_LLM_API_KEY,
    # ...). Without the prefix, a generic ambient variable such as DATABASE_URL
    # -- common on developer machines -- would silently override this app's
    # configuration and point it somewhere unintended.
    model_config = SettingsConfigDict(
        env_prefix="CLINEVAL_",
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ClinEval AI"
    app_env: str = "development"
    log_level: str = "INFO"
    random_seed: int = 42

    database_url: str = "sqlite:///./clineval.db"

    # Optional: only consulted by evaluators/model adapters that explicitly
    # need to call a hosted LLM. Left unset, the app runs entirely offline.
    llm_api_key: str | None = None
    llm_provider: str | None = None

    # Where committed synthetic benchmark files live.
    benchmark_dir: Path = BASE_DIR / "data" / "synthetic"

    @property
    def is_llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_provider)


settings = Settings()
