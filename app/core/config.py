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

    # Per-provider credentials. Separate fields rather than one shared key so a
    # user can configure several providers at once, and so no provider can
    # accidentally pick up another's credential. All default to None: the
    # application runs fully offline with none of them set, and no benchmark
    # data leaves the machine unless a provider is explicitly configured.
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None

    # Base URL override for OpenAI-compatible local servers (Ollama, vLLM,
    # LM Studio). Set this and a local model runs through the same interface.
    local_base_url: str | None = None

    # Model names come from configuration, never hard-coded, because provider
    # model identifiers change and a pinned name silently rots.
    summarizer_model: str | None = None
    judge_model: str | None = None

    # Where committed synthetic benchmark files live.
    benchmark_dir: Path = BASE_DIR / "data" / "synthetic"

    # Generated model outputs are cached here so a benchmark can be re-evaluated
    # without re-paying for generation. Runtime state - gitignored.
    cache_dir: Path = BASE_DIR / "data" / "cache"
    runs_dir: Path = BASE_DIR / "data" / "runs"

    @property
    def is_llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_provider)

    def api_key_for(self, provider: str) -> str | None:
        """The configured credential for a provider, or None.

        Returns None rather than raising so callers can report a clear
        "provider not configured" message instead of a traceback.
        """
        return {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.google_api_key,
            "google": self.google_api_key,
            # A local OpenAI-compatible server usually needs no real key.
            "local": self.openai_api_key or "not-required",
            "mock": "not-required",
        }.get(provider.lower())

    def configured_providers(self) -> list[str]:
        """Providers with a usable credential, for reporting to the user."""
        available = ["mock"]
        if self.anthropic_api_key:
            available.append("anthropic")
        if self.openai_api_key:
            available.append("openai")
        if self.google_api_key:
            available.append("gemini")
        if self.local_base_url:
            available.append("local")
        return available


settings = Settings()
