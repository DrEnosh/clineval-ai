"""Tests for configuration loading.

The security-relevant assertion here is that no secret has a default value and
that the app is offline by default: `is_llm_configured` must be False unless
both a provider and a key are explicitly supplied.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import BASE_DIR, Settings


class TestSettings:
    def test_no_default_credentials(self):
        settings = Settings(_env_file=None)
        assert settings.llm_api_key is None
        assert settings.llm_provider is None

    def test_offline_by_default(self):
        assert Settings(_env_file=None).is_llm_configured is False

    def test_requires_both_provider_and_key_to_be_configured(self):
        assert Settings(_env_file=None, llm_api_key="k").is_llm_configured is False
        assert Settings(_env_file=None, llm_provider="p").is_llm_configured is False
        assert Settings(_env_file=None, llm_api_key="k", llm_provider="p").is_llm_configured is True

    def test_reads_prefixed_values_from_environment(self, monkeypatch):
        monkeypatch.setenv("CLINEVAL_APP_ENV", "staging")
        monkeypatch.setenv("CLINEVAL_RANDOM_SEED", "1234")
        settings = Settings(_env_file=None)
        assert settings.app_env == "staging"
        assert settings.random_seed == 1234

    def test_database_url_defaults_to_local_sqlite(self):
        assert Settings(_env_file=None).database_url.startswith("sqlite:///")

    def test_ignores_unprefixed_ambient_variables(self, monkeypatch):
        """A generic DATABASE_URL on the developer's machine must not silently
        repoint this application's database."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://somewhere/else")
        assert Settings(_env_file=None).database_url.startswith("sqlite:///")

    def test_benchmark_dir_is_resolved_from_project_root(self):
        settings = Settings(_env_file=None)
        assert settings.benchmark_dir == BASE_DIR / "data" / "synthetic"
        assert Path(BASE_DIR / "pyproject.toml").exists()
