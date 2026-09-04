from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark import generate_default_benchmark
from app.core.config import settings
from app.models.schemas import BenchmarkCase


@pytest.fixture(scope="session")
def benchmark_cases() -> list[BenchmarkCase]:
    return generate_default_benchmark()


@pytest.fixture(autouse=True)
def isolated_runtime_dirs(tmp_path, monkeypatch):
    """Redirect the output cache and run directory to a per-test location.

    Two problems this solves. Tests must never write into the developer's real
    `data/cache/` or `data/runs/` - a test suite that pollutes user data is
    badly behaved. And cache state leaking between tests makes them
    order-dependent: a caching test passed alone and failed in a full run
    because an earlier test had already populated the shared cache.
    """
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")
    yield
