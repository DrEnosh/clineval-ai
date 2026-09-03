from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark import generate_default_benchmark
from app.models.schemas import BenchmarkCase


@pytest.fixture(scope="session")
def benchmark_cases() -> list[BenchmarkCase]:
    return generate_default_benchmark()
