"""Benchmark generation.

Two benchmark artifacts live here, with different provenance:

- `curated`: the 10 hand-authored Phase 1 seed cases, one per `CaseType`.
- `synthesizer`: the Phase 2 generated benchmark, built ground-truth-first
  from `StructuredCase` objects (see app/models/structured.py).
"""

from app.benchmark.curated import (
    BENCHMARK_VERSION,
    generate_default_benchmark,
    load_benchmark,
    save_benchmark,
)
from app.benchmark.integrity import IntegrityIssue, validate_benchmark, validate_case
from app.benchmark.synthesizer import (
    build_metadata,
    build_structured_case,
    generate_benchmark,
    load_generated_benchmark,
    save_generated_benchmark,
    to_benchmark_case,
)

__all__ = [
    "BENCHMARK_VERSION",
    "IntegrityIssue",
    "build_metadata",
    "build_structured_case",
    "generate_benchmark",
    "generate_default_benchmark",
    "load_benchmark",
    "load_generated_benchmark",
    "save_benchmark",
    "save_generated_benchmark",
    "to_benchmark_case",
    "validate_benchmark",
    "validate_case",
]
