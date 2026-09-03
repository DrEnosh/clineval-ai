"""ClinEval AI FastAPI application (Phase 1: read-only benchmark endpoints).

Phase 1 exposes only what is actually implemented - health, benchmark listing,
and case retrieval. Endpoints for running evaluations, comparing models, and
generating reports are added in later phases (see docs/ARCHITECTURE.md) rather
than stubbed out here returning fake data.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException

from app import __version__
from app.benchmark import BENCHMARK_VERSION, generate_default_benchmark
from app.models.schemas import BenchmarkCase

app = FastAPI(
    title="ClinEval AI",
    version=__version__,
    description=(
        "Evaluation platform for clinical AI note-summarization systems. "
        "Research and evaluation use only - not a medical device, not clinically "
        "validated, and not for diagnosis, treatment, or medical advice."
    ),
)


@lru_cache(maxsize=1)
def _benchmark() -> list[BenchmarkCase]:
    """The in-memory Phase 1 benchmark. Cached because generation is
    deterministic and has no external dependencies; the database-backed
    benchmark store replaces this in Phase 7."""
    return generate_default_benchmark()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "benchmark_version": BENCHMARK_VERSION}


@app.get("/benchmarks")
def list_benchmarks() -> dict[str, object]:
    cases = _benchmark()
    return {
        "benchmark_id": "synthetic-default",
        "benchmark_version": BENCHMARK_VERSION,
        "case_count": len(cases),
        "case_types": sorted({case.case_type.value for case in cases}),
    }


@app.get("/benchmarks/cases")
def list_cases() -> list[dict[str, object]]:
    return [
        {
            "case_id": case.case_id,
            "case_type": case.case_type.value,
            "encounter_type": case.encounter_type,
            "critical_fact_count": len(case.critical_facts),
        }
        for case in _benchmark()
    ]


@app.get("/benchmarks/cases/{case_id}", response_model=BenchmarkCase)
def get_case(case_id: str) -> BenchmarkCase:
    for case in _benchmark():
        if case.case_id == case_id:
            return case
    raise HTTPException(status_code=404, detail=f"Benchmark case not found: {case_id}")
