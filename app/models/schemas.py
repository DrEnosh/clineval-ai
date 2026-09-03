"""Core Pydantic schemas shared by the benchmark, evaluation, scoring, database,
and API layers.

These types are the contract between layers: the benchmark generator produces
`BenchmarkCase` objects, evaluators consume a `BenchmarkCase` + `AIOutput` and
produce `Failure`/`CaseResult` objects, and the (future) database layer stores
exactly these shapes. Changing a field here is a breaking change across the
system, so keep this module the single source of truth for these shapes
rather than redefining subsets elsewhere.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.enums import (
    CaseType,
    DataProvenance,
    FactImportance,
    FactPolarity,
    FactType,
    FailureType,
    MedicationStatus,
    RiskLevel,
    Severity,
    TemporalTrend,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Gold-standard annotation layer (independent of any model under evaluation)
# ---------------------------------------------------------------------------


class ClinicalFact(BaseModel):
    """A single atomic, human-annotated fact extracted from a source note.

    This is the base unit of the gold-standard layer described in
    docs/ARCHITECTURE.md. Facts are authored alongside the benchmark case,
    never derived from the AI system being evaluated.
    """

    fact_id: str
    type: FactType
    content: str
    importance: FactImportance
    polarity: FactPolarity = FactPolarity.AFFIRMED
    attributes: dict[str, Any] = Field(default_factory=dict)


class MedicationFact(ClinicalFact):
    """A medication-specific fact, adding structured fields evaluators need
    to detect wrong drug / wrong dose / wrong status errors independent of
    exact wording.
    """

    type: FactType = FactType.MEDICATION
    drug_name: str
    status: MedicationStatus
    dose: str | None = None
    route: str | None = None


class TemporalFact(ClinicalFact):
    """A fact whose clinical meaning depends on a value changing over time,
    e.g. "creatinine increased from 1.4 to 1.9 mg/dL".
    """

    type: FactType = FactType.TEMPORAL
    earlier_value: str | None = None
    later_value: str | None = None
    unit: str | None = None
    trend: TemporalTrend = TemporalTrend.UNKNOWN


class RiskFlag(BaseModel):
    """A clinically meaningful risk condition the benchmark case is designed
    to surface if a model's summary is unreliable (e.g. missing a med-error
    signal). Independent of any particular evaluation run.
    """

    flag_id: str
    description: str
    severity: Severity


class BenchmarkCase(BaseModel):
    """One clinical note-summarization benchmark case.

    Field groups mirror the project spec: structured clinical context
    (conditions/medications/.../vital_signs), the free-text note and
    reference summary, and the independent gold-standard annotation layer
    (critical_facts / non_critical_facts / medication_facts / temporal_facts
    / negations / expected_risk_flags) that evaluators compare AI output
    against.
    """

    case_id: str
    case_type: CaseType
    benchmark_version: str
    provenance: DataProvenance

    # Structured clinical context
    age_group: str
    sex: str
    encounter_type: str
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    procedures: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    laboratory_values: list[str] = Field(default_factory=list)
    vital_signs: list[str] = Field(default_factory=list)
    clinical_timeline: list[str] = Field(default_factory=list)

    # Free text
    clinical_note: str
    reference_summary: str

    # Gold-standard annotation layer
    critical_facts: list[ClinicalFact] = Field(default_factory=list)
    non_critical_facts: list[ClinicalFact] = Field(default_factory=list)
    medication_facts: list[MedicationFact] = Field(default_factory=list)
    temporal_facts: list[TemporalFact] = Field(default_factory=list)
    negations: list[ClinicalFact] = Field(default_factory=list)
    expected_risk_flags: list[RiskFlag] = Field(default_factory=list)

    # Subgroup dimensions for benchmark-level subgroup analysis.
    # NOTE: subgroup results describe the benchmark only and must not be
    # presented as evidence about real-world population fairness.
    subgroup: dict[str, str] = Field(default_factory=dict)

    @field_validator("case_id")
    @classmethod
    def _case_id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("case_id must not be blank")
        return v


# ---------------------------------------------------------------------------
# Model-under-test and its output
# ---------------------------------------------------------------------------


class Model(BaseModel):
    """A registered AI system under evaluation."""

    id: str
    name: str
    provider: str
    version: str
    configuration: dict[str, Any] = Field(default_factory=dict)


class AIOutput(BaseModel):
    """One generated summary for one case, from one model.

    `run_index` distinguishes repeated generations of the same case/model
    pair, which the consistency-testing evaluator needs.
    """

    case_id: str
    model_id: str
    summary_text: str
    run_index: int = 0
    generated_at: datetime = Field(default_factory=_utcnow)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation outputs
# ---------------------------------------------------------------------------


class Failure(BaseModel):
    """A single, evidenced, explainable evaluation finding.

    Every score the platform reports must be traceable to `Failure` records
    like this one rather than presented as a bare number (see docs/ARCHITECTURE.md,
    "Product philosophy").
    """

    type: FailureType
    severity: Severity
    explanation: str
    evidence: str
    case_id: str
    fact_id: str | None = None


class CaseResult(BaseModel):
    """Per-case evaluation output within one evaluation run."""

    evaluation_run_id: str
    case_id: str
    metrics: dict[str, float] = Field(default_factory=dict)
    failures: list[Failure] = Field(default_factory=list)


class EvaluationRun(BaseModel):
    """One reproducible evaluation run: a model evaluated against a benchmark
    under a specific, recorded configuration.

    Every field needed to reproduce the run is recorded on the run itself
    rather than inferred later (see docs/ARCHITECTURE.md, Observability).
    """

    id: str
    timestamp: datetime = Field(default_factory=_utcnow)
    model_id: str
    benchmark_id: str
    benchmark_version: str
    evaluation_version: str
    prompt_version: str | None = None
    random_seed: int | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    score: dict[str, float] | None = None
    risk: RiskLevel | None = None


class Report(BaseModel):
    """Pointer to a generated evaluation report artifact for a run."""

    evaluation_run_id: str
    generated_at: datetime = Field(default_factory=_utcnow)
    report_location: str
