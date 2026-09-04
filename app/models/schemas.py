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
    DifficultyTier,
    FactImportance,
    FactPolarity,
    FactType,
    FailureModeCategory,
    FindingType,
    MatchLevel,
    MedicationStatus,
    RiskLevel,
    Severity,
    TemporalTrend,
)
from app.models.structured import FailureOpportunity, StructuredCase


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

    # Phase 2 traceability. `source_ref` is the `entity_id` of the structured
    # element this fact was derived from, and `criticality_rule` is the id of
    # the rule that assigned its importance (see app/benchmark/criticality.py).
    # Both are None on the hand-authored Phase 1 seed cases, which predate the
    # structured-case pipeline.
    source_ref: str | None = None
    criticality_rule: str | None = None


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
    diagnoses: list[str] = Field(default_factory=list)
    clinical_findings: list[str] = Field(default_factory=list)
    relevant_negative_findings: list[str] = Field(default_factory=list)
    follow_up: list[str] = Field(default_factory=list)

    # Free text
    clinical_note: str
    reference_summary: str

    # Gold-standard annotation layer
    critical_facts: list[ClinicalFact] = Field(default_factory=list)
    non_critical_facts: list[ClinicalFact] = Field(default_factory=list)
    medication_facts: list[MedicationFact] = Field(default_factory=list)
    temporal_facts: list[TemporalFact] = Field(default_factory=list)
    negations: list[ClinicalFact] = Field(default_factory=list)
    contradiction_facts: list[ClinicalFact] = Field(default_factory=list)
    expected_risk_flags: list[RiskFlag] = Field(default_factory=list)

    # Phase 2 generation layer. `structured_case` is the source of truth this
    # case was derived from - note, reference summary, and every annotation
    # above are three independent renderings of it. None on the hand-authored
    # Phase 1 seed cases, which were written before this pipeline existed.
    structured_case: StructuredCase | None = None
    difficulty: DifficultyTier | None = None
    categories: list[FailureModeCategory] = Field(default_factory=list)
    expected_failure_opportunities: list[FailureOpportunity] = Field(default_factory=list)
    generation_metadata: dict[str, Any] = Field(default_factory=dict)

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


class EvaluationFinding(BaseModel):
    """A single evidenced, explainable evaluation finding.

    Every score the platform reports must be traceable to findings like this
    one rather than presented as a bare number (see docs/ARCHITECTURE.md,
    "Product philosophy"). The evidence fields are what make that real: a
    finding records what the source says AND what the output says, so a
    reviewer can judge the call without re-reading the case.

    `severity` is deliberately optional at construction. Evaluators detect;
    the severity policy classifies (see app/evaluation/severity.py). A finding
    reaching a caller with `severity is None` means the policy was never
    applied, which the runner guarantees against.
    """

    finding_id: str
    case_id: str
    evaluator_name: str
    evaluator_version: str
    finding_type: FindingType
    explanation: str

    severity: Severity | None = None
    fact_id: str | None = None
    source_ref: str | None = None
    source_evidence: str | None = None
    output_evidence: str | None = None

    # 1.0 for a deterministic detection with no ambiguity; lower where the
    # evaluator relied on a weaker match level or a heuristic.
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    match_level: MatchLevel | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class MetricValue(BaseModel):
    """One computed metric, carrying the counts it was derived from.

    Storing numerator and denominator alongside the value is what lets an
    aggregate be recomputed by pooling rather than by averaging averages, and
    lets a reader see that "100% retention" came from 1 of 1 rather than 40
    of 40. `value` is None where the denominator is zero - an undefined
    metric is reported as undefined, never as 0.0 or 1.0.
    """

    name: str
    value: float | None
    numerator: float
    denominator: float
    definition: str

    @property
    def is_defined(self) -> bool:
        return self.value is not None


class CaseResult(BaseModel):
    """Per-case evaluation output within one evaluation run."""

    evaluation_run_id: str
    case_id: str
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    findings: list[EvaluationFinding] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """The complete output of evaluating one model against one benchmark."""

    evaluation_run_id: str
    benchmark_version: str
    model_id: str
    evaluation_version: str
    generated_at: datetime = Field(default_factory=_utcnow)
    cases_evaluated: int = 0
    evaluators: list[dict[str, str]] = Field(default_factory=list)
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    case_results: list[CaseResult] = Field(default_factory=list)
    findings: list[EvaluationFinding] = Field(default_factory=list)
    configuration: dict[str, Any] = Field(default_factory=dict)


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
