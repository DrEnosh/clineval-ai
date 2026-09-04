"""Controlled vocabularies shared across schemas, evaluation, scoring, and the API.

Keeping these as enums (rather than free-text strings) is what makes scoring
rules and severity policy in docs/ARCHITECTURE.md enforceable in code instead
of just documented conventions.
"""

from __future__ import annotations

from enum import Enum


class Severity(str, Enum):
    """Failure severity ladder. Definitions live in docs/ARCHITECTURE.md."""

    INFO = "INFO"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskLevel(str, Enum):
    """Overall deployment-risk classification for an evaluation run."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class FactType(str, Enum):
    CONDITION = "condition"
    MEDICATION = "medication"
    ALLERGY = "allergy"
    PROCEDURE = "procedure"
    OBSERVATION = "observation"
    LAB = "lab"
    VITAL = "vital"
    DEMOGRAPHIC = "demographic"
    TEMPORAL = "temporal"
    NEGATION = "negation"
    OTHER = "other"


class FactImportance(str, Enum):
    CRITICAL = "critical"
    NON_CRITICAL = "non_critical"


class FactPolarity(str, Enum):
    """Whether the source note affirms or negates a fact.

    Distinguishing these is what lets the evaluator catch a model turning
    "no history of diabetes" into "patient has diabetes" (see
    docs/ARCHITECTURE.md, negation handling).
    """

    AFFIRMED = "affirmed"
    NEGATED = "negated"


class MedicationStatus(str, Enum):
    CURRENT = "current"
    DISCONTINUED = "discontinued"
    CHANGED_DOSE = "changed_dose"
    NEW = "new"
    HISTORICAL = "historical"


class TemporalTrend(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    UNKNOWN = "unknown"


class CaseType(str, Enum):
    """The curated Phase 1 seed set's typology - exactly one hand-built case
    per member (see app/benchmark/curated.py).

    Phase 2's generated cases use `FailureModeCategory` instead, because a
    generated case routinely combines several failure modes at once and so
    needs a *list* of categories rather than a single type. Each generated
    case still records a dominant `case_type` for continuity.
    """

    NORMAL = "normal"
    CRITICAL_OMISSION = "critical_omission"
    HALLUCINATION = "hallucination"
    NEGATION = "negation"
    TEMPORAL = "temporal"
    MEDICATION = "medication"
    MULTI_MORBIDITY = "multi_morbidity"
    HIGH_COMPLEXITY = "high_complexity"
    CONFLICTING_INFORMATION = "conflicting_information"
    DEMOGRAPHIC_SUBGROUP = "demographic_subgroup"


class FindingType(str, Enum):
    """What an evaluator detected.

    Detection only - severity is assigned separately by the severity policy
    (see app/evaluation/severity.py), so an evaluator never decides how bad
    its own finding is.
    """

    # Retention
    CRITICAL_OMISSION = "critical_omission"
    NON_CRITICAL_OMISSION = "non_critical_omission"
    CONTRADICTION = "contradiction"

    # Negation
    NEGATION_REVERSAL = "negation_reversal"
    NEGATION_OMISSION = "negation_omission"

    # Medication
    MEDICATION_OMISSION = "medication_omission"
    MEDICATION_HALLUCINATION = "medication_hallucination"
    MEDICATION_STATUS_ERROR = "medication_status_error"
    MEDICATION_DOSE_ERROR = "medication_dose_error"

    # Temporal
    TEMPORAL_INFORMATION_LOSS = "temporal_information_loss"
    TEMPORAL_TREND_REVERSAL = "temporal_trend_reversal"
    TEMPORAL_VALUE_CONFUSION = "temporal_value_confusion"
    HISTORICAL_CURRENT_CONFUSION = "historical_current_confusion"

    # Values
    VALUE_ERROR = "value_error"
    VALUE_APPROXIMATION = "value_approximation"

    # Support
    UNSUPPORTED_ADDITION = "unsupported_addition"
    CONFLICT_RESOLUTION_ERROR = "conflict_resolution_error"

    # Operational
    EMPTY_OUTPUT = "empty_output"
    MALFORMED_OUTPUT = "malformed_output"
    MISSING_ANNOTATION = "missing_annotation"
    INCONSISTENT_ACROSS_RUNS = "inconsistent_across_runs"
    OTHER = "other"


class MatchLevel(str, Enum):
    """How a source term was matched in model output.

    The ladder is ordered from most to least certain. Recording the level on
    every match means a finding can say not just "not found" but "not found
    at any matching level", and a reviewer can see whether a match rested on
    an alias or an exact string (see docs/EVALUATION.md, fact matching).
    """

    EXACT = "exact"
    NORMALIZED = "normalized"
    ALIAS = "alias"
    PARTIAL = "partial"
    NONE = "none"


class SupportStatus(str, Enum):
    """Whether model output supports, omits, or contradicts a source fact."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    OMITTED = "omitted"
    CONTRADICTED = "contradicted"


class ValueComparison(str, Enum):
    """Result of comparing a clinical numeric value against its source."""

    CORRECT = "correct"
    APPROXIMATE = "approximate"
    INCORRECT = "incorrect"
    OMITTED = "omitted"
    NOT_COMPARABLE = "not_comparable"


# ---------------------------------------------------------------------------
# Phase 4: model generation and LLM adjudication
# ---------------------------------------------------------------------------


class ProviderKind(str, Enum):
    """Which backend a `ModelProvider` talks to.

    `MOCK` is deterministic and offline; it is what tests and CI use, so no
    suite ever depends on a paid external model.
    """

    MOCK = "mock"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    LOCAL = "local"


class AdjudicationJudgment(str, Enum):
    """The verdict an LLM adjudicator may return.

    Deliberately small and closed. `UNCERTAIN` exists so the judge has a way
    to decline: a judge forced to choose between SUPPORTED and CONTRADICTED
    will guess, and a guess recorded as a verdict is worse than an admission
    of uncertainty.
    """

    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNCERTAIN = "UNCERTAIN"


class AdjudicationStatus(str, Enum):
    """What happened to a finding that was considered for adjudication.

    `ADJUDICATION_REQUIRED` is the explicit "unresolved" state. When the
    deterministic evaluator and the LLM disagree in a way no documented rule
    resolves, the result stays here rather than silently picking a side.
    """

    NOT_REQUIRED = "not_required"
    AGREED = "agreed"
    RESOLVED_BY_LLM = "resolved_by_llm"
    ADJUDICATION_REQUIRED = "adjudication_required"
    FAILED = "failed"
    SKIPPED = "skipped"


class DisagreementType(str, Enum):
    """How a deterministic finding and an LLM judgment differ."""

    NONE = "none"
    LLM_MORE_LENIENT = "llm_more_lenient"
    LLM_MORE_STRICT = "llm_more_strict"
    LLM_UNCERTAIN = "llm_uncertain"
    INCOMPATIBLE = "incompatible"


class DataProvenance(str, Enum):
    """Where a benchmark case's content originated. Required on every case
    per the project's data-provenance requirement (see docs/ARCHITECTURE.md).
    """

    SYNTHETIC_TEMPLATE = "synthetic_template"
    SYNTHEA_DERIVED = "synthea_derived"
    MANUAL_SYNTHETIC = "manual_synthetic"


# ---------------------------------------------------------------------------
# Phase 2: synthetic benchmark generation
# ---------------------------------------------------------------------------


class DifficultyTier(str, Enum):
    """Benchmark difficulty tiers. Exact definitions in docs/BENCHMARK.md.

    These describe how much clinical content a case carries and how adversarial
    its construction is - they are benchmark design labels, not statements
    about how hard a case is for a clinician.
    """

    EASY = "EASY"
    MODERATE = "MODERATE"
    HARD = "HARD"
    ADVERSARIAL = "ADVERSARIAL"


class FailureModeCategory(str, Enum):
    """The failure modes a generated case is built to exercise.

    A case carries a *list* of these: real clinical documentation mixes
    failure modes, and the interesting cases are the ones that combine them.
    """

    STRAIGHTFORWARD = "straightforward"
    CRITICAL_OMISSION = "critical_omission"
    HALLUCINATION_OPPORTUNITY = "hallucination_opportunity"
    NEGATION = "negation"
    TEMPORAL_PROGRESSION = "temporal_progression"
    MEDICATION_CHANGE = "medication_change"
    MEDICATION_DISCONTINUATION = "medication_discontinuation"
    MEDICATION_DOSE_CHANGE = "medication_dose_change"
    MULTIPLE_COMORBIDITIES = "multiple_comorbidities"
    HIGH_COMPLEXITY = "high_complexity"
    CONFLICTING_INFORMATION = "conflicting_information"
    ABNORMAL_LABS = "abnormal_labs"
    ABNORMAL_VITALS = "abnormal_vitals"
    RELEVANT_NEGATIVE_FINDINGS = "relevant_negative_findings"
    HISTORICAL_VS_CURRENT = "historical_vs_current"
    PROCEDURES_AND_FOLLOWUP = "procedures_and_followup"
    DUPLICATE_DOCUMENTATION = "duplicate_documentation"
    LONG_NOTE = "long_note"
    SHORT_NOTE = "short_note"
    COMBINED_CRITICAL = "combined_critical"
    #: Adversarial text embedded in the note that attempts to address the model.
    #: Deliberately NOT in any tier's default pool - these cases test the
    #: evaluation harness, not the summarizer, so including them by default
    #: would dilute the summarization benchmark. Reachable with
    #: `--categories prompt_injection`.
    PROMPT_INJECTION = "prompt_injection"


class AbnormalFlag(str, Enum):
    """How a measured value compares to its reference range.

    CRITICAL_LOW / CRITICAL_HIGH mean "outside the panic threshold", which is
    what criticality rule CR-01 keys on (see app/benchmark/criticality.py).
    """

    NORMAL = "normal"
    LOW = "low"
    HIGH = "high"
    CRITICAL_LOW = "critical_low"
    CRITICAL_HIGH = "critical_high"


class ConditionStatus(str, Enum):
    ACTIVE = "active"
    HISTORICAL = "historical"
    RESOLVED = "resolved"


class FindingSignificance(str, Enum):
    ROUTINE = "routine"
    IMPORTANT = "important"
    CRITICAL = "critical"


class AllergySeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class TimelineEventType(str, Enum):
    ENCOUNTER_START = "encounter_start"
    ASSESSMENT = "assessment"
    LAB_RESULT = "lab_result"
    VITAL_MEASUREMENT = "vital_measurement"
    MEDICATION_CHANGE = "medication_change"
    PROCEDURE = "procedure"
    DETERIORATION = "deterioration"
    IMPROVEMENT = "improvement"
    REASSESSMENT = "reassessment"
    DISPOSITION = "disposition"


class FailureOpportunityType(str, Enum):
    """A known way a summarizer could go wrong on this case.

    These are *benchmark conditions* describing what a model might get wrong -
    they are never fabricated model outputs, and they never alter the ground
    truth (see docs/BENCHMARK.md, controlled perturbation framework).
    """

    OMISSION = "omission"
    HALLUCINATION = "hallucination"
    NEGATION_TRAP = "negation_trap"
    TEMPORAL_TRAP = "temporal_trap"
    MEDICATION_TRAP = "medication_trap"
    HISTORICAL_CURRENT_TRAP = "historical_current_trap"
    CONFLICT_TRAP = "conflict_trap"
    REDUNDANCY_TRAP = "redundancy_trap"
