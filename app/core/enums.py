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
    """The benchmark's stress-test dimensions (see docs/ARCHITECTURE.md, Phase 5).

    Phase 1 ships exactly one hand-built case per type; later phases scale
    each type out to many randomized cases.
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


class FailureType(str, Enum):
    UNSUPPORTED_ADDITION = "unsupported_addition"
    CONTRADICTION = "contradiction"
    CRITICAL_OMISSION = "critical_omission"
    NON_CRITICAL_OMISSION = "non_critical_omission"
    NEGATION_REVERSAL = "negation_reversal"
    TEMPORAL_ERROR = "temporal_error"
    MEDICATION_ERROR = "medication_error"
    INCONSISTENT_ACROSS_RUNS = "inconsistent_across_runs"
    OTHER = "other"


class DataProvenance(str, Enum):
    """Where a benchmark case's content originated. Required on every case
    per the project's data-provenance requirement (see docs/ARCHITECTURE.md).
    """

    SYNTHETIC_TEMPLATE = "synthetic_template"
    SYNTHEA_DERIVED = "synthea_derived"
    MANUAL_SYNTHETIC = "manual_synthetic"
