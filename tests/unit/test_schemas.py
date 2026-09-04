"""Tests for the core data schemas.

These lock down the contract every other layer depends on: fact polarity,
medication structure, temporal structure, and the requirement that a case
carries provenance.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.enums import (
    CaseType,
    DataProvenance,
    FactImportance,
    FactPolarity,
    FactType,
    FindingType,
    MedicationStatus,
    RiskLevel,
    Severity,
    TemporalTrend,
)
from app.models.schemas import (
    AIOutput,
    BenchmarkCase,
    CaseResult,
    ClinicalFact,
    EvaluationRun,
    EvaluationFinding,
    MetricValue,
    MedicationFact,
    TemporalFact,
)


def _minimal_case(**overrides) -> BenchmarkCase:
    payload = dict(
        case_id="CASE_TEST",
        case_type=CaseType.NORMAL,
        benchmark_version="v1",
        provenance=DataProvenance.MANUAL_SYNTHETIC,
        age_group="45-54",
        sex="female",
        encounter_type="Office visit",
        clinical_note="Patient seen today.",
        reference_summary="Patient seen today.",
    )
    payload.update(overrides)
    return BenchmarkCase(**payload)


class TestClinicalFact:
    def test_defaults_to_affirmed_polarity(self):
        fact = ClinicalFact(
            fact_id="F001", type=FactType.CONDITION, content="Hypertension",
            importance=FactImportance.CRITICAL,
        )
        assert fact.polarity is FactPolarity.AFFIRMED

    def test_negated_polarity_is_preserved(self):
        fact = ClinicalFact(
            fact_id="F002", type=FactType.CONDITION, content="No history of diabetes",
            importance=FactImportance.CRITICAL, polarity=FactPolarity.NEGATED,
        )
        assert fact.polarity is FactPolarity.NEGATED

    def test_rejects_unknown_fact_type(self):
        with pytest.raises(ValidationError):
            ClinicalFact(
                fact_id="F003", type="not_a_real_type", content="x",
                importance=FactImportance.CRITICAL,
            )


class TestMedicationFact:
    def test_carries_structured_drug_fields(self):
        med = MedicationFact(
            fact_id="M001", content="Stopped amlodipine", importance=FactImportance.CRITICAL,
            drug_name="Amlodipine", status=MedicationStatus.DISCONTINUED, dose="5 mg daily",
        )
        assert med.type is FactType.MEDICATION
        assert med.drug_name == "Amlodipine"
        assert med.status is MedicationStatus.DISCONTINUED

    def test_requires_drug_name_and_status(self):
        with pytest.raises(ValidationError):
            MedicationFact(
                fact_id="M002", content="Some med", importance=FactImportance.CRITICAL,
            )


class TestTemporalFact:
    def test_records_both_values_and_trend(self):
        temporal = TemporalFact(
            fact_id="T001",
            content="Creatinine increased from 1.4 to 1.9 mg/dL",
            importance=FactImportance.CRITICAL,
            earlier_value="1.4 mg/dL", later_value="1.9 mg/dL", unit="mg/dL",
            trend=TemporalTrend.INCREASING,
        )
        assert temporal.type is FactType.TEMPORAL
        assert temporal.earlier_value != temporal.later_value
        assert temporal.trend is TemporalTrend.INCREASING

    def test_trend_defaults_to_unknown(self):
        temporal = TemporalFact(
            fact_id="T002", content="Creatinine checked", importance=FactImportance.NON_CRITICAL,
        )
        assert temporal.trend is TemporalTrend.UNKNOWN


class TestBenchmarkCase:
    def test_minimal_case_defaults_to_empty_annotation_lists(self):
        case = _minimal_case()
        assert case.critical_facts == []
        assert case.medication_facts == []
        assert case.negations == []
        assert case.expected_risk_flags == []

    def test_rejects_blank_case_id(self):
        with pytest.raises(ValidationError):
            _minimal_case(case_id="   ")

    def test_requires_provenance(self):
        with pytest.raises(ValidationError):
            BenchmarkCase(
                case_id="CASE_X", case_type=CaseType.NORMAL, benchmark_version="v1",
                age_group="45-54", sex="female", encounter_type="Office visit",
                clinical_note="x", reference_summary="x",
            )

    def test_round_trips_through_json(self):
        case = _minimal_case(
            critical_facts=[
                ClinicalFact(
                    fact_id="F001", type=FactType.LAB,
                    content="Creatinine increased from 1.4 to 1.9 mg/dL",
                    importance=FactImportance.CRITICAL,
                )
            ]
        )
        restored = BenchmarkCase.model_validate(case.model_dump(mode="json"))
        assert restored == case


class TestEvaluationSchemas:
    def test_finding_records_its_evaluator_and_evidence(self):
        """Phase 3 replaced the Phase 1 `Failure` placeholder: a finding must
        now identify the evaluator that produced it and carry both sides of
        the evidence, so it can be judged without re-reading the case."""
        finding = EvaluationFinding(
            finding_id="CASE_002-critical_retention-000",
            case_id="CASE_002",
            evaluator_name="critical_retention",
            evaluator_version="0.3.0",
            finding_type=FindingType.CRITICAL_OMISSION,
            explanation="Critical lab trend omitted from summary.",
            source_evidence="Creatinine increased from 1.4 to 1.9 mg/dL",
            output_evidence="Creatinine was 1.4 mg/dL.",
            fact_id="CASE_002-F001",
        )
        assert finding.source_evidence and finding.output_evidence
        assert finding.evaluator_name == "critical_retention"

    def test_finding_severity_is_unset_until_the_policy_assigns_it(self):
        """Evaluators detect; the severity policy classifies."""
        finding = EvaluationFinding(
            finding_id="F", case_id="C", evaluator_name="e", evaluator_version="1",
            finding_type=FindingType.CRITICAL_OMISSION, explanation="x",
        )
        assert finding.severity is None

    def test_finding_requires_its_identifying_fields(self):
        with pytest.raises(ValidationError):
            EvaluationFinding(
                finding_type=FindingType.CRITICAL_OMISSION,
                explanation="missing identity fields",
            )

    def test_finding_confidence_is_bounded(self):
        with pytest.raises(ValidationError):
            EvaluationFinding(
                finding_id="F", case_id="C", evaluator_name="e", evaluator_version="1",
                finding_type=FindingType.OTHER, explanation="x", confidence=1.5,
            )

    def test_metric_value_reports_undefined_rather_than_zero(self):
        """A metric with no denominator has no value - not 0.0, not 1.0."""
        metric = MetricValue(
            name="critical_information_retention", value=None,
            numerator=0, denominator=0, definition="retained / total",
        )
        assert metric.value is None
        assert not metric.is_defined

    def test_ai_output_defaults_run_index_to_zero(self):
        output = AIOutput(case_id="CASE_001", model_id="model-a", summary_text="A summary.")
        assert output.run_index == 0
        assert output.generated_at is not None

    def test_case_result_collects_findings_and_metrics(self):
        result = CaseResult(
            evaluation_run_id="RUN_1", case_id="CASE_002",
            metrics={
                "critical_information_retention": MetricValue(
                    name="critical_information_retention", value=0.5,
                    numerator=1, denominator=2, definition="retained / total",
                )
            },
            findings=[
                EvaluationFinding(
                    finding_id="CASE_002-critical_retention-000",
                    case_id="CASE_002", evaluator_name="critical_retention",
                    evaluator_version="0.3.0",
                    finding_type=FindingType.CRITICAL_OMISSION,
                    explanation="Omitted renal decline.",
                    source_evidence="Creatinine increased from 1.4 to 1.9 mg/dL",
                    severity=Severity.HIGH,
                )
            ],
        )
        assert result.metrics["critical_information_retention"].value == 0.5
        assert len(result.findings) == 1

    def test_evaluation_run_records_reproducibility_fields(self):
        run = EvaluationRun(
            id="RUN_1", model_id="model-a", benchmark_id="synthetic-default",
            benchmark_version="v1", evaluation_version="0.1.0",
            prompt_version="p1", random_seed=42,
            configuration={"temperature": 0.0}, risk=RiskLevel.MODERATE,
        )
        assert run.random_seed == 42
        assert run.benchmark_version == "v1"
        assert run.evaluation_version == "0.1.0"
        assert run.risk is RiskLevel.MODERATE
