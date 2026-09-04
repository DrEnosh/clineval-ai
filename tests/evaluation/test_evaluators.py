"""Tests for the deterministic evaluators.

Each evaluator gets positive tests (a correct summary must not be flagged),
negative tests (a known defect must be detected), and boundary tests (empty
output, aliases, tolerance, missing values, repeated mentions).

The positive tests matter as much as the negative ones. An evaluator that
flags every textual difference is useless, so "wording differs, meaning
preserved" is tested explicitly throughout.
"""

from __future__ import annotations

import pytest

from app.core.enums import (
    AbnormalFlag,
    AllergySeverity,
    ConditionStatus,
    DifficultyTier,
    FactImportance,
    FactPolarity,
    FactType,
    FindingType,
    MedicationStatus,
    TemporalTrend,
)
from app.evaluation.evaluators import (
    CriticalRetentionEvaluator,
    FactSupportEvaluator,
    MedicationConsistencyEvaluator,
    NegationConsistencyEvaluator,
    TemporalConsistencyEvaluator,
)
from app.models.schemas import AIOutput, BenchmarkCase, ClinicalFact, MedicationFact
from app.models.structured import (
    AllergyEntry,
    ClinicalFindingEntry,
    ConditionEntry,
    ConflictEntry,
    DiagnosisEntry,
    EncounterInfo,
    LabResultEntry,
    LabTrendEntry,
    MedicationEntry,
    PatientProfile,
    ProcedureEntry,
    StructuredCase,
    VitalSignEntry,
)
from app.core.enums import CaseType, DataProvenance


def _structured(**overrides) -> StructuredCase:
    payload = dict(
        case_id="SYN_T",
        seed="t:0",
        generator_version="0.2.0",
        patient=PatientProfile(patient_id="SYN-PT-00001", age=62, age_group="55-64", sex="female"),
        encounter=EncounterInfo(
            encounter_id="SYN-ENC-1", encounter_type="Nephrology follow-up",
            setting="outpatient_specialty",
        ),
        difficulty=DifficultyTier.MODERATE,
    )
    payload.update(overrides)
    return StructuredCase(**payload)


def _case(structured: StructuredCase, **overrides) -> BenchmarkCase:
    payload = dict(
        case_id=structured.case_id,
        case_type=CaseType.NORMAL,
        benchmark_version="test-v1",
        provenance=DataProvenance.SYNTHETIC_TEMPLATE,
        age_group="55-64",
        sex="female",
        encounter_type="Nephrology follow-up",
        clinical_note="Source note.",
        reference_summary="Reference summary.",
        structured_case=structured,
    )
    payload.update(overrides)
    return BenchmarkCase(**payload)


def _output(text: str) -> AIOutput:
    return AIOutput(case_id="SYN_T", model_id="test", summary_text=text)


def _critical_fact(fact_id: str, content: str, source_ref: str, rule: str = "CR-01") -> ClinicalFact:
    return ClinicalFact(
        fact_id=fact_id, type=FactType.LAB, content=content,
        importance=FactImportance.CRITICAL, source_ref=source_ref, criticality_rule=rule,
    )


# ---------------------------------------------------------------------------
# Critical retention
# ---------------------------------------------------------------------------


class TestCriticalRetention:
    @pytest.fixture
    def evaluator(self):
        return CriticalRetentionEvaluator()

    @pytest.fixture
    def case(self):
        structured = _structured(
            labs=[
                LabResultEntry(
                    entity_id="LAB-001", name="Creatinine", value=1.9, unit="mg/dL",
                    abnormal_flag=AbnormalFlag.HIGH,
                )
            ],
            allergies=[
                AllergyEntry(
                    entity_id="ALG-001", substance="Penicillin", reaction="rash",
                    severity=AllergySeverity.SEVERE,
                )
            ],
        )
        return _case(
            structured,
            critical_facts=[
                _critical_fact("F001", "Creatinine 1.9 mg/dL", "LAB-001"),
                _critical_fact("F002", "Allergy: Penicillin", "ALG-001", "CR-05"),
            ],
        )

    def test_correct_summary_produces_no_findings(self, evaluator, case):
        result = evaluator.evaluate(
            case, _output("Creatinine is 1.9 mg/dL. Severe allergy: Penicillin (rash).")
        )
        assert result.findings == []
        assert result.metrics["critical_information_retention"].value == 1.0

    def test_omitted_fact_is_detected_with_its_id(self, evaluator, case):
        result = evaluator.evaluate(case, _output("Creatinine is 1.9 mg/dL."))
        omissions = [f for f in result.findings if f.finding_type is FindingType.CRITICAL_OMISSION]
        assert len(omissions) == 1
        assert omissions[0].fact_id == "F002"
        assert omissions[0].source_ref == "ALG-001"

    def test_retention_metric_reflects_omission(self, evaluator, case):
        result = evaluator.evaluate(case, _output("Creatinine is 1.9 mg/dL."))
        metric = result.metrics["critical_information_retention"]
        assert metric.value == 0.5
        assert (metric.numerator, metric.denominator) == (1.0, 2.0)

    def test_findings_carry_source_and_output_evidence(self, evaluator, case):
        result = evaluator.evaluate(case, _output("Nothing relevant here."))
        for finding in result.findings:
            assert finding.source_evidence
            assert finding.evaluator_name == "critical_retention"
            assert finding.evaluator_version
            assert finding.explanation

    def test_wrong_value_is_a_contradiction_not_an_omission(self, evaluator, case):
        result = evaluator.evaluate(
            case, _output("Creatinine is 0.7 mg/dL. Severe allergy: Penicillin (rash).")
        )
        types = {f.finding_type for f in result.findings}
        assert FindingType.CONTRADICTION in types
        assert result.metrics["critical_contradiction_rate"].value == 0.5

    def test_alias_in_output_still_counts_as_retained(self, evaluator, case):
        result = evaluator.evaluate(
            case, _output("Cr is 1.9 mg/dL. Severe allergy: Penicillin (rash).")
        )
        assert result.metrics["critical_information_retention"].value == 1.0

    def test_empty_output_flags_once_and_reports_zero_retention(self, evaluator, case):
        result = evaluator.evaluate(case, _output(""))
        assert [f.finding_type for f in result.findings] == [FindingType.EMPTY_OUTPUT]
        assert result.metrics["critical_information_retention"].value == 0.0

    def test_whitespace_only_output_is_treated_as_empty(self, evaluator, case):
        result = evaluator.evaluate(case, _output("   \n\t  "))
        assert [f.finding_type for f in result.findings] == [FindingType.EMPTY_OUTPUT]

    def test_case_with_no_critical_facts_reports_undefined(self, evaluator):
        case = _case(_structured(), critical_facts=[])
        result = evaluator.evaluate(case, _output("A summary."))
        assert result.metrics["critical_information_retention"].value is None

    def test_dangling_source_ref_is_reported_not_swallowed(self, evaluator):
        case = _case(
            _structured(),
            critical_facts=[_critical_fact("F001", "Something", "NOPE-999")],
        )
        result = evaluator.evaluate(case, _output("A summary."))
        assert [f.finding_type for f in result.findings] == [FindingType.MISSING_ANNOTATION]

    def test_repeated_mention_is_not_double_counted(self, evaluator, case):
        result = evaluator.evaluate(
            case,
            _output(
                "Creatinine is 1.9 mg/dL. Creatinine 1.9 again. Severe allergy: Penicillin (rash)."
            ),
        )
        assert result.findings == []


# ---------------------------------------------------------------------------
# Medication consistency
# ---------------------------------------------------------------------------


class TestMedicationConsistency:
    @pytest.fixture
    def evaluator(self):
        return MedicationConsistencyEvaluator()

    @pytest.fixture
    def case(self):
        structured = _structured(
            medications=[
                MedicationEntry(
                    entity_id="MED-001", drug_name="Lisinopril", status=MedicationStatus.CURRENT,
                    dose="20 mg", route="oral", frequency="daily",
                ),
                MedicationEntry(
                    entity_id="MED-002", drug_name="Amlodipine",
                    status=MedicationStatus.DISCONTINUED, dose="5 mg",
                    change_timing="1 month ago",
                ),
                MedicationEntry(
                    entity_id="MED-003", drug_name="Furosemide",
                    status=MedicationStatus.CHANGED_DOSE, dose="40 mg", prior_dose="20 mg",
                ),
            ]
        )
        return _case(structured)

    @pytest.fixture
    def correct_summary(self):
        return (
            "Continued on Lisinopril 20 mg. "
            "Amlodipine was discontinued 1 month ago. "
            "Furosemide was increased from 20 mg to 40 mg."
        )

    def test_correct_summary_produces_no_findings(self, evaluator, case, correct_summary):
        result = evaluator.evaluate(case, _output(correct_summary))
        assert result.findings == []
        assert result.metrics["medication_consistency"].value == 1.0

    def test_omitted_medication_is_detected(self, evaluator, case, correct_summary):
        summary = correct_summary.replace("Continued on Lisinopril 20 mg. ", "")
        result = evaluator.evaluate(case, _output(summary))
        omissions = [f for f in result.findings if f.finding_type is FindingType.MEDICATION_OMISSION]
        assert len(omissions) == 1
        assert omissions[0].details["drug"] == "Lisinopril"

    def test_discontinued_drug_reported_as_current_is_a_status_error(self, evaluator, case):
        summary = (
            "Continued on Lisinopril 20 mg. Amlodipine is continued at 5 mg. "
            "Furosemide was increased from 20 mg to 40 mg."
        )
        result = evaluator.evaluate(case, _output(summary))
        errors = [f for f in result.findings if f.finding_type is FindingType.MEDICATION_STATUS_ERROR]
        assert len(errors) == 1
        assert errors[0].details["drug"] == "Amlodipine"
        assert errors[0].details["expected_status"] == "discontinued"

    def test_wrong_dose_is_detected(self, evaluator, case, correct_summary):
        summary = correct_summary.replace("Lisinopril 20 mg", "Lisinopril 80 mg")
        result = evaluator.evaluate(case, _output(summary))
        assert any(f.finding_type is FindingType.MEDICATION_DOSE_ERROR for f in result.findings)

    def test_invented_drug_is_detected(self, evaluator, case, correct_summary):
        result = evaluator.evaluate(
            case, _output(correct_summary + " Also started on Warfarin 5 mg.")
        )
        hallucinations = [
            f for f in result.findings if f.finding_type is FindingType.MEDICATION_HALLUCINATION
        ]
        assert len(hallucinations) == 1
        assert hallucinations[0].details["drug"] == "Warfarin"

    def test_dose_change_stating_both_doses_is_correct(self, evaluator, case, correct_summary):
        """Naming the prior dose alongside the current one is correct reporting
        of the change, not a dose error."""
        result = evaluator.evaluate(case, _output(correct_summary))
        assert not any(f.finding_type is FindingType.MEDICATION_DOSE_ERROR for f in result.findings)

    def test_dose_change_stating_only_the_prior_dose_is_an_error(self, evaluator, case):
        summary = (
            "Continued on Lisinopril 20 mg. Amlodipine was discontinued 1 month ago. "
            "Furosemide 20 mg was increased."
        )
        result = evaluator.evaluate(case, _output(summary))
        assert any(f.finding_type is FindingType.MEDICATION_DOSE_ERROR for f in result.findings)

    def test_time_expressions_are_not_read_as_doses(self, evaluator, case):
        """Regression: 'discontinued 2 weeks ago' had its '2' compared against
        the drug's dose, reporting a false dose error."""
        structured = _structured(
            medications=[
                MedicationEntry(
                    entity_id="MED-001", drug_name="Insulin glargine",
                    status=MedicationStatus.DISCONTINUED, dose="30 units",
                    change_timing="2 weeks ago",
                )
            ]
        )
        result = evaluator.evaluate(
            _case(structured), _output("Insulin glargine was discontinued 2 weeks ago.")
        )
        assert result.findings == []

    def test_neighbouring_sentence_cue_does_not_leak(self, evaluator, case):
        """Regression: a fixed character window let one drug's status cue be
        read for a different drug in an adjacent sentence."""
        summary = (
            "Nitrofurantoin 100 mg was newly started. Continued on Lisinopril 20 mg. "
            "Amlodipine was discontinued 1 month ago. "
            "Furosemide was increased from 20 mg to 40 mg."
        )
        result = evaluator.evaluate(case, _output(summary))
        status_errors = [
            f for f in result.findings if f.finding_type is FindingType.MEDICATION_STATUS_ERROR
        ]
        assert status_errors == []

    def test_drug_alias_is_recognised(self, evaluator):
        structured = _structured(
            medications=[
                MedicationEntry(
                    entity_id="MED-001", drug_name="Hydrochlorothiazide",
                    status=MedicationStatus.CURRENT, dose="25 mg",
                )
            ]
        )
        result = evaluator.evaluate(_case(structured), _output("Continued on HCTZ 25 mg."))
        assert not any(
            f.finding_type is FindingType.MEDICATION_OMISSION for f in result.findings
        )

    def test_empty_output_reports_all_omitted(self, evaluator, case):
        result = evaluator.evaluate(case, _output(""))
        assert [f.finding_type for f in result.findings] == [FindingType.EMPTY_OUTPUT]
        assert result.metrics["medication_omission_rate"].value == 1.0

    def test_case_with_no_medications_is_undefined(self, evaluator):
        result = evaluator.evaluate(_case(_structured()), _output("A summary."))
        assert result.metrics["medication_consistency"].value is None


# ---------------------------------------------------------------------------
# Negation consistency
# ---------------------------------------------------------------------------


class TestNegationConsistency:
    @pytest.fixture
    def evaluator(self):
        return NegationConsistencyEvaluator()

    @pytest.fixture
    def case(self):
        structured = _structured(
            findings=[
                ClinicalFindingEntry(
                    entity_id="FIND-001", description="no history of diabetes", present=False,
                )
            ]
        )
        return _case(
            structured,
            negations=[
                ClinicalFact(
                    fact_id="N001", type=FactType.OBSERVATION,
                    content="No history of diabetes", importance=FactImportance.CRITICAL,
                    polarity=FactPolarity.NEGATED, source_ref="FIND-001",
                    criticality_rule="CR-08",
                )
            ],
        )

    def test_preserved_negation_produces_no_findings(self, evaluator, case):
        result = evaluator.evaluate(case, _output("Documented negatives: no history of diabetes."))
        assert result.findings == []
        assert result.metrics["negation_consistency"].value == 1.0

    def test_reversed_negation_is_detected(self, evaluator, case):
        """The signature failure: the source denies it, the output asserts it."""
        result = evaluator.evaluate(case, _output("The patient has a history of diabetes."))
        reversals = [f for f in result.findings if f.finding_type is FindingType.NEGATION_REVERSAL]
        assert len(reversals) == 1
        assert reversals[0].fact_id == "N001"
        assert result.metrics["negation_reversal_rate"].value == 1.0

    def test_reversal_is_not_reported_as_a_mere_omission(self, evaluator, case):
        """Regression: matching the full wording 'no history of diabetes' meant
        a reversal failed to match and was reported as an omission, inverting
        the severity of the worst failure mode the benchmark models."""
        result = evaluator.evaluate(case, _output("Patient has diabetes."))
        types = {f.finding_type for f in result.findings}
        assert FindingType.NEGATION_REVERSAL in types
        assert FindingType.NEGATION_OMISSION not in types

    def test_omitted_negation_is_not_a_reversal(self, evaluator, case):
        """Dropping a negative loses information but asserts nothing false."""
        result = evaluator.evaluate(case, _output("Patient seen for follow-up."))
        assert result.metrics["negation_reversal_rate"].value == 0.0
        assert [f.finding_type for f in result.findings] == [FindingType.NEGATION_OMISSION]

    def test_alias_reversal_is_still_detected(self, evaluator):
        structured = _structured(
            findings=[
                ClinicalFindingEntry(
                    entity_id="FIND-001", description="no history of GERD", present=False
                )
            ]
        )
        case = _case(
            structured,
            negations=[
                ClinicalFact(
                    fact_id="N001", type=FactType.OBSERVATION, content="No history of GERD",
                    importance=FactImportance.CRITICAL, polarity=FactPolarity.NEGATED,
                    source_ref="FIND-001",
                )
            ],
        )
        result = evaluator.evaluate(
            case, _output("Patient has gastroesophageal reflux disease.")
        )
        assert any(f.finding_type is FindingType.NEGATION_REVERSAL for f in result.findings)

    def test_empty_output_reverses_nothing(self, evaluator, case):
        result = evaluator.evaluate(case, _output(""))
        assert result.metrics["negation_reversal_rate"].value == 0.0

    def test_case_with_no_negations_is_undefined(self, evaluator):
        result = evaluator.evaluate(_case(_structured()), _output("A summary."))
        assert result.metrics["negation_consistency"].value is None


# ---------------------------------------------------------------------------
# Temporal consistency
# ---------------------------------------------------------------------------


class TestTemporalConsistency:
    @pytest.fixture
    def evaluator(self):
        return TemporalConsistencyEvaluator()

    @pytest.fixture
    def case(self):
        structured = _structured(
            labs=[
                LabResultEntry(
                    entity_id="LAB-001", name="Creatinine", value=1.4, unit="mg/dL",
                    day_offset=-90, timing_description="3 months ago",
                ),
                LabResultEntry(
                    entity_id="LAB-002", name="Creatinine", value=1.9, unit="mg/dL",
                    abnormal_flag=AbnormalFlag.HIGH,
                ),
            ],
            lab_trends=[
                LabTrendEntry(
                    entity_id="TRND-001", lab_name="Creatinine", unit="mg/dL",
                    earlier_ref="LAB-001", later_ref="LAB-002",
                    earlier_value=1.4, later_value=1.9,
                    earlier_timing="3 months ago", later_timing="today",
                    trend=TemporalTrend.INCREASING, clinically_significant=True,
                )
            ],
        )
        return _case(structured)

    def test_both_endpoints_preserved_produces_no_findings(self, evaluator, case):
        result = evaluator.evaluate(
            case, _output("Creatinine rose from 1.4 to 1.9 mg/dL between 3 months ago and today.")
        )
        assert result.findings == []
        assert result.metrics["temporal_consistency"].value == 1.0

    def test_reversed_order_with_correct_direction_is_accepted(self, evaluator, case):
        """'rose to 1.9 from 1.4' states the same trend; only the direction
        word decides correctness."""
        result = evaluator.evaluate(case, _output("Creatinine rose to 1.9 from 1.4 mg/dL."))
        assert result.findings == []

    def test_reversed_trend_is_detected(self, evaluator, case):
        result = evaluator.evaluate(case, _output("Creatinine fell from 1.9 to 1.4 mg/dL."))
        assert any(
            f.finding_type is FindingType.TEMPORAL_TREND_REVERSAL for f in result.findings
        )

    def test_historical_value_reported_as_current_is_value_confusion(self, evaluator, case):
        """Reporting the stale value actively misinforms, unlike merely
        dropping the earlier one."""
        result = evaluator.evaluate(case, _output("Creatinine was 1.4 mg/dL."))
        assert any(
            f.finding_type is FindingType.TEMPORAL_VALUE_CONFUSION for f in result.findings
        )

    def test_only_current_value_is_information_loss(self, evaluator, case):
        result = evaluator.evaluate(case, _output("Creatinine is 1.9 mg/dL."))
        assert any(
            f.finding_type is FindingType.TEMPORAL_INFORMATION_LOSS for f in result.findings
        )
        assert result.metrics["temporal_information_loss_rate"].value == 1.0

    def test_missing_analyte_is_information_loss(self, evaluator, case):
        result = evaluator.evaluate(case, _output("Patient seen in clinic."))
        assert result.metrics["temporal_consistency"].value == 0.0

    def test_resolved_condition_presented_as_current_is_detected(self, evaluator):
        structured = _structured(
            conditions=[
                ConditionEntry(
                    entity_id="COND-001", name="Cellulitis", status=ConditionStatus.RESOLVED,
                    onset_description="resolved 2 years ago",
                )
            ]
        )
        result = evaluator.evaluate(
            _case(structured), _output("Active problems include cellulitis.")
        )
        assert any(
            f.finding_type is FindingType.HISTORICAL_CURRENT_CONFUSION for f in result.findings
        )

    def test_resolved_condition_correctly_framed_is_accepted(self, evaluator):
        structured = _structured(
            conditions=[
                ConditionEntry(
                    entity_id="COND-001", name="Cellulitis", status=ConditionStatus.RESOLVED,
                    onset_description="resolved 2 years ago",
                )
            ]
        )
        result = evaluator.evaluate(
            _case(structured), _output("Historical, not active: cellulitis (resolved 2 years ago).")
        )
        assert result.findings == []

    def test_superseded_statement_reported_as_current_is_detected(self, evaluator):
        structured = _structured(
            conflicts=[
                ConflictEntry(
                    entity_id="CONF-001", topic="chest pain",
                    earlier_statement="patient denies chest pain",
                    earlier_timing="on initial assessment",
                    later_statement="patient reports chest pain",
                    later_timing="on reassessment",
                    resolution="The later assessment supersedes the initial history.",
                )
            ]
        )
        result = evaluator.evaluate(
            _case(structured), _output("Patient denies chest pain.")
        )
        assert any(
            f.finding_type is FindingType.CONFLICT_RESOLUTION_ERROR for f in result.findings
        )

    def test_case_with_no_trends_is_undefined(self, evaluator):
        result = evaluator.evaluate(_case(_structured()), _output("A summary."))
        assert result.metrics["temporal_consistency"].value is None


# ---------------------------------------------------------------------------
# Fact support
# ---------------------------------------------------------------------------


class TestFactSupport:
    @pytest.fixture
    def evaluator(self):
        return FactSupportEvaluator()

    @pytest.fixture
    def case(self):
        structured = _structured(
            conditions=[
                ConditionEntry(
                    entity_id="COND-001", name="Essential hypertension", is_primary=True
                )
            ],
            diagnoses=[
                DiagnosisEntry(entity_id="DX-001", name="Essential hypertension", is_primary=True)
            ],
            vitals=[
                VitalSignEntry(
                    entity_id="VIT-001", name="Blood pressure", value="168/94", unit="mmHg",
                    abnormal_flag=AbnormalFlag.HIGH,
                )
            ],
            procedures=[
                ProcedureEntry(entity_id="PROC-001", name="Chest X-ray", performed=False)
            ],
        )
        return _case(structured)

    def test_faithful_summary_has_no_contradictions(self, evaluator, case):
        result = evaluator.evaluate(
            case,
            _output(
                "Patient with essential hypertension. Blood pressure 168/94 mmHg. "
                "Chest X-ray was not performed."
            ),
        )
        assert result.findings == []

    def test_unsupported_condition_is_detected(self, evaluator, case):
        result = evaluator.evaluate(
            case, _output("Patient with essential hypertension and cellulitis.")
        )
        additions = [f for f in result.findings if f.finding_type is FindingType.UNSUPPORTED_ADDITION]
        assert any(f.details.get("term") == "Cellulitis" for f in additions)

    def test_negated_unsupported_term_is_not_a_hallucination(self, evaluator, case):
        """'no evidence of pneumonia' for a case without pneumonia is correct."""
        result = evaluator.evaluate(
            case,
            _output(
                "Patient with essential hypertension. No evidence of pneumonia. "
                "Blood pressure 168/94 mmHg. Chest X-ray was not performed."
            ),
        )
        assert not any(
            f.details.get("term") == "Community-acquired pneumonia" for f in result.findings
        )

    def test_hyphenated_condition_present_in_case_is_supported(self, evaluator):
        """Regression: a normalized term was compared against a raw corpus, so
        'Community-acquired pneumonia' was reported unsupported in a case that
        contains it."""
        structured = _structured(
            conditions=[
                ConditionEntry(
                    entity_id="COND-001", name="Community-acquired pneumonia", is_primary=True
                )
            ]
        )
        result = evaluator.evaluate(
            _case(structured), _output("Patient with community-acquired pneumonia.")
        )
        assert not any(
            f.finding_type is FindingType.UNSUPPORTED_ADDITION for f in result.findings
        )

    def test_fabricated_value_is_detected(self, evaluator, case):
        result = evaluator.evaluate(
            case,
            _output("Essential hypertension. Blood pressure 168/94 mmHg. Creatinine was 7.77."),
        )
        assert any(f.details.get("value") == "7.77" for f in result.findings)

    def test_procedure_not_performed_reported_as_done_is_a_contradiction(self, evaluator, case):
        result = evaluator.evaluate(
            case,
            _output(
                "Essential hypertension. Blood pressure 168/94 mmHg. "
                "Chest X-ray showed no acute abnormality."
            ),
        )
        assert any(f.finding_type is FindingType.CONTRADICTION for f in result.findings)

    def test_harmless_extra_prose_is_not_flagged(self, evaluator, case):
        """Non-clinical padding must not be scored as hallucination."""
        result = evaluator.evaluate(
            case,
            _output(
                "The patient was seen today and appeared comfortable. "
                "Essential hypertension. Blood pressure 168/94 mmHg. "
                "Chest X-ray was not performed. Follow-up arranged."
            ),
        )
        assert result.findings == []

    def test_empty_output_has_undefined_unsupported_rate(self, evaluator, case):
        """An empty output asserts nothing, so nothing can be unsupported."""
        result = evaluator.evaluate(case, _output(""))
        assert result.metrics["unsupported_fact_rate"].value is None
        assert result.metrics["supported_fact_rate"].value == 0.0

    def test_case_without_structured_source_does_not_crash(self, evaluator):
        case = _case(_structured())
        case.structured_case = None
        result = evaluator.evaluate(case, _output("Some summary."))
        assert result.error is None


# ---------------------------------------------------------------------------
# Cross-evaluator robustness
# ---------------------------------------------------------------------------


ALL_EVALUATORS = [
    CriticalRetentionEvaluator,
    MedicationConsistencyEvaluator,
    NegationConsistencyEvaluator,
    TemporalConsistencyEvaluator,
    FactSupportEvaluator,
]


class TestRobustness:
    """No evaluator may crash on unexpected model output."""

    @pytest.mark.parametrize("evaluator_class", ALL_EVALUATORS)
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "\n\n\t",
            "<html><body>Not a summary</body></html>",
            '{"error": "model refused"}',
            "1234567890" * 500,
            "!!!???...",
            "Le patient a une insuffisance renale chronique.",
        ],
        ids=[
            "empty", "spaces", "whitespace", "html", "json_error",
            "very_long", "punctuation", "other_language",
        ],
    )
    def test_evaluators_never_raise(self, evaluator_class, text):
        evaluator = evaluator_class()
        structured = _structured(
            medications=[
                MedicationEntry(
                    entity_id="MED-001", drug_name="Lisinopril",
                    status=MedicationStatus.CURRENT, dose="20 mg",
                )
            ],
            labs=[
                LabResultEntry(entity_id="LAB-001", name="Creatinine", value=1.9, unit="mg/dL")
            ],
        )
        case = _case(structured, critical_facts=[_critical_fact("F1", "Creatinine 1.9", "LAB-001")])
        result = evaluator.evaluate(case, _output(text))
        assert result.error is None
        assert result.evaluator_name == evaluator_class.name

    @pytest.mark.parametrize("evaluator_class", ALL_EVALUATORS)
    def test_evaluators_handle_a_case_with_no_content(self, evaluator_class):
        evaluator = evaluator_class()
        result = evaluator.evaluate(_case(_structured()), _output("A summary."))
        assert result.error is None

    @pytest.mark.parametrize("evaluator_class", ALL_EVALUATORS)
    def test_findings_are_deterministic(self, evaluator_class):
        evaluator = evaluator_class()
        structured = _structured(
            labs=[LabResultEntry(entity_id="LAB-001", name="Creatinine", value=1.9, unit="mg/dL")]
        )
        case = _case(structured, critical_facts=[_critical_fact("F1", "Creatinine 1.9", "LAB-001")])
        first = evaluator.evaluate(case, _output("Nothing relevant."))
        second = evaluator.evaluate(case, _output("Nothing relevant."))
        assert [f.model_dump() for f in first.findings] == [f.model_dump() for f in second.findings]

    @pytest.mark.parametrize("evaluator_class", ALL_EVALUATORS)
    def test_evaluators_never_assign_severity_themselves(self, evaluator_class):
        """Detection and classification are separate layers; the runner
        applies the severity policy."""
        evaluator = evaluator_class()
        structured = _structured(
            labs=[LabResultEntry(entity_id="LAB-001", name="Creatinine", value=1.9, unit="mg/dL")]
        )
        case = _case(structured, critical_facts=[_critical_fact("F1", "Creatinine 1.9", "LAB-001")])
        result = evaluator.evaluate(case, _output(""))
        assert all(finding.severity is None for finding in result.findings)
