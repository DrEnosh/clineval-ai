"""Tests for the entity support analyzers and the injection harness."""

from __future__ import annotations

import pytest

from app.core.enums import (
    AbnormalFlag,
    AllergySeverity,
    ConditionStatus,
    DifficultyTier,
    FindingSignificance,
    MedicationStatus,
    SupportStatus,
    TemporalTrend,
)
from app.benchmark import generate_benchmark
from app.evaluation.injection import INJECTIONS, inject, inject_all
from app.evaluation.matching import FactMatcher
from app.evaluation.support import (
    analyze_entity,
    detect_medication_status,
    dose_quantities_near,
    supported_entity_kinds,
)
from app.models.structured import (
    AllergyEntry,
    ClinicalFindingEntry,
    ConditionEntry,
    ConflictEntry,
    DiagnosisEntry,
    EncounterInfo,
    FollowUpPlanEntry,
    LabResultEntry,
    LabTrendEntry,
    MedicationEntry,
    PatientProfile,
    ProcedureEntry,
    StructuredCase,
    VitalSignEntry,
)


@pytest.fixture
def matcher() -> FactMatcher:
    return FactMatcher()


class TestDispatch:
    def test_every_expected_entity_type_has_an_analyzer(self):
        kinds = supported_entity_kinds()
        for expected in (
            "ConditionEntry", "MedicationEntry", "LabResultEntry", "VitalSignEntry",
            "ProcedureEntry", "ClinicalFindingEntry", "AllergyEntry", "LabTrendEntry",
            "DiagnosisEntry", "ConflictEntry",
        ):
            assert expected in kinds

    def test_unsupported_entity_type_returns_none_rather_than_raising(self, matcher):
        """A new structured type must not crash evaluation before its analyzer
        is written."""
        plan = FollowUpPlanEntry(entity_id="FU-1", interval="2 weeks", provider="primary care")
        assert analyze_entity(plan, "Follow up in 2 weeks.", matcher) is None

    def test_analyze_entity_creates_a_matcher_when_none_is_given(self):
        condition = ConditionEntry(entity_id="C-1", name="Asthma")
        assert analyze_entity(condition, "Patient has asthma.") is not None


class TestConditionAndDiagnosis:
    def test_active_condition_mentioned_is_supported(self, matcher):
        entity = ConditionEntry(entity_id="C-1", name="Asthma")
        assert analyze_entity(entity, "Patient has asthma.", matcher).status is SupportStatus.SUPPORTED

    def test_active_condition_negated_is_contradicted(self, matcher):
        entity = ConditionEntry(entity_id="C-1", name="Asthma")
        assert analyze_entity(entity, "No history of asthma.", matcher).status is SupportStatus.CONTRADICTED

    def test_absent_condition_is_omitted(self, matcher):
        entity = ConditionEntry(entity_id="C-1", name="Asthma")
        assert analyze_entity(entity, "Patient is well.", matcher).status is SupportStatus.OMITTED

    def test_resolved_condition_framed_as_past_is_supported(self, matcher):
        entity = ConditionEntry(
            entity_id="C-1", name="Cellulitis", status=ConditionStatus.RESOLVED,
            onset_description="resolved 2 years ago",
        )
        result = analyze_entity(entity, "Historical, not active: cellulitis.", matcher)
        assert result.status is SupportStatus.SUPPORTED

    def test_resolved_condition_framed_as_current_is_contradicted(self, matcher):
        entity = ConditionEntry(
            entity_id="C-1", name="Cellulitis", status=ConditionStatus.RESOLVED
        )
        result = analyze_entity(entity, "Active problems include cellulitis.", matcher)
        assert result.status is SupportStatus.CONTRADICTED

    def test_diagnosis_negated_is_contradicted(self, matcher):
        entity = DiagnosisEntry(entity_id="D-1", name="Sepsis", is_primary=True)
        assert analyze_entity(entity, "No evidence of sepsis.", matcher).status is SupportStatus.CONTRADICTED


class TestMedicationStatusDetection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Lisinopril was discontinued last month.", "discontinued"),
            ("Lisinopril is no longer taking effect on the regimen.", "discontinued"),
            ("Lisinopril 20 mg was newly started.", "new"),
            ("Lisinopril was increased from 10 mg to 20 mg.", "changed_dose"),
            ("Lisinopril was taken in the past.", "historical"),
            ("Lisinopril is not a current medication.", "historical"),
            ("Continued on Lisinopril 20 mg.", "current"),
        ],
    )
    def test_status_cues(self, text, expected):
        assert detect_medication_status("Lisinopril", text) == expected

    def test_absent_drug_has_no_status(self):
        assert detect_medication_status("Lisinopril", "Patient is well.") is None

    def test_drug_named_without_a_cue_has_no_status(self):
        assert detect_medication_status("Lisinopril", "Lisinopril 20 mg.") is None

    def test_discontinued_is_checked_before_current(self):
        """'no longer taking' contains 'taking', so order matters."""
        assert detect_medication_status("Lisinopril", "No longer taking Lisinopril.") == "discontinued"


class TestDoseQuantities:
    def test_extracts_dose_units_only(self):
        text = "Lisinopril 20 mg was started 2 weeks ago."
        assert dose_quantities_near("Lisinopril", text) == [20.0]

    def test_handles_several_units(self):
        assert dose_quantities_near("Insulin glargine", "Insulin glargine 30 units nightly.") == [30.0]

    def test_returns_nothing_when_no_dose_is_stated(self):
        assert dose_quantities_near("Lisinopril", "Lisinopril was discontinued 2 weeks ago.") == []


class TestMedicationSupport:
    def test_current_medication_present_is_supported(self, matcher):
        entity = MedicationEntry(
            entity_id="M-1", drug_name="Lisinopril", status=MedicationStatus.CURRENT, dose="20 mg"
        )
        assert analyze_entity(entity, "Continued on Lisinopril 20 mg.", matcher).status is SupportStatus.SUPPORTED

    def test_absent_medication_is_omitted(self, matcher):
        entity = MedicationEntry(entity_id="M-1", drug_name="Lisinopril", status=MedicationStatus.CURRENT)
        assert analyze_entity(entity, "Patient is well.", matcher).status is SupportStatus.OMITTED

    def test_change_without_a_stated_status_is_partial(self, matcher):
        """A new or stopped drug named with no framing loses the change."""
        entity = MedicationEntry(
            entity_id="M-1", drug_name="Lisinopril", status=MedicationStatus.DISCONTINUED,
            dose="20 mg",
        )
        result = analyze_entity(entity, "Lisinopril 20 mg.", matcher)
        assert result.status is SupportStatus.PARTIAL

    def test_changed_dose_reported_as_current_is_acceptable(self, matcher):
        """A dose-changed drug is current; the dose check catches a wrong dose."""
        entity = MedicationEntry(
            entity_id="M-1", drug_name="Furosemide", status=MedicationStatus.CHANGED_DOSE,
            dose="40 mg", prior_dose="20 mg",
        )
        result = analyze_entity(entity, "Continued on Furosemide 40 mg.", matcher)
        assert result.status is SupportStatus.SUPPORTED

    def test_discontinued_reported_as_current_is_never_acceptable(self, matcher):
        entity = MedicationEntry(
            entity_id="M-1", drug_name="Amlodipine", status=MedicationStatus.DISCONTINUED,
            dose="5 mg",
        )
        result = analyze_entity(entity, "Continued on Amlodipine 5 mg.", matcher)
        assert result.status is SupportStatus.CONTRADICTED


class TestValueSupport:
    def test_correct_lab_value_is_supported(self, matcher):
        entity = LabResultEntry(entity_id="L-1", name="Creatinine", value=1.9, unit="mg/dL")
        assert analyze_entity(entity, "Creatinine 1.9 mg/dL.", matcher).status is SupportStatus.SUPPORTED

    def test_analyte_without_a_value_is_partial(self, matcher):
        entity = LabResultEntry(entity_id="L-1", name="Creatinine", value=1.9, unit="mg/dL")
        assert analyze_entity(entity, "Creatinine was checked.", matcher).status is SupportStatus.PARTIAL

    def test_approximate_value_is_supported(self, matcher):
        entity = LabResultEntry(entity_id="L-1", name="Creatinine", value=1.90, unit="mg/dL")
        assert analyze_entity(entity, "Creatinine 1.92 mg/dL.", matcher).status is SupportStatus.SUPPORTED

    def test_wrong_single_value_is_contradicted(self, matcher):
        entity = LabResultEntry(entity_id="L-1", name="Creatinine", value=1.9, unit="mg/dL")
        assert analyze_entity(entity, "Creatinine 0.7 mg/dL.", matcher).status is SupportStatus.CONTRADICTED

    def test_blood_pressure_pair_must_match(self, matcher):
        entity = VitalSignEntry(entity_id="V-1", name="Blood pressure", value="168/94", unit="mmHg")
        assert analyze_entity(entity, "BP 168/94 mmHg.", matcher).status is SupportStatus.SUPPORTED
        assert analyze_entity(entity, "BP 168/72 mmHg.", matcher).status is SupportStatus.CONTRADICTED

    def test_non_numeric_vital_checks_presence_only(self, matcher):
        entity = VitalSignEntry(entity_id="V-1", name="Temperature", value="afebrile", unit=None)
        assert analyze_entity(entity, "Temperature afebrile.", matcher).status is SupportStatus.SUPPORTED


class TestOtherEntities:
    def test_allergy_present_is_supported(self, matcher):
        entity = AllergyEntry(
            entity_id="A-1", substance="Penicillin", reaction="rash",
            severity=AllergySeverity.SEVERE,
        )
        assert analyze_entity(entity, "Severe allergy: Penicillin (rash).", matcher).status is SupportStatus.SUPPORTED

    def test_allergy_negated_is_contradicted(self, matcher):
        entity = AllergyEntry(entity_id="A-1", substance="Penicillin", reaction="rash")
        assert analyze_entity(entity, "No known allergy to penicillin.", matcher).status is SupportStatus.CONTRADICTED

    def test_performed_procedure_absent_is_omitted(self, matcher):
        entity = ProcedureEntry(entity_id="P-1", name="Echocardiogram", performed=True)
        assert analyze_entity(entity, "Patient is well.", matcher).status is SupportStatus.OMITTED

    def test_unperformed_procedure_absent_is_supported(self, matcher):
        """Omitting a study that was never done loses nothing."""
        entity = ProcedureEntry(entity_id="P-1", name="Echocardiogram", performed=False)
        assert analyze_entity(entity, "Patient is well.", matcher).status is SupportStatus.SUPPORTED

    def test_unperformed_procedure_reported_as_done_is_contradicted(self, matcher):
        entity = ProcedureEntry(entity_id="P-1", name="Echocardiogram", performed=False)
        result = analyze_entity(entity, "Echocardiogram showed normal function.", matcher)
        assert result.status is SupportStatus.CONTRADICTED

    def test_present_finding_negated_is_contradicted(self, matcher):
        entity = ClinicalFindingEntry(
            entity_id="F-1", description="peripheral edema", present=True,
            significance=FindingSignificance.IMPORTANT,
        )
        assert analyze_entity(entity, "No peripheral edema.", matcher).status is SupportStatus.CONTRADICTED

    def test_conflict_later_statement_is_supported(self, matcher):
        entity = ConflictEntry(
            entity_id="CF-1", topic="chest pain",
            earlier_statement="patient denies chest pain", earlier_timing="triage",
            later_statement="patient reports chest pain", later_timing="reassessment",
            resolution="Later supersedes.",
        )
        assert analyze_entity(entity, "Patient reports chest pain.", matcher).status is SupportStatus.SUPPORTED

    def test_conflict_absent_entirely_is_omitted(self, matcher):
        entity = ConflictEntry(
            entity_id="CF-1", topic="chest pain",
            earlier_statement="patient denies chest pain", earlier_timing="triage",
            later_statement="patient reports chest pain", later_timing="reassessment",
            resolution="Later supersedes.",
        )
        assert analyze_entity(entity, "Patient seen today.", matcher).status is SupportStatus.OMITTED


class TestInjectionHarness:
    @pytest.fixture(scope="class")
    @classmethod
    def cases(cls):
        return generate_benchmark(case_count=20, seed=42)

    def test_unknown_injection_raises(self, cases):
        with pytest.raises(KeyError, match="Unknown injection"):
            inject("not_an_injection", cases[0])

    def test_inject_all_covers_every_injection(self, cases):
        assert set(inject_all(cases[0])) == set(INJECTIONS)

    def test_injections_are_deterministic(self, cases):
        for name in INJECTIONS:
            first = inject(name, cases[0])
            second = inject(name, cases[0])
            if first is None:
                assert second is None
            else:
                assert first.corrupted_summary == second.corrupted_summary

    def test_inapplicable_injection_returns_none(self):
        """A case with no structured source cannot be corrupted."""
        from app.core.enums import CaseType, DataProvenance
        from app.models.schemas import BenchmarkCase

        bare = BenchmarkCase(
            case_id="X", case_type=CaseType.NORMAL, benchmark_version="v",
            provenance=DataProvenance.MANUAL_SYNTHETIC, age_group="45-54", sex="female",
            encounter_type="Visit", clinical_note="n", reference_summary="s",
        )
        for name in INJECTIONS:
            assert inject(name, bare) is None

    def test_injections_declare_their_expected_finding(self, cases):
        for case in cases:
            for injection in inject_all(case).values():
                if injection is not None:
                    assert injection.expected_finding in injection.acceptable_findings
                    assert injection.description.strip()

    def test_acceptable_findings_includes_the_primary_expectation(self):
        from app.core.enums import FindingType
        from app.evaluation.injection import Injection

        injection = Injection(
            name="x", corrupted_summary="s",
            expected_finding=FindingType.CONTRADICTION, description="d",
        )
        assert injection.acceptable_findings == frozenset({FindingType.CONTRADICTION})
