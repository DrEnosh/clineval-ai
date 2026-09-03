"""Tests for the note and reference-summary renderers.

The renderers are what make the ground-truth-first claim real, so these tests
focus on the properties that claim depends on: the summary preserves what the
inclusion rules say it must, drops what they say it must not, and distinguishes
current from historical content.
"""

from __future__ import annotations

from random import Random

import pytest

from app.benchmark.annotator import _sentence_case, annotate
from app.benchmark.note_renderer import render_note
from app.benchmark.summary_renderer import render_reference_summary
from app.core.enums import (
    AbnormalFlag,
    AllergySeverity,
    ConditionStatus,
    FindingSignificance,
    MedicationStatus,
    TemporalTrend,
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
from app.core.enums import DifficultyTier


def _case(**overrides) -> StructuredCase:
    payload = dict(
        case_id="SYN_TEST",
        seed="test:0",
        generator_version="0.2.0",
        patient=PatientProfile(patient_id="SYN-PT-00001", age=62, age_group="55-64", sex="female"),
        encounter=EncounterInfo(
            encounter_id="SYN-ENC-00001",
            encounter_type="Nephrology follow-up",
            setting="outpatient_specialty",
        ),
        difficulty=DifficultyTier.MODERATE,
    )
    payload.update(overrides)
    return StructuredCase(**payload)


class TestNoteRenderer:
    def test_renders_a_non_empty_note_with_headers(self):
        case = _case(diagnoses=[DiagnosisEntry(entity_id="DX-001", name="CKD", is_primary=True)])
        note = render_note(case, Random("r"))
        assert "CLINICAL NOTE" in note
        assert "SUBJECTIVE:" in note
        assert "CKD" in note

    def test_includes_patient_age_and_encounter(self):
        note = render_note(_case(), Random("r"))
        assert "62-year-old female" in note
        assert "nephrology follow-up" in note.lower()

    def test_marks_discontinued_medications_explicitly(self):
        case = _case(
            medications=[
                MedicationEntry(
                    entity_id="MED-001", drug_name="Amlodipine",
                    status=MedicationStatus.DISCONTINUED, dose="5 mg",
                    change_timing="1 month ago", change_reason="discontinued due to edema",
                )
            ]
        )
        note = render_note(case, Random("r"))
        assert "DISCONTINUED" in note
        assert "Amlodipine" in note

    def test_shows_both_doses_for_a_dose_change(self):
        case = _case(
            medications=[
                MedicationEntry(
                    entity_id="MED-001", drug_name="Furosemide",
                    status=MedicationStatus.CHANGED_DOSE, dose="40 mg", prior_dose="20 mg",
                )
            ]
        )
        note = render_note(case, Random("r"))
        assert "40 mg" in note and "20 mg" in note

    def test_flags_abnormal_labs_with_reference_ranges(self):
        case = _case(
            labs=[
                LabResultEntry(
                    entity_id="LAB-001", name="Creatinine", value=1.9, unit="mg/dL",
                    abnormal_flag=AbnormalFlag.HIGH, reference_low=0.6, reference_high=1.2,
                )
            ]
        )
        note = render_note(case, Random("r"))
        assert "HIGH" in note
        assert "ref 0.6-1.2" in note

    def test_renders_pertinent_negatives(self):
        case = _case(
            findings=[
                ClinicalFindingEntry(
                    entity_id="FIND-001", description="no history of diabetes", present=False,
                    significance=FindingSignificance.CRITICAL,
                )
            ]
        )
        assert "no history of diabetes" in render_note(case, Random("r"))

    def test_renders_both_sides_of_a_conflict_and_its_resolution(self):
        case = _case(
            conflicts=[
                ConflictEntry(
                    entity_id="CONF-001", topic="chest pain",
                    earlier_statement="patient denies chest pain", earlier_timing="triage",
                    later_statement="patient reports chest pain", later_timing="reassessment",
                    resolution="The later assessment supersedes the initial history.",
                )
            ]
        )
        note = render_note(case, Random("r"))
        assert "denies chest pain" in note
        assert "reports chest pain" in note
        assert "supersedes" in note

    def test_distinguishes_resolved_conditions_from_active_ones(self):
        case = _case(
            conditions=[
                ConditionEntry(
                    entity_id="COND-001", name="Cellulitis", status=ConditionStatus.RESOLVED,
                    onset_description="resolved 2 years ago",
                )
            ]
        )
        assert "not an active problem" in render_note(case, Random("r"))

    def test_reports_a_study_that_was_not_performed(self):
        case = _case(
            procedures=[
                ProcedureEntry(entity_id="PROC-001", name="Chest X-ray", performed=False)
            ]
        )
        assert "was not performed" in render_note(case, Random("r"))

    def test_rendering_is_deterministic_for_a_given_rng_seed(self):
        case = _case()
        assert render_note(case, Random("same")) == render_note(case, Random("same"))

    def test_preserves_abbreviation_casing_in_findings(self):
        case = _case(
            findings=[
                ClinicalFindingEntry(entity_id="FIND-001", description="no history of GERD", present=True)
            ]
        )
        assert "GERD" in render_note(case, Random("r"))


class TestSummaryRenderer:
    def test_summary_states_the_primary_condition(self):
        case = _case(
            conditions=[
                ConditionEntry(
                    entity_id="COND-001", name="Chronic kidney disease stage 3", is_primary=True
                )
            ]
        )
        summary = render_reference_summary(case)
        assert "chronic kidney disease stage 3" in summary.lower()

    def test_summary_preserves_a_significant_trend_with_both_values(self):
        case = _case(
            lab_trends=[
                LabTrendEntry(
                    entity_id="TRND-001", lab_name="Creatinine", unit="mg/dL",
                    earlier_ref="LAB-001", later_ref="LAB-002",
                    earlier_value=1.4, later_value=1.9,
                    earlier_timing="3 months ago", later_timing="today",
                    trend=TemporalTrend.INCREASING, clinically_significant=True,
                )
            ]
        )
        summary = render_reference_summary(case)
        assert "1.4" in summary and "1.9" in summary
        assert "rose" in summary

    def test_summary_omits_an_insignificant_trend(self):
        case = _case(
            lab_trends=[
                LabTrendEntry(
                    entity_id="TRND-001", lab_name="Calcium", unit="mg/dL",
                    earlier_ref="LAB-001", later_ref="LAB-002",
                    earlier_value=9.1, later_value=9.2,
                    earlier_timing="1 month ago", later_timing="today",
                    trend=TemporalTrend.INCREASING, clinically_significant=False,
                )
            ]
        )
        assert "9.1" not in render_reference_summary(case)

    def test_summary_preserves_medication_changes(self):
        case = _case(
            medications=[
                MedicationEntry(
                    entity_id="MED-001", drug_name="Amlodipine",
                    status=MedicationStatus.DISCONTINUED, change_timing="1 month ago",
                ),
                MedicationEntry(
                    entity_id="MED-002", drug_name="Furosemide",
                    status=MedicationStatus.CHANGED_DOSE, dose="40 mg", prior_dose="20 mg",
                ),
            ]
        )
        summary = render_reference_summary(case)
        assert "Amlodipine was discontinued" in summary
        assert "Furosemide was increased from 20 mg to 40 mg" in summary

    def test_summary_marks_historical_medication_as_not_current(self):
        case = _case(
            medications=[
                MedicationEntry(
                    entity_id="MED-001", drug_name="Warfarin", status=MedicationStatus.HISTORICAL
                )
            ]
        )
        assert "not a current medication" in render_reference_summary(case)

    def test_summary_preserves_severe_allergies(self):
        case = _case(
            allergies=[
                AllergyEntry(
                    entity_id="ALG-001", substance="Peanuts", reaction="anaphylaxis",
                    severity=AllergySeverity.SEVERE,
                )
            ]
        )
        assert "Severe allergy" in render_reference_summary(case)

    def test_summary_preserves_important_negatives(self):
        case = _case(
            findings=[
                ClinicalFindingEntry(
                    entity_id="FIND-001", description="no history of diabetes", present=False,
                    significance=FindingSignificance.CRITICAL,
                )
            ]
        )
        assert "no history of diabetes" in render_reference_summary(case)

    def test_summary_drops_routine_normal_findings(self):
        case = _case(
            findings=[
                ClinicalFindingEntry(
                    entity_id="FIND-001", description="alert and oriented", present=True,
                    significance=FindingSignificance.ROUTINE,
                )
            ]
        )
        assert "alert and oriented" not in render_reference_summary(case)

    def test_summary_drops_normal_labs_without_a_trend(self):
        case = _case(
            labs=[
                LabResultEntry(
                    entity_id="LAB-001", name="Creatinine", value=0.9, unit="mg/dL",
                    abnormal_flag=AbnormalFlag.NORMAL,
                )
            ]
        )
        assert "0.9" not in render_reference_summary(case)

    def test_summary_reports_critical_labs_as_critically_abnormal(self):
        case = _case(
            labs=[
                LabResultEntry(
                    entity_id="LAB-001", name="Potassium", value=6.5, unit="mEq/L",
                    abnormal_flag=AbnormalFlag.CRITICAL_HIGH,
                )
            ]
        )
        assert "critically abnormal" in render_reference_summary(case)

    def test_summary_carries_forward_a_study_not_performed(self):
        case = _case(
            procedures=[ProcedureEntry(entity_id="PROC-001", name="Chest X-ray", performed=False)]
        )
        assert "Not performed at this visit" in render_reference_summary(case)

    def test_summary_states_the_conflict_resolution(self):
        case = _case(
            conflicts=[
                ConflictEntry(
                    entity_id="CONF-001", topic="fever",
                    earlier_statement="denies fever", earlier_timing="triage",
                    later_statement="reports subjective fevers", later_timing="reassessment",
                    resolution="The later history supersedes the triage report.",
                )
            ]
        )
        summary = render_reference_summary(case)
        assert "reports subjective fevers" in summary
        assert "supersedes" in summary

    def test_summary_includes_the_follow_up_plan(self):
        case = _case(
            follow_up=[
                FollowUpPlanEntry(
                    entity_id="FU-001", interval="2 weeks", provider="nephrology",
                    actions=["repeat metabolic panel"],
                )
            ]
        )
        summary = render_reference_summary(case)
        assert "repeat metabolic panel" in summary
        assert "2 weeks" in summary

    def test_summary_is_deterministic(self):
        case = _case()
        assert render_reference_summary(case) == render_reference_summary(case)

    def test_summary_reports_abnormal_vitals_only(self):
        case = _case(
            vitals=[
                VitalSignEntry(entity_id="VIT-001", name="Heart rate", value="72", unit="bpm"),
                VitalSignEntry(
                    entity_id="VIT-002", name="Blood pressure", value="168/98", unit="mmHg",
                    abnormal_flag=AbnormalFlag.HIGH,
                ),
            ]
        )
        summary = render_reference_summary(case)
        assert "168/98" in summary
        assert "72" not in summary


class TestAnnotatorHelpers:
    def test_sentence_case_preserves_internal_capitalisation(self):
        assert _sentence_case("no history of GERD") == "No history of GERD"
        assert _sentence_case("") == ""

    def test_annotator_files_critical_negations_in_both_buckets(self):
        """A reversed critical negative is the worst failure the benchmark
        models, so it must count toward critical retention as well as being
        listed as a negation."""
        case = _case(
            findings=[
                ClinicalFindingEntry(
                    entity_id="FIND-001", description="no history of diabetes", present=False,
                    significance=FindingSignificance.CRITICAL,
                )
            ]
        )
        annotations = annotate(case)
        assert len(annotations.negations) == 1
        assert annotations.negations[0] in annotations.critical_facts

    def test_annotator_does_not_double_count_active_conditions(self):
        """Active conditions are annotated once via their diagnosis entry."""
        case = _case(
            conditions=[
                ConditionEntry(entity_id="COND-001", name="Hypertension", is_primary=True)
            ],
            diagnoses=[DiagnosisEntry(entity_id="DX-001", name="Hypertension", is_primary=True)],
        )
        annotations = annotate(case)
        condition_facts = [
            f for f in annotations.critical_facts + annotations.non_critical_facts
            if "Hypertension" in f.content
        ]
        assert len(condition_facts) == 1

    def test_annotator_records_source_refs_on_every_fact(self):
        case = _case(
            diagnoses=[DiagnosisEntry(entity_id="DX-001", name="Sepsis", is_primary=True)],
            medications=[
                MedicationEntry(entity_id="MED-001", drug_name="Ceftriaxone", status=MedicationStatus.NEW)
            ],
        )
        annotations = annotate(case)
        for fact in annotations.critical_facts + annotations.medication_facts:
            assert fact.source_ref
