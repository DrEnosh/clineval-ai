"""Tests for the criticality rules.

Each rule is tested directly against the structured entity it classifies, so
"why is this fact critical?" is verifiable independently of the generator.
"""

from __future__ import annotations

from app.benchmark import criticality
from app.benchmark.content import LAB_TEMPLATES, AllergyTemplate
from app.core.enums import (
    AbnormalFlag,
    AllergySeverity,
    FindingSignificance,
    MedicationStatus,
    Severity,
    TemporalTrend,
)
from app.models.structured import (
    AllergyEntry,
    ClinicalFindingEntry,
    ConflictEntry,
    DiagnosisEntry,
    LabResultEntry,
    LabTrendEntry,
    MedicationEntry,
    VitalSignEntry,
)


def _lab(name: str, value: float) -> LabResultEntry:
    template = LAB_TEMPLATES[name]
    return LabResultEntry(
        entity_id="LAB-001", name=name, value=value, unit=template.unit,
        abnormal_flag=template.classify(value),
        reference_low=template.reference_low, reference_high=template.reference_high,
    )


class TestRuleRegistry:
    def test_every_rule_has_a_description_rationale_and_severity(self):
        for rule_id, rule in criticality.CRITICALITY_RULES.items():
            assert rule.rule_id == rule_id
            assert rule.description.strip()
            assert rule.rationale.strip()
            assert isinstance(rule.severity_if_missed, Severity)

    def test_rules_are_retrievable_by_id(self):
        assert criticality.rule("CR-01").rule_id == "CR-01"

    def test_ten_documented_rules_exist(self):
        assert len(criticality.CRITICALITY_RULES) == 10


class TestLabClassification:
    def test_panic_value_is_cr01(self):
        assert criticality.classify_lab(_lab("Potassium", 6.4)) == "CR-01"

    def test_abnormal_key_analyte_is_cr02(self):
        assert criticality.classify_lab(_lab("Creatinine", 1.8)) == "CR-02"

    def test_abnormal_non_key_analyte_is_not_critical(self):
        assert criticality.classify_lab(_lab("LDL cholesterol", 160)) is None

    def test_normal_value_is_not_critical(self):
        assert criticality.classify_lab(_lab("Creatinine", 0.9)) is None

    def test_panic_takes_precedence_over_merely_abnormal(self):
        """A value both above the reference range and beyond the panic
        threshold must classify as CR-01, not CR-02."""
        potassium = _lab("Potassium", 7.0)
        assert potassium.abnormal_flag is AbnormalFlag.CRITICAL_HIGH
        assert criticality.classify_lab(potassium) == "CR-01"

    def test_low_direction_analyte_classifies_low_values(self):
        assert LAB_TEMPLATES["Hemoglobin"].classify(6.0) is AbnormalFlag.CRITICAL_LOW
        assert LAB_TEMPLATES["Hemoglobin"].classify(10.0) is AbnormalFlag.LOW
        assert LAB_TEMPLATES["Hemoglobin"].classify(13.5) is AbnormalFlag.NORMAL


class TestTrendClassification:
    def _trend(self, significant: bool) -> LabTrendEntry:
        return LabTrendEntry(
            entity_id="TRND-001", lab_name="Creatinine", unit="mg/dL",
            earlier_ref="LAB-001", later_ref="LAB-002",
            earlier_value=1.4, later_value=1.9,
            earlier_timing="3 months ago", later_timing="today",
            trend=TemporalTrend.INCREASING, clinically_significant=significant,
        )

    def test_significant_trend_is_cr03(self):
        assert criticality.classify_lab_trend(self._trend(True)) == "CR-03"

    def test_insignificant_trend_is_not_critical(self):
        assert criticality.classify_lab_trend(self._trend(False)) is None


class TestMedicationClassification:
    def _med(self, status: MedicationStatus) -> MedicationEntry:
        return MedicationEntry(entity_id="MED-001", drug_name="Lisinopril", status=status)

    def test_new_discontinued_and_dose_change_are_cr04(self):
        for status in (
            MedicationStatus.NEW, MedicationStatus.DISCONTINUED, MedicationStatus.CHANGED_DOSE
        ):
            assert criticality.classify_medication(self._med(status)) == "CR-04"

    def test_current_and_historical_are_not_major_changes(self):
        assert criticality.classify_medication(self._med(MedicationStatus.CURRENT)) is None
        assert criticality.classify_medication(self._med(MedicationStatus.HISTORICAL)) is None


class TestOtherRules:
    def test_severe_allergy_is_cr05(self):
        severe = AllergyEntry(
            entity_id="ALG-001", substance="Peanuts", reaction="anaphylaxis",
            severity=AllergySeverity.SEVERE,
        )
        mild = severe.model_copy(update={"severity": AllergySeverity.MILD})
        assert criticality.classify_allergy(severe) == "CR-05"
        assert criticality.classify_allergy(mild) is None

    def test_primary_diagnosis_is_cr06(self):
        primary = DiagnosisEntry(entity_id="DX-001", name="Sepsis", is_primary=True)
        secondary = primary.model_copy(update={"is_primary": False})
        assert criticality.classify_diagnosis(primary) == "CR-06"
        assert criticality.classify_diagnosis(secondary) is None

    def test_every_conflict_is_cr07(self):
        conflict = ConflictEntry(
            entity_id="CONF-001", topic="chest pain",
            earlier_statement="denies chest pain", earlier_timing="triage",
            later_statement="reports chest pain", later_timing="reassessment",
            resolution="later supersedes",
        )
        assert criticality.classify_conflict(conflict) == "CR-07"

    def test_critical_finding_is_cr08_regardless_of_polarity(self):
        negative = ClinicalFindingEntry(
            entity_id="FIND-001", description="no history of diabetes", present=False,
            significance=FindingSignificance.CRITICAL,
        )
        positive = negative.model_copy(update={"present": True})
        routine = negative.model_copy(update={"significance": FindingSignificance.ROUTINE})
        assert criticality.classify_finding(negative) == "CR-08"
        assert criticality.classify_finding(positive) == "CR-08"
        assert criticality.classify_finding(routine) is None

    def test_panic_vital_is_cr09(self):
        critical = VitalSignEntry(
            entity_id="VIT-001", name="Heart rate", value="165", unit="bpm",
            abnormal_flag=AbnormalFlag.CRITICAL_HIGH,
        )
        high = critical.model_copy(update={"abnormal_flag": AbnormalFlag.HIGH})
        normal = critical.model_copy(update={"abnormal_flag": AbnormalFlag.NORMAL})
        assert criticality.classify_vital(critical) == "CR-09"
        assert criticality.classify_vital(high) is None
        assert criticality.classify_vital(normal) is None
