"""Edge-case and defensive-path tests for benchmark generation.

Covers the branches that only fire when the content library is exhausted or a
feature is applied to a case that cannot support it - the paths that would
otherwise only be discovered by a crash during a large generation run.
"""

from __future__ import annotations

from random import Random

import pytest

from app.benchmark import content, features
from app.benchmark.case_builder import (
    CaseDraft,
    build_base_case,
    next_dose_up,
    sample_abnormal_value,
    sample_critical_value,
    sample_normal_value,
)
from app.benchmark.content import LAB_TEMPLATES, MEDICATION_TEMPLATES
from app.core.enums import (
    AbnormalFlag,
    ConditionStatus,
    DifficultyTier,
    FailureModeCategory,
    MedicationStatus,
)


@pytest.fixture
def draft() -> CaseDraft:
    rng = Random("edge")
    draft = CaseDraft(case_id="SYN_EDGE", seed="edge", rng=rng, difficulty=DifficultyTier.MODERATE)
    build_base_case(draft, rng)
    return draft


class TestSampling:
    def test_normal_values_land_inside_the_reference_range(self):
        rng = Random("s")
        for template in LAB_TEMPLATES.values():
            value = sample_normal_value(template, rng)
            assert template.classify(value) is AbnormalFlag.NORMAL, template.name

    def test_abnormal_values_are_abnormal_but_not_panic(self):
        rng = Random("s")
        for template in LAB_TEMPLATES.values():
            value = sample_abnormal_value(template, rng)
            flag = template.classify(value)
            assert flag is not AbnormalFlag.NORMAL, f"{template.name} -> {value}"
            assert flag not in {AbnormalFlag.CRITICAL_LOW, AbnormalFlag.CRITICAL_HIGH}, (
                f"{template.name} -> {value} should not be panic-range"
            )

    def test_critical_values_are_panic_range_where_thresholds_exist(self):
        rng = Random("s")
        for template in LAB_TEMPLATES.values():
            value = sample_critical_value(template, rng)
            if value is None:
                assert template.critical_low is None and template.critical_high is None
                continue
            assert template.classify(value) in {
                AbnormalFlag.CRITICAL_LOW, AbnormalFlag.CRITICAL_HIGH
            }, f"{template.name} -> {value}"

    def test_analyte_without_panic_thresholds_returns_no_critical_value(self):
        assert sample_critical_value(LAB_TEMPLATES["Hemoglobin A1c"], Random("s")) is None

    def test_next_dose_up_walks_the_ladder(self):
        template = MEDICATION_TEMPLATES["Metoprolol succinate"]
        assert next_dose_up(template, "12.5 mg") == "25 mg"
        assert next_dose_up(template, template.doses[-1]) is None

    def test_next_dose_up_rejects_a_dose_not_on_the_ladder(self):
        assert next_dose_up(MEDICATION_TEMPLATES["Lisinopril"], "999 mg") is None


class TestDraftMechanics:
    def test_entity_ids_are_sequential_per_prefix(self, draft):
        assert draft.next_id("ZZ") == "ZZ-001"
        assert draft.next_id("ZZ") == "ZZ-002"
        assert draft.next_id("YY") == "YY-001"

    def test_categories_are_not_duplicated(self, draft):
        draft.add_category(FailureModeCategory.NEGATION)
        draft.add_category(FailureModeCategory.NEGATION)
        assert draft.categories.count(FailureModeCategory.NEGATION) == 1

    def test_base_case_establishes_patient_encounter_and_primary_problem(self, draft):
        assert draft.patient is not None
        assert draft.encounter is not None
        assert any(c.is_primary for c in draft.conditions)
        assert draft.vitals

    def test_patient_id_is_a_synthetic_surrogate(self, draft):
        assert draft.patient.patient_id.startswith("SYN-PT-")

    def test_has_drug_and_has_lab_reflect_draft_contents(self, draft):
        template = LAB_TEMPLATES["Calcium"]
        assert not draft.has_lab("Calcium")
        draft.add_lab(template, 9.2)
        assert draft.has_lab("Calcium")


class TestFeatureEdgePaths:
    def test_negation_is_a_no_op_when_every_condition_is_already_used(self, draft):
        for template in content.CONDITION_TEMPLATES:
            if template.name not in draft.condition_names():
                draft.add_condition(template)
        before = len(draft.findings)
        features.apply_negation(draft, Random("n"))
        assert len(draft.findings) == before

    def test_historical_vs_current_is_a_no_op_when_conditions_are_exhausted(self, draft):
        for template in content.CONDITION_TEMPLATES:
            if template.name not in draft.condition_names():
                draft.add_condition(template)
        before = len(draft.conditions)
        features.apply_historical_vs_current(draft, Random("h"))
        assert len(draft.conditions) == before

    def test_dose_change_falls_back_when_the_drug_has_one_dose(self, draft, monkeypatch):
        """Albuterol has a single dose, so a dose *change* is impossible; the
        feature must degrade to a new-medication change rather than crash."""
        monkeypatch.setattr(
            features, "_add_medication_for_case",
            lambda draft, rng: MEDICATION_TEMPLATES["Albuterol"],
        )
        features.apply_medication_dose_change(draft, Random("d"))
        albuterol = [m for m in draft.medications if m.drug_name == "Albuterol"]
        assert albuterol and albuterol[0].status is MedicationStatus.NEW

    def test_duplicate_documentation_survives_a_case_with_no_findings(self, draft):
        draft.findings.clear()
        features.apply_duplicate_documentation(draft, Random("r"))
        assert draft.duplicate_documentation is True

    def test_relevant_negatives_is_a_no_op_when_all_are_already_present(self, draft):
        pool = list(content.GENERIC_NEGATIVES)
        for template in draft.condition_templates:
            pool.extend(template.negatives)
        for description in dict.fromkeys(pool):
            draft.add_finding(description, present=False)
        before = len(draft.findings)
        features.apply_relevant_negative_findings(draft, Random("r"))
        assert len(draft.findings) == before

    def test_high_complexity_sets_a_length_of_stay(self, draft):
        features.apply_high_complexity(draft, Random("c"))
        assert draft.encounter.length_of_stay_days is not None

    def test_high_complexity_produces_a_rise_then_partial_recovery(self, draft):
        features.apply_high_complexity(draft, Random("c"))
        creatinines = [lab for lab in draft.labs if lab.name == "Creatinine"]
        assert len(creatinines) == 3
        baseline, peak, recovery = creatinines
        assert peak.value > baseline.value
        assert baseline.value < recovery.value < peak.value

    def test_combined_critical_applies_three_traps(self, draft):
        features.apply_combined_critical(draft, Random("k"))
        kinds = {o.type.value for o in draft.failure_opportunities}
        assert "negation_trap" in kinds
        assert "temporal_trap" in kinds
        assert "medication_trap" in kinds

    def test_every_feature_builder_runs_without_error(self, draft):
        """Smoke test: no feature may crash on a freshly built base case."""
        for category, builder in features.FEATURE_BUILDERS.items():
            fresh_rng = Random(f"smoke:{category.value}")
            builder(draft, fresh_rng)
        assert draft.build() is not None


class TestStructuredCaseHelpers:
    @pytest.fixture
    def structured(self, draft):
        features.apply_medication_discontinuation(draft, Random("m"))
        features.apply_historical_vs_current(draft, Random("h"))
        features.apply_abnormal_labs(draft, Random("l"))
        features.apply_negation(draft, Random("n"))
        return draft.build()

    def test_active_and_historical_conditions_partition_conditions(self, structured):
        assert len(structured.active_conditions) + len(structured.historical_conditions) == len(
            structured.conditions
        )
        assert all(c.status is ConditionStatus.ACTIVE for c in structured.active_conditions)

    def test_current_medications_exclude_discontinued_and_historical(self, structured):
        statuses = {m.status for m in structured.current_medications}
        assert MedicationStatus.DISCONTINUED not in statuses
        assert MedicationStatus.HISTORICAL not in statuses

    def test_medication_changes_lists_only_changes(self, structured):
        assert all(m.is_change for m in structured.medication_changes)

    def test_abnormal_labs_lists_only_abnormal_results(self, structured):
        assert all(lab.is_abnormal for lab in structured.abnormal_labs)

    def test_relevant_negatives_lists_only_absent_findings(self, structured):
        assert all(not f.present for f in structured.relevant_negatives)

    def test_entity_index_covers_every_entity(self, structured):
        index = structured.entity_index()
        assert len(index) == len(structured.all_entities())

    def test_text_corpus_contains_structured_content(self, structured):
        corpus = structured.text_corpus()
        for med in structured.medications:
            assert med.drug_name.lower() in corpus

    def test_value_formatting_drops_trailing_zeroes(self):
        from app.models.structured import LabResultEntry

        whole = LabResultEntry(entity_id="LAB-1", name="WBC", value=14.0, unit="K/uL")
        fractional = LabResultEntry(entity_id="LAB-2", name="Creatinine", value=1.9, unit="mg/dL")
        assert whole.format_value() == "14"
        assert fractional.format_value() == "1.9"
