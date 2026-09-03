"""Tests for benchmark integrity checking.

Two halves. The first runs every check against a real generated benchmark and
asserts it is clean. The second deliberately corrupts a case and asserts the
relevant check *catches* it - a validator that never fails is worth nothing,
so each check is proven to fire.
"""

from __future__ import annotations

import pytest

from app.benchmark import generate_benchmark
from app.benchmark.integrity import (
    check_annotations_reference_source,
    check_case_ids_unique,
    check_case_structure,
    check_critical_facts_grounded,
    check_medication_annotations,
    check_metadata_matches,
    check_negation_annotations,
    check_no_duplicate_measurements,
    check_no_identifiable_information,
    check_summary_supported,
    check_temporal_annotations,
    validate_benchmark,
    validate_case,
)
from app.benchmark.synthesizer import build_metadata
from app.core.benchmark_config import load_benchmark_config
from app.core.enums import (
    DifficultyTier,
    FactImportance,
    FactPolarity,
    MedicationStatus,
)


@pytest.fixture(scope="module")
def config():
    return load_benchmark_config()


@pytest.fixture(scope="module")
def cases(config):
    return generate_benchmark(case_count=60, seed=42, config=config)


@pytest.fixture
def case(config):
    """A single mutable adversarial case for corruption tests."""
    return generate_benchmark(
        case_count=1, seed=99, config=config, difficulty=DifficultyTier.ADVERSARIAL
    )[0]


class TestCleanBenchmarkPasses:
    def test_full_benchmark_has_no_integrity_issues(self, cases):
        issues = validate_benchmark(cases)
        assert issues == [], "\n".join(str(i) for i in issues[:10])

    def test_generated_100_case_benchmark_is_clean(self, config):
        issues = validate_benchmark(generate_benchmark(case_count=100, seed=42, config=config))
        assert issues == [], "\n".join(str(i) for i in issues[:10])

    def test_metadata_matches_its_cases(self, cases, config):
        metadata = build_metadata(cases, 42, config)
        assert check_metadata_matches(cases, metadata) == []

    def test_case_ids_are_unique(self, cases):
        assert check_case_ids_unique(cases) == []

    def test_no_identifiable_information_anywhere(self, cases):
        for case in cases:
            assert check_no_identifiable_information(case) == []

    def test_every_summary_is_fully_supported(self, cases):
        for case in cases:
            assert check_summary_supported(case) == []


class TestChecksActuallyFire:
    """Each check is proven to catch the defect it exists for."""

    def test_catches_annotation_with_dangling_source_ref(self, case):
        case.critical_facts[0].source_ref = "NOPE-999"
        issues = check_annotations_reference_source(case)
        assert any("NOPE-999" in i.detail for i in issues)

    def test_catches_annotation_with_no_source_ref(self, case):
        case.critical_facts[0].source_ref = None
        assert any("no source_ref" in i.detail for i in check_annotations_reference_source(case))

    def test_catches_critical_fact_without_criticality_rule(self, case):
        case.critical_facts[0].criticality_rule = None
        assert any("criticality rule" in i.detail for i in check_critical_facts_grounded(case))

    def test_catches_mislabelled_importance_in_critical_bucket(self, case):
        case.critical_facts[0].importance = FactImportance.NON_CRITICAL
        assert any("marked non_critical" in i.detail for i in check_critical_facts_grounded(case))

    def test_catches_medication_annotation_naming_the_wrong_drug(self, case):
        case.medication_facts[0].drug_name = "Nonexistatin"
        issues = check_medication_annotations(case)
        assert any("Nonexistatin" in i.detail for i in issues)

    def test_catches_medication_status_disagreeing_with_source(self, case):
        fact = next(
            f for f in case.medication_facts if f.status is not MedicationStatus.HISTORICAL
        )
        fact.status = MedicationStatus.HISTORICAL
        assert any("disagrees" in i.detail for i in check_medication_annotations(case))

    def test_catches_temporal_fact_with_identical_endpoints(self, case):
        if not case.temporal_facts:
            pytest.skip("case has no temporal facts")
        case.temporal_facts[0].later_value = case.temporal_facts[0].earlier_value
        assert any("identical" in i.detail for i in check_temporal_annotations(case))

    def test_catches_temporal_fact_referencing_a_missing_trend(self, case):
        if not case.temporal_facts:
            pytest.skip("case has no temporal facts")
        case.temporal_facts[0].source_ref = "TRND-999"
        assert any("trend entity" in i.detail for i in check_temporal_annotations(case))

    def test_catches_negation_not_marked_negated(self, case):
        case.negations[0].polarity = FactPolarity.AFFIRMED
        assert any("not marked NEGATED" in i.detail for i in check_negation_annotations(case))

    def test_catches_negation_pointing_at_a_present_finding(self, case):
        negation = case.negations[0]
        finding = next(
            f for f in case.structured_case.findings if f.entity_id == negation.source_ref
        )
        finding.present = True
        assert any("present=True" in i.detail for i in check_negation_annotations(case))

    def test_catches_summary_naming_a_drug_not_in_the_case(self, case):
        case.reference_summary += " Patient was also started on Warfarin 5 mg daily."
        issues = check_summary_supported(case)
        assert any("Warfarin" in i.detail for i in issues)

    def test_catches_summary_inventing_a_numeric_value(self, case):
        case.reference_summary += " Creatinine was 8.77 mg/dL."
        assert any("8.77" in i.detail for i in check_summary_supported(case))

    def test_catches_summary_naming_a_condition_not_in_the_case(self, case):
        present = case.structured_case.text_corpus()
        invented = next(
            name for name in ["Seizure disorder", "Cellulitis", "Hypothyroidism"]
            if name.lower() not in present
        )
        case.reference_summary += f" Patient also has {invented.lower()}."
        assert any(invented in i.detail for i in check_summary_supported(case))

    def test_catches_duplicate_case_ids(self, cases):
        duplicated = [cases[0], cases[0]]
        assert check_case_ids_unique(duplicated)

    def test_catches_conflicting_values_at_one_timepoint(self, case):
        labs = case.structured_case.labs
        clone = labs[0].model_copy(update={"entity_id": "LAB-999", "value": labs[0].value + 5})
        case.structured_case.labs.append(clone)
        assert any("conflicting" in i.detail for i in check_no_duplicate_measurements(case))

    def test_catches_an_email_address(self, case):
        case.clinical_note += "\nContact: patient@example.com"
        assert any("email" in i.detail for i in check_no_identifiable_information(case))

    def test_catches_a_phone_number(self, case):
        case.clinical_note += "\nCall 555-867-5309 for results."
        assert any("phone" in i.detail for i in check_no_identifiable_information(case))

    def test_catches_a_calendar_date(self, case):
        case.clinical_note += "\nSeen on 03/14/2024."
        assert any("calendar_date" in i.detail for i in check_no_identifiable_information(case))

    def test_catches_a_named_person(self, case):
        case.clinical_note += "\nReviewed with Dr. Almeida."
        assert any("named_person" in i.detail for i in check_no_identifiable_information(case))

    def test_catches_a_non_synthetic_patient_id(self, case):
        case.structured_case.patient.patient_id = "12345678"
        assert any("synthetic surrogate" in i.detail for i in check_no_identifiable_information(case))

    def test_catches_an_empty_note(self, case):
        case.clinical_note = "   "
        assert any("empty clinical note" in i.detail for i in check_case_structure(case))

    def test_catches_an_empty_summary(self, case):
        case.reference_summary = ""
        assert any("empty reference summary" in i.detail for i in check_case_structure(case))

    def test_catches_a_summary_longer_than_its_note(self, case):
        case.reference_summary = case.clinical_note + " extra"
        assert any("not shorter" in i.detail for i in check_case_structure(case))

    def test_catches_metadata_that_drifted_from_its_cases(self, cases, config):
        metadata = build_metadata(cases, 42, config)
        metadata["counts"]["critical_facts"] += 1
        assert any("counts" in i.detail for i in check_metadata_matches(cases, metadata))

    def test_catches_metadata_with_the_wrong_case_count(self, cases, config):
        metadata = build_metadata(cases, 42, config)
        metadata["case_count"] = 9999
        assert any("case_count" in i.detail for i in check_metadata_matches(cases, metadata))


class TestValidatorBehaviour:
    def test_validate_case_aggregates_every_check(self, case):
        case.critical_facts[0].source_ref = "NOPE"
        case.clinical_note += "\nContact: x@y.com"
        checks = {issue.check for issue in validate_case(case)}
        assert "annotations_reference_source" in checks
        assert "no_identifiable_information" in checks

    def test_checks_are_skipped_for_cases_without_a_structured_source(self):
        """The curated Phase 1 cases predate the structured pipeline; the
        structural checks must tolerate them rather than reporting false
        positives."""
        from app.benchmark import generate_default_benchmark

        for curated in generate_default_benchmark():
            assert check_annotations_reference_source(curated) == []
            assert check_medication_annotations(curated) == []
            assert check_summary_supported(curated) == []

    def test_curated_cases_pass_the_privacy_and_structure_checks(self):
        from app.benchmark import generate_default_benchmark

        for curated in generate_default_benchmark():
            assert check_no_identifiable_information(curated) == []
            assert check_case_structure(curated) == []
