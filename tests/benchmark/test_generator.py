"""Tests for the synthetic benchmark.

Beyond "does it generate", these assert the gold-standard annotation layer is
internally coherent: fact ids are unique, importance is used correctly,
negated facts really are negated in the note, and each stress-test case type
actually contains the annotation it is meant to exercise. A benchmark whose
ground truth is wrong would silently produce wrong evaluations, so these
checks matter more than they look.
"""

from __future__ import annotations

import json

import pytest

from app.benchmark import (
    BENCHMARK_VERSION,
    generate_default_benchmark,
    load_benchmark,
    save_benchmark,
)
from app.core.enums import (
    CaseType,
    DataProvenance,
    FactImportance,
    FactPolarity,
    MedicationStatus,
    TemporalTrend,
)


class TestBenchmarkGeneration:
    def test_generates_ten_cases(self, benchmark_cases):
        assert len(benchmark_cases) == 10

    def test_covers_every_case_type_exactly_once(self, benchmark_cases):
        types = [case.case_type for case in benchmark_cases]
        assert sorted(t.value for t in types) == sorted(t.value for t in CaseType)
        assert len(types) == len(set(types))

    def test_case_ids_are_unique(self, benchmark_cases):
        ids = [case.case_id for case in benchmark_cases]
        assert len(ids) == len(set(ids))

    def test_generation_is_deterministic(self):
        first = generate_default_benchmark()
        second = generate_default_benchmark()
        assert [c.model_dump(mode="json") for c in first] == [
            c.model_dump(mode="json") for c in second
        ]

    def test_all_cases_declare_synthetic_provenance(self, benchmark_cases):
        assert all(c.provenance is DataProvenance.MANUAL_SYNTHETIC for c in benchmark_cases)

    def test_all_cases_carry_benchmark_version(self, benchmark_cases):
        assert all(c.benchmark_version == BENCHMARK_VERSION for c in benchmark_cases)

    def test_every_case_has_note_and_reference_summary(self, benchmark_cases):
        for case in benchmark_cases:
            assert case.clinical_note.strip(), case.case_id
            assert case.reference_summary.strip(), case.case_id

    def test_reference_summary_is_shorter_than_note(self, benchmark_cases):
        for case in benchmark_cases:
            assert len(case.reference_summary) < len(case.clinical_note), case.case_id

    def test_every_case_has_subgroup_dimensions(self, benchmark_cases):
        for case in benchmark_cases:
            assert case.subgroup.keys() >= {"age_group", "sex", "encounter_setting"}, case.case_id


class TestGoldStandardIntegrity:
    def test_fact_ids_are_unique_within_a_case(self, benchmark_cases):
        for case in benchmark_cases:
            all_facts = (
                case.critical_facts + case.non_critical_facts + case.medication_facts
                + case.temporal_facts + case.negations
            )
            ids = [f.fact_id for f in all_facts]
            assert len(ids) == len(set(ids)), f"{case.case_id} has duplicate fact ids: {ids}"

    def test_fact_ids_are_namespaced_by_case(self, benchmark_cases):
        for case in benchmark_cases:
            all_facts = (
                case.critical_facts + case.non_critical_facts + case.medication_facts
                + case.temporal_facts + case.negations
            )
            for fact in all_facts:
                assert fact.fact_id.startswith(case.case_id), fact.fact_id

    def test_critical_facts_are_marked_critical(self, benchmark_cases):
        for case in benchmark_cases:
            for fact in case.critical_facts:
                assert fact.importance is FactImportance.CRITICAL, fact.fact_id

    def test_non_critical_facts_are_marked_non_critical(self, benchmark_cases):
        for case in benchmark_cases:
            for fact in case.non_critical_facts:
                assert fact.importance is FactImportance.NON_CRITICAL, fact.fact_id

    def test_negations_are_all_negated_polarity(self, benchmark_cases):
        for case in benchmark_cases:
            for fact in case.negations:
                assert fact.polarity is FactPolarity.NEGATED, fact.fact_id

    def test_every_fact_has_non_empty_content(self, benchmark_cases):
        for case in benchmark_cases:
            all_facts = (
                case.critical_facts + case.non_critical_facts + case.medication_facts
                + case.temporal_facts + case.negations
            )
            for fact in all_facts:
                assert fact.content.strip(), fact.fact_id

    def test_temporal_facts_have_distinct_earlier_and_later_values(self, benchmark_cases):
        for case in benchmark_cases:
            for fact in case.temporal_facts:
                assert fact.earlier_value, fact.fact_id
                assert fact.later_value, fact.fact_id
                assert fact.earlier_value != fact.later_value, fact.fact_id
                assert fact.trend is not TemporalTrend.UNKNOWN, fact.fact_id

    def test_medication_facts_name_a_drug(self, benchmark_cases):
        for case in benchmark_cases:
            for med in case.medication_facts:
                assert med.drug_name.strip(), med.fact_id
                assert isinstance(med.status, MedicationStatus)

    def test_risk_flag_ids_are_unique_within_a_case(self, benchmark_cases):
        for case in benchmark_cases:
            ids = [f.flag_id for f in case.expected_risk_flags]
            assert len(ids) == len(set(ids)), case.case_id


class TestCaseTypeCoverage:
    """Each stress-test case must actually contain the annotation it exists to test."""

    def _case(self, cases, case_type):
        return next(c for c in cases if c.case_type is case_type)

    def test_critical_omission_case_has_critical_facts(self, benchmark_cases):
        case = self._case(benchmark_cases, CaseType.CRITICAL_OMISSION)
        assert case.critical_facts
        assert case.expected_risk_flags

    def test_negation_case_has_multiple_critical_negations(self, benchmark_cases):
        case = self._case(benchmark_cases, CaseType.NEGATION)
        critical_negations = [
            f for f in case.negations if f.importance is FactImportance.CRITICAL
        ]
        assert len(critical_negations) >= 2

    def test_temporal_case_has_temporal_facts_with_trends(self, benchmark_cases):
        case = self._case(benchmark_cases, CaseType.TEMPORAL)
        assert case.temporal_facts
        assert all(f.trend is not TemporalTrend.UNKNOWN for f in case.temporal_facts)

    def test_medication_case_covers_all_medication_statuses(self, benchmark_cases):
        case = self._case(benchmark_cases, CaseType.MEDICATION)
        statuses = {med.status for med in case.medication_facts}
        assert statuses == set(MedicationStatus), f"missing: {set(MedicationStatus) - statuses}"

    def test_multi_morbidity_case_has_several_conditions_and_medications(self, benchmark_cases):
        case = self._case(benchmark_cases, CaseType.MULTI_MORBIDITY)
        assert len(case.conditions) >= 3
        assert len(case.medication_facts) >= 3

    def test_high_complexity_case_is_longest_and_richest(self, benchmark_cases):
        case = self._case(benchmark_cases, CaseType.HIGH_COMPLEXITY)
        assert len(case.clinical_note) > 800
        assert len(case.critical_facts) >= 3
        assert len(case.clinical_timeline) >= 3

    def test_conflicting_information_case_links_superseded_facts(self, benchmark_cases):
        case = self._case(benchmark_cases, CaseType.CONFLICTING_INFORMATION)
        superseded = [f for f in case.negations if "superseded_by" in f.attributes]
        assert superseded, "conflicting case must mark which earlier fact is superseded"
        superseding_ids = {f.fact_id for f in case.critical_facts}
        for fact in superseded:
            assert fact.attributes["superseded_by"] in superseding_ids

    def test_hallucination_case_documents_what_must_not_be_invented(self, benchmark_cases):
        case = self._case(benchmark_cases, CaseType.HALLUCINATION)
        assert case.expected_risk_flags
        assert any(f.polarity is FactPolarity.NEGATED for f in case.critical_facts)

    def test_normal_case_has_no_expected_risk_flags(self, benchmark_cases):
        case = self._case(benchmark_cases, CaseType.NORMAL)
        assert case.expected_risk_flags == []


class TestNoteAnnotationConsistency:
    """Spot-checks that annotations are grounded in the note text, since an
    annotation describing something the note never says is unusable ground truth."""

    def test_negated_facts_appear_with_negation_language_in_note(self, benchmark_cases):
        negation_markers = ("no ", "denies", "not ", "without", "avoided", "unremarkable")
        for case in benchmark_cases:
            for fact in case.negations:
                note = case.clinical_note.lower()
                assert any(m in note for m in negation_markers), case.case_id

    def test_medication_drug_names_appear_in_note(self, benchmark_cases):
        for case in benchmark_cases:
            note = case.clinical_note.lower()
            for med in case.medication_facts:
                assert med.drug_name.lower() in note, f"{case.case_id}: {med.drug_name}"

    def test_temporal_values_appear_in_note(self, benchmark_cases):
        for case in benchmark_cases:
            note = case.clinical_note.lower()
            for fact in case.temporal_facts:
                earlier_number = fact.earlier_value.split()[0].lower()
                later_number = fact.later_value.split()[0].lower()
                assert earlier_number in note, f"{case.case_id}: {fact.earlier_value}"
                assert later_number in note, f"{case.case_id}: {fact.later_value}"


class TestBenchmarkPersistence:
    def test_save_then_load_round_trips(self, benchmark_cases, tmp_path):
        path = tmp_path / "nested" / "benchmark.json"
        save_benchmark(benchmark_cases, path)
        assert path.exists()
        assert load_benchmark(path) == benchmark_cases

    def test_saved_file_is_valid_json_array(self, benchmark_cases, tmp_path):
        path = save_benchmark(benchmark_cases, tmp_path / "benchmark.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        assert len(payload) == 10

    def test_missing_file_raises_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="generate_benchmark"):
            load_benchmark(tmp_path / "does_not_exist.json")

    def test_malformed_file_fails_loudly(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('[{"case_id": "X"}]', encoding="utf-8")
        with pytest.raises(Exception):
            load_benchmark(path)
