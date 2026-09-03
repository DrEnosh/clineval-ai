"""Tests for the synthetic benchmark generation pipeline.

Covers the two properties the benchmark's trustworthiness rests on:
reproducibility (same inputs, same bytes) and the ground-truth-first data flow
(annotations derive from the structured case, never from rendered text).
"""

from __future__ import annotations

import json

import pytest

from app.benchmark.synthesizer import (
    _dominant_case_type,
    build_metadata,
    build_structured_case,
    case_seed,
    generate_benchmark,
    load_generated_benchmark,
    save_generated_benchmark,
    select_categories,
    to_benchmark_case,
)
from app.core.benchmark_config import BenchmarkConfig, load_benchmark_config
from app.core.enums import (
    CaseType,
    DataProvenance,
    DifficultyTier,
    FailureModeCategory,
    MedicationStatus,
)


@pytest.fixture(scope="module")
def config() -> BenchmarkConfig:
    return load_benchmark_config()


@pytest.fixture(scope="module")
def generated(config: BenchmarkConfig):
    return generate_benchmark(case_count=40, seed=42, config=config)


class TestDeterminism:
    def test_same_seed_and_count_reproduce_identical_cases(self, config):
        first = generate_benchmark(case_count=12, seed=7, config=config)
        second = generate_benchmark(case_count=12, seed=7, config=config)
        assert [c.model_dump(mode="json") for c in first] == [
            c.model_dump(mode="json") for c in second
        ]

    def test_different_seeds_produce_different_cases(self, config):
        first = generate_benchmark(case_count=12, seed=7, config=config)
        second = generate_benchmark(case_count=12, seed=8, config=config)
        assert [c.clinical_note for c in first] != [c.clinical_note for c in second]

    def test_structured_case_is_deterministic_for_a_seed(self, config):
        first = build_structured_case("X", "seed:1", DifficultyTier.HARD, config)
        second = build_structured_case("X", "seed:1", DifficultyTier.HARD, config)
        assert first.model_dump(mode="json") == second.model_dump(mode="json")

    def test_case_seed_is_stable_and_index_specific(self):
        assert case_seed(42, 3) == "42:3"
        assert case_seed(42, 3) != case_seed(42, 4)

    def test_seed_is_recorded_for_reproduction(self, generated):
        for case in generated:
            assert case.generation_metadata["seed"]
            assert case.generation_metadata["generator_version"]


class TestGenerationShape:
    def test_generates_requested_number_of_cases(self, config):
        assert len(generate_benchmark(case_count=17, seed=1, config=config)) == 17

    def test_rejects_non_positive_case_count(self, config):
        with pytest.raises(ValueError, match="at least 1"):
            generate_benchmark(case_count=0, seed=1, config=config)

    def test_case_ids_are_unique_and_ordered(self, generated):
        ids = [c.case_id for c in generated]
        assert len(ids) == len(set(ids))
        assert ids[0] == "SYN_0001"

    def test_difficulty_mix_matches_configuration_exactly(self, config):
        cases = generate_benchmark(case_count=100, seed=42, config=config)
        counts: dict[str, int] = {}
        for case in cases:
            counts[case.difficulty.value] = counts.get(case.difficulty.value, 0) + 1
        assert counts["EASY"] == 20
        assert counts["MODERATE"] == 35
        assert counts["HARD"] == 30
        assert counts["ADVERSARIAL"] == 15

    def test_forced_difficulty_applies_to_every_case(self, config):
        cases = generate_benchmark(
            case_count=8, seed=3, config=config, difficulty=DifficultyTier.ADVERSARIAL
        )
        assert all(c.difficulty is DifficultyTier.ADVERSARIAL for c in cases)

    def test_category_restriction_limits_the_draw_pool(self, config):
        allowed = {FailureModeCategory.NEGATION, FailureModeCategory.ABNORMAL_LABS}
        cases = generate_benchmark(
            case_count=10, seed=5, config=config,
            difficulty=DifficultyTier.MODERATE, categories=list(allowed),
        )
        for case in cases:
            assert set(case.categories) <= allowed

    def test_every_case_declares_synthetic_provenance(self, generated):
        assert all(c.provenance is DataProvenance.SYNTHETIC_TEMPLATE for c in generated)

    def test_every_case_carries_a_structured_source(self, generated):
        for case in generated:
            assert case.structured_case is not None
            assert case.structured_case.case_id == case.case_id

    def test_every_case_has_note_and_shorter_summary(self, generated):
        for case in generated:
            assert case.clinical_note.strip()
            assert case.reference_summary.strip()
            assert len(case.reference_summary) < len(case.clinical_note)

    def test_summary_is_not_a_copy_of_the_note(self, generated):
        """The summary must be an independent rendering, not an excerpt."""
        for case in generated:
            assert case.reference_summary not in case.clinical_note

    def test_subgroup_dimensions_are_populated(self, generated):
        for case in generated:
            assert case.subgroup.keys() >= {"age_group", "sex", "encounter_setting", "difficulty"}


class TestCategoryCoverage:
    def test_a_full_run_exercises_every_category(self, config):
        cases = generate_benchmark(case_count=100, seed=42, config=config)
        seen = {category for case in cases for category in case.categories}
        missing = set(FailureModeCategory) - seen
        assert not missing, f"categories never generated: {sorted(c.value for c in missing)}"

    def test_selected_categories_follow_the_tier_configuration(self, config):
        from random import Random

        tier = config.tier_for(DifficultyTier.HARD)
        categories = select_categories(DifficultyTier.HARD, config, Random("t"))
        assert set(tier.guaranteed) <= set(categories)
        assert set(categories) <= set(tier.pool) | set(tier.guaranteed)

    def test_easy_cases_carry_fewer_categories_than_hard_cases(self, config):
        easy = generate_benchmark(case_count=20, seed=11, config=config, difficulty=DifficultyTier.EASY)
        hard = generate_benchmark(case_count=20, seed=11, config=config, difficulty=DifficultyTier.HARD)
        mean_easy = sum(len(c.categories) for c in easy) / len(easy)
        mean_hard = sum(len(c.categories) for c in hard) / len(hard)
        assert mean_easy < mean_hard

    def test_hard_cases_carry_more_facts_than_easy_cases(self, config):
        def fact_count(case):
            return len(case.critical_facts) + len(case.non_critical_facts) + len(case.medication_facts)

        easy = generate_benchmark(case_count=20, seed=2, config=config, difficulty=DifficultyTier.EASY)
        hard = generate_benchmark(case_count=20, seed=2, config=config, difficulty=DifficultyTier.HARD)
        assert sum(map(fact_count, easy)) / 20 < sum(map(fact_count, hard)) / 20

    def test_dominant_case_type_prefers_the_most_consequential_category(self):
        assert _dominant_case_type([FailureModeCategory.NEGATION]) is CaseType.NEGATION
        assert (
            _dominant_case_type(
                [FailureModeCategory.NEGATION, FailureModeCategory.CONFLICTING_INFORMATION]
            )
            is CaseType.CONFLICTING_INFORMATION
        )
        assert _dominant_case_type([FailureModeCategory.SHORT_NOTE]) is CaseType.NORMAL
        assert _dominant_case_type([]) is CaseType.NORMAL


class TestGroundTruthFirst:
    """The ground-truth-first principle, asserted rather than assumed."""

    def test_annotations_trace_to_structured_entities(self, generated):
        for case in generated:
            index = case.structured_case.entity_index()
            facts = (
                case.critical_facts + case.non_critical_facts + case.medication_facts
                + case.temporal_facts + case.negations
            )
            for fact in facts:
                assert fact.source_ref in index, f"{case.case_id}: {fact.fact_id}"

    def test_medication_annotations_match_structured_medications(self, generated):
        for case in generated:
            structured = {m.entity_id: m for m in case.structured_case.medications}
            for fact in case.medication_facts:
                entity = structured[fact.source_ref]
                assert fact.drug_name == entity.drug_name
                assert fact.status is entity.status
                assert fact.dose == entity.dose

    def test_temporal_annotations_reference_real_lab_results(self, generated):
        for case in generated:
            labs = {lab.entity_id for lab in case.structured_case.labs}
            trends = {t.entity_id: t for t in case.structured_case.lab_trends}
            for fact in case.temporal_facts:
                trend = trends[fact.source_ref]
                assert trend.earlier_ref in labs
                assert trend.later_ref in labs

    def test_negation_annotations_reference_genuinely_negated_content(self, generated):
        for case in generated:
            findings = {f.entity_id: f for f in case.structured_case.findings}
            procedures = {p.entity_id: p for p in case.structured_case.procedures}
            for fact in case.negations:
                if fact.source_ref in findings:
                    assert findings[fact.source_ref].present is False
                else:
                    assert procedures[fact.source_ref].performed is False

    def test_critical_facts_all_carry_a_criticality_rule(self, generated):
        for case in generated:
            for fact in case.critical_facts:
                assert fact.criticality_rule, f"{case.case_id}: {fact.fact_id}"

    def test_note_and_annotations_agree_on_medications(self, generated):
        """Both derive from the same structured case, so they cannot disagree."""
        for case in generated:
            note = case.clinical_note.lower()
            for fact in case.medication_facts:
                assert fact.drug_name.lower() in note


class TestClinicalCoherence:
    def test_negation_traps_name_conditions_the_patient_does_not_have(self, generated):
        """A case must never both assert and deny the same diagnosis."""
        for case in generated:
            active = {c.name.lower() for c in case.structured_case.active_conditions}
            for negation in case.negations:
                content = negation.content.lower()
                if content.startswith("no history of"):
                    denied = content.replace("no history of", "").strip()
                    assert denied not in active, f"{case.case_id}: asserts and denies {denied}"

    def test_no_conflicting_values_for_one_analyte_at_one_timepoint(self, generated):
        for case in generated:
            seen: dict[tuple[str, int], float] = {}
            for lab in case.structured_case.labs:
                key = (lab.name, lab.day_offset)
                if key in seen:
                    assert seen[key] == lab.value, f"{case.case_id}: conflicting {lab.name}"
                seen[key] = lab.value

    def test_dose_changes_record_a_different_prior_dose(self, generated):
        for case in generated:
            for med in case.structured_case.medications:
                if med.status is MedicationStatus.CHANGED_DOSE and med.prior_dose:
                    assert med.prior_dose != med.dose

    def test_timeline_is_ordered_by_time(self, generated):
        for case in generated:
            offsets = [event.day_offset for event in case.structured_case.timeline]
            assert offsets == sorted(offsets)

    def test_discontinued_medications_are_labelled_in_the_note(self, generated):
        for case in generated:
            discontinued = [
                m for m in case.structured_case.medications
                if m.status is MedicationStatus.DISCONTINUED
            ]
            for med in discontinued:
                assert "DISCONTINUED" in case.clinical_note


class TestFailureOpportunities:
    def test_adversarial_cases_document_failure_opportunities(self, config):
        cases = generate_benchmark(
            case_count=10, seed=9, config=config, difficulty=DifficultyTier.ADVERSARIAL
        )
        for case in cases:
            assert case.expected_failure_opportunities, case.case_id

    def test_opportunities_reference_real_entities_when_targeted(self, generated):
        for case in generated:
            index = case.structured_case.entity_index()
            for opportunity in case.expected_failure_opportunities:
                if opportunity.target_ref is not None:
                    assert opportunity.target_ref in index

    def test_risk_flags_only_promote_serious_opportunities(self, generated):
        for case in generated:
            for flag in case.expected_risk_flags:
                assert flag.severity.value in {"HIGH", "CRITICAL"}


class TestMetadata:
    def test_metadata_counts_match_the_cases(self, generated, config):
        metadata = build_metadata(generated, 42, config)
        assert metadata["case_count"] == len(generated)
        assert metadata["counts"]["critical_facts"] == sum(len(c.critical_facts) for c in generated)
        assert metadata["counts"]["temporal_facts"] == sum(len(c.temporal_facts) for c in generated)
        assert metadata["counts"]["negation_facts"] == sum(len(c.negations) for c in generated)

    def test_metadata_answers_the_required_questions(self, generated, config):
        metadata = build_metadata(generated, 42, config)
        containing = metadata["cases_containing"]
        assert containing.keys() >= {
            "negation", "medication_change", "temporal_trend", "critical_facts",
            "deliberate_failure_opportunity", "high_complexity",
        }

    def test_metadata_records_provenance_and_seed(self, generated, config):
        metadata = build_metadata(generated, 42, config)
        assert metadata["seed"] == 42
        assert metadata["contains_real_patient_data"] is False
        assert metadata["provenance"] == DataProvenance.SYNTHETIC_TEMPLATE.value

    def test_metadata_is_empty_safe(self, config):
        metadata = build_metadata([], 1, config)
        assert metadata["case_count"] == 0
        assert metadata["averages"]["note_characters"] == 0.0


class TestPersistence:
    def test_saves_cases_and_metadata(self, generated, config, tmp_path):
        metadata = build_metadata(generated, 42, config)
        cases_path, metadata_path = save_generated_benchmark(generated, metadata, tmp_path)
        assert cases_path.exists() and metadata_path.exists()
        assert config.version in str(cases_path)

    def test_round_trips_through_json(self, generated, config, tmp_path):
        metadata = build_metadata(generated, 42, config)
        cases_path, _ = save_generated_benchmark(generated, metadata, tmp_path)
        assert load_generated_benchmark(cases_path) == generated

    def test_refuses_to_silently_overwrite_a_version(self, generated, config, tmp_path):
        metadata = build_metadata(generated, 42, config)
        save_generated_benchmark(generated, metadata, tmp_path)
        with pytest.raises(FileExistsError, match="force"):
            save_generated_benchmark(generated, metadata, tmp_path)

    def test_force_allows_overwrite(self, generated, config, tmp_path):
        metadata = build_metadata(generated, 42, config)
        save_generated_benchmark(generated, metadata, tmp_path)
        cases_path, _ = save_generated_benchmark(generated, metadata, tmp_path, force=True)
        assert cases_path.exists()

    def test_saved_metadata_is_valid_json(self, generated, config, tmp_path):
        metadata = build_metadata(generated, 42, config)
        _, metadata_path = save_generated_benchmark(generated, metadata, tmp_path)
        assert json.loads(metadata_path.read_text(encoding="utf-8"))["case_count"] == len(generated)

    def test_missing_file_raises_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="generate_benchmark"):
            load_generated_benchmark(tmp_path / "nope.json")


class TestRendererIndependence:
    def test_to_benchmark_case_is_pure(self, config):
        """Rendering twice from one structured case yields identical output."""
        structured = build_structured_case("SYN_TEST", "seed:0", DifficultyTier.HARD, config)
        first = to_benchmark_case(structured, "test-v1")
        second = to_benchmark_case(structured, "test-v1")
        assert first.model_dump(mode="json") == second.model_dump(mode="json")

    def test_rendering_does_not_mutate_the_structured_case(self, config):
        structured = build_structured_case("SYN_TEST", "seed:0", DifficultyTier.HARD, config)
        before = structured.model_dump(mode="json")
        to_benchmark_case(structured, "test-v1")
        assert structured.model_dump(mode="json") == before
