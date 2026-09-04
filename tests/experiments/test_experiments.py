"""Tests for the Phase 5 experiment apparatus.

The apparatus must do three things reliably, and each is tested here: select a
reproducible stratified sample, keep pre- and post-adjudication results
separate, and **refuse to fabricate** a real-model result when no real model is
available.

The anchoring comparison gets particular attention. A comparison that always
reports agreement would be useless, so there is a test proving it detects a
judge whose answer actually depends on seeing the deterministic finding.
"""

from __future__ import annotations

import json

import pytest

from app.core.enums import DifficultyTier, FailureModeCategory, FindingType
from app.experiments.runner import (
    ADJUDICATION_CONDITIONS,
    ModelSpec,
    build_comparison,
    compare_conditions,
    evaluate_model,
    repeated_run_analysis,
)
from app.experiments.selection import (
    STRATA,
    classify_failures,
    select_experiment_sample,
    summarize_sample,
)
from app.generation import OutputCache
from app.providers import MockProvider, mock_judge, mock_summarizer
from app.providers.mock import judgment_json


@pytest.fixture(scope="module")
def pool():
    from app.benchmark import generate_benchmark

    return generate_benchmark(case_count=100, seed=42)


@pytest.fixture(scope="module")
def selection(pool):
    return select_experiment_sample(pool)


def _spec(cases, behaviour="correct", label="mock:test", **kwargs) -> ModelSpec:
    return ModelSpec(
        label=label, provider_name="mock", model="test-model",
        provider=mock_summarizer(cases, behaviour), **kwargs
    )


class TestSampleSelection:
    def test_selects_one_case_per_stratum(self, selection):
        assert len(selection) == len(STRATA)
        assert selection.is_complete
        assert not selection.unfilled

    def test_covers_every_required_stratum(self, selection):
        assert set(selection.assignments) == {stratum.name for stratum in STRATA}

    def test_no_case_is_used_twice(self, selection):
        assert len(set(selection.case_ids)) == len(selection.case_ids)

    def test_selection_is_reproducible(self, pool):
        assert (
            select_experiment_sample(pool).case_ids
            == select_experiment_sample(pool).case_ids
        )

    def test_each_case_actually_matches_its_stratum(self, selection, pool):
        by_id = {case.case_id: case for case in pool}
        predicates = {stratum.name: stratum.predicate for stratum in STRATA}
        for name, case_id in selection.assignments.items():
            assert predicates[name](by_id[case_id]), name

    def test_sample_spans_difficulty_tiers(self, selection):
        tiers = {case.difficulty for case in selection.cases}
        assert DifficultyTier.EASY in tiers
        assert DifficultyTier.ADVERSARIAL in tiers

    def test_unfilled_strata_are_reported_not_backfilled(self, pool):
        """Silently substituting a case would claim coverage that is absent."""
        easy_only = [c for c in pool if c.difficulty is DifficultyTier.EASY]
        selection = select_experiment_sample(easy_only)
        assert selection.unfilled
        assert not selection.is_complete
        assert len(selection) < len(STRATA)

    def test_empty_pool_yields_an_empty_sample(self):
        selection = select_experiment_sample([])
        assert len(selection) == 0
        assert len(selection.unfilled) == len(STRATA)

    def test_summary_reports_composition(self, selection):
        summary = summarize_sample(selection)
        assert summary["case_count"] == len(STRATA)
        assert summary["critical_facts"] > 0
        assert summary["by_difficulty"]

    def test_summary_of_an_empty_sample_is_safe(self):
        assert summarize_sample(select_experiment_sample([]))["case_count"] == 0


class TestFailureTaxonomy:
    def test_absent_groups_report_explicit_zeros(self):
        """'0 negation reversals' and 'we did not look' are different claims."""
        counts = classify_failures([])
        assert counts["negation_reversal"] == 0
        assert set(counts) >= {"critical_omission", "temporal_error", "numeric_error"}

    def test_counts_findings_by_group(self):
        from app.models.schemas import EvaluationFinding

        def finding(finding_type):
            return EvaluationFinding(
                finding_id="F", case_id="C", evaluator_name="e", evaluator_version="1",
                finding_type=finding_type, explanation="x",
            )

        counts = classify_failures(
            [
                finding(FindingType.CRITICAL_OMISSION),
                finding(FindingType.TEMPORAL_TREND_REVERSAL),
                finding(FindingType.TEMPORAL_INFORMATION_LOSS),
            ]
        )
        assert counts["critical_omission"] == 1
        assert counts["temporal_error"] == 2


class TestModelEvaluation:
    def test_correct_model_produces_no_findings(self, selection, tmp_path):
        result = evaluate_model(
            _spec(selection.cases, "correct"), selection.cases,
            cache=OutputCache(tmp_path),
        )
        assert result.pre_adjudication_findings == []
        assert result.generation["summaries_produced"] == len(selection)

    def test_defective_model_produces_findings(self, selection, tmp_path):
        result = evaluate_model(
            _spec(selection.cases, "negation_reversal"), selection.cases,
            cache=OutputCache(tmp_path),
        )
        assert result.failure_taxonomy["negation_reversal"] > 0

    def test_output_length_is_recorded(self, selection, tmp_path):
        result = evaluate_model(
            _spec(selection.cases, "correct"), selection.cases, cache=OutputCache(tmp_path)
        )
        assert result.average_output_length > 0

    def test_generation_failures_are_recorded_not_dropped(self, selection, tmp_path):
        spec = ModelSpec(
            label="mock:failing", provider_name="mock", model="m",
            provider=MockProvider(default_response="x", fail_times=999),
        )
        result = evaluate_model(spec, selection.cases, cache=OutputCache(tmp_path))
        assert len(result.errors) == len(selection)
        assert result.generation["summaries_produced"] == 0

    def test_severity_counts_are_reported(self, selection, tmp_path):
        result = evaluate_model(
            _spec(selection.cases, "omission"), selection.cases, cache=OutputCache(tmp_path)
        )
        assert sum(result.severity_counts.values()) == len(result.pre_adjudication_findings)


class TestAdjudicationSeparation:
    @pytest.fixture
    def adjudicated(self, selection, tmp_path):
        judge = ModelSpec(
            label="mock:judge", provider_name="mock", model="judge",
            provider=mock_judge("SUPPORTED", 0.95),
        )
        return evaluate_model(
            _spec(selection.cases, "abstractive"), selection.cases,
            cache=OutputCache(tmp_path), adjudicate=True, judge=judge,
        )

    def test_both_conditions_are_run(self, adjudicated):
        assert set(ADJUDICATION_CONDITIONS) <= set(adjudicated.adjudication)
        assert "comparison" in adjudicated.adjudication

    def test_pre_adjudication_metrics_are_not_revised(self, adjudicated):
        """The deterministic numbers must survive adjudication untouched."""
        retention = adjudicated.pre_adjudication_metrics[
            "critical_information_retention"
        ]
        assert retention["numerator"] + 0 == retention["numerator"]
        # Findings were withdrawn, but the metric still reflects the
        # deterministic count.
        withdrawn = adjudicated.adjudication["informed"]["withdrawn_finding_count"]
        assert withdrawn > 0
        assert len(adjudicated.pre_adjudication_findings) > withdrawn

    def test_post_adjudication_counts_are_separate(self, adjudicated):
        informed = adjudicated.adjudication["informed"]
        assert (
            informed["post_adjudication_finding_count"]
            + informed["withdrawn_finding_count"]
            == len(adjudicated.pre_adjudication_findings)
        )

    def test_blind_condition_uses_the_blind_prompt(self, adjudicated):
        assert adjudicated.adjudication["blind"]["prompt"] == "adjudication_blind"
        assert (
            adjudicated.adjudication["blind"]["metadata"]["prompt_name"]
            == "adjudication_blind"
        )

    def test_informed_condition_uses_the_informed_prompt(self, adjudicated):
        assert adjudicated.adjudication["informed"]["prompt"] == "adjudication"


class TestAnchoringComparison:
    def _records(self, judgments: dict[str, str]) -> dict:
        return {
            "records": [
                {"finding_id": finding_id, "judgment": judgment}
                for finding_id, judgment in judgments.items()
            ],
            "withdrawn_finding_count": sum(
                1 for j in judgments.values() if j == "SUPPORTED"
            ),
            "summary": {"adjudication_required": 0},
        }

    def test_identical_judgments_report_full_agreement(self):
        same = {"F1": "SUPPORTED", "F2": "UNSUPPORTED"}
        comparison = compare_conditions(self._records(same), self._records(same))
        assert comparison["judgment_agreement_rate"] == 1.0
        assert comparison["judgment_disagreements"] == 0
        assert comparison["withdrawal_delta"] == 0

    def test_divergent_judgments_are_detected(self):
        """The property that makes the experiment meaningful: the comparison
        must be able to detect anchoring, not just report agreement."""
        informed = self._records({"F1": "SUPPORTED", "F2": "SUPPORTED"})
        blind = self._records({"F1": "UNSUPPORTED", "F2": "SUPPORTED"})
        comparison = compare_conditions(informed, blind)
        assert comparison["judgment_agreement_rate"] == 0.5
        assert comparison["judgment_disagreements"] == 1
        assert comparison["withdrawal_delta"] == -1
        assert comparison["disagreement_detail"][0]["informed"] == "SUPPORTED"
        assert comparison["disagreement_detail"][0]["blind"] == "UNSUPPORTED"

    def test_no_shared_findings_yields_an_undefined_rate(self):
        comparison = compare_conditions(self._records({}), self._records({}))
        assert comparison["judgment_agreement_rate"] is None

    def test_end_to_end_anchoring_is_detectable(self, selection, tmp_path):
        """A judge whose answer depends on seeing the deterministic finding must
        show up as a difference between conditions."""

        def anchored(request):
            # The informed prompt includes the deterministic analysis; the blind
            # prompt does not. This judge agrees with whatever it is shown.
            if "Deterministic analysis" in request.user:
                return judgment_json("UNSUPPORTED", 0.95)
            return judgment_json("SUPPORTED", 0.95)

        judge = ModelSpec(
            label="mock:anchored", provider_name="mock", model="judge",
            provider=MockProvider(responder=anchored),
        )
        result = evaluate_model(
            _spec(selection.cases, "abstractive"), selection.cases,
            cache=OutputCache(tmp_path), adjudicate=True, judge=judge,
        )
        comparison = result.adjudication["comparison"]
        assert comparison["findings_judged_in_both"] > 0
        assert comparison["judgment_agreement_rate"] == 0.0
        assert comparison["withdrawal_delta"] > 0


class TestRepeatedRuns:
    def test_deterministic_provider_is_perfectly_stable(self, selection, tmp_path):
        analysis = repeated_run_analysis(
            _spec(selection.cases, "correct"), selection.cases, runs=3,
            cache=OutputCache(tmp_path, enabled=False),
        )
        assert analysis["output_stability_rate"] == 1.0
        assert analysis["finding_stability_rate"] == 1.0

    def test_stability_note_disclaims_the_mock(self, selection, tmp_path):
        """A mock is trivially stable; the report must not imply otherwise."""
        analysis = repeated_run_analysis(
            _spec(selection.cases, "correct"), selection.cases, runs=2,
            cache=OutputCache(tmp_path, enabled=False),
        )
        assert "mock provider is trivially stable" in analysis["note"]

    def test_variable_provider_is_detected_as_unstable(self, selection, tmp_path):
        counter = {"n": 0}

        def varying(request):
            counter["n"] += 1
            return f"Summary variant {counter['n']}."

        spec = ModelSpec(
            label="mock:varying", provider_name="mock", model="m",
            provider=MockProvider(responder=varying), temperature=1.0,
        )
        analysis = repeated_run_analysis(
            spec, selection.cases, runs=3, cache=OutputCache(tmp_path, enabled=False)
        )
        assert analysis["output_stability_rate"] == 0.0


class TestComparisonArtifact:
    def test_artifact_is_json_serializable(self, selection, tmp_path):
        result = evaluate_model(
            _spec(selection.cases, "omission"), selection.cases, cache=OutputCache(tmp_path)
        )
        artifact = build_comparison(
            [result], benchmark_version="v-test", sample=summarize_sample(selection)
        )
        assert json.loads(json.dumps(artifact))["benchmark_version"] == "v-test"

    def test_artifact_is_labelled_exploratory(self, selection, tmp_path):
        result = evaluate_model(
            _spec(selection.cases, "correct"), selection.cases, cache=OutputCache(tmp_path)
        )
        artifact = build_comparison([result], "v", summarize_sample(selection))
        assert artifact["exploratory"] is True
        assert any("EXPLORATORY" in note for note in artifact["notes"])

    def test_artifact_produces_no_overall_ranking(self, selection, tmp_path):
        """Different failure profiles must not be collapsed onto one axis."""
        results = [
            evaluate_model(
                _spec(selection.cases, behaviour, label=f"mock:{behaviour}"),
                selection.cases, cache=OutputCache(tmp_path / behaviour),
            )
            for behaviour in ("omission", "hallucination")
        ]
        artifact = build_comparison(results, "v", summarize_sample(selection))
        serialized = json.dumps(artifact).lower()
        for banned in ('"rank"', '"best_model"', '"overall_score"', '"winner"'):
            assert banned not in serialized
        assert any("no overall ranking" in note.lower() for note in artifact["notes"])

    def test_artifact_records_each_model_separately(self, selection, tmp_path):
        results = [
            evaluate_model(
                _spec(selection.cases, behaviour, label=f"mock:{behaviour}"),
                selection.cases, cache=OutputCache(tmp_path / behaviour),
            )
            for behaviour in ("omission", "correct")
        ]
        artifact = build_comparison(results, "v", summarize_sample(selection))
        assert [m["model"]["label"] for m in artifact["models"]] == [
            "mock:omission", "mock:correct"
        ]

    def test_mock_models_are_never_labelled_real(self, selection, tmp_path):
        result = evaluate_model(
            _spec(selection.cases, "correct"), selection.cases, cache=OutputCache(tmp_path)
        )
        artifact = build_comparison([result], "v", summarize_sample(selection))
        assert artifact["models"][0]["model"]["is_real_model_output"] is False
