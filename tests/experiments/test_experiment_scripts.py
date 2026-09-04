"""Tests for the Phase 5 CLI scripts and the challenge suite.

The property that matters most here is the refusal: `run_experiment` must not
substitute a mock when a real provider is unavailable. A run that silently
degraded to synthetic output would write a result file indistinguishable at a
glance from a real experiment, which is the single worst failure this phase
could have.
"""

from __future__ import annotations

import json

import pytest

from app.core.config import Settings
from scripts import analyze_runs, run_experiment


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path / "experiments"


def load_artifact(out_dir):
    paths = [p for p in out_dir.glob("exp-*.json") if not p.name.endswith("-review.json")]
    assert len(paths) == 1, f"expected one artifact, found {len(paths)}"
    return json.loads(paths[0].read_text(encoding="utf-8"))


def load_review(out_dir):
    paths = list(out_dir.glob("exp-*-review.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


class TestRefusalToFabricate:
    def test_unconfigured_provider_is_refused(self, out_dir, capsys, monkeypatch):
        blank = Settings(
            anthropic_api_key=None, openai_api_key=None, google_api_key=None,
            local_base_url=None, _env_file=None,
        )
        monkeypatch.setattr("app.providers.registry.default_settings", blank)
        monkeypatch.setattr("app.providers.remote.default_settings", blank)

        exit_code = run_experiment.main(
            ["--models", "anthropic:some-model", "--out", str(out_dir)]
        )
        assert exit_code == 1

        error = capsys.readouterr().err
        assert "CLINEVAL_ANTHROPIC_API_KEY" in error
        assert "does not fall back to a mock provider" in error

    def test_no_artifact_is_written_on_refusal(self, out_dir, monkeypatch):
        """A refused run must leave nothing behind that could be mistaken for
        a result."""
        blank = Settings(anthropic_api_key=None, _env_file=None)
        monkeypatch.setattr("app.providers.registry.default_settings", blank)
        monkeypatch.setattr("app.providers.remote.default_settings", blank)

        run_experiment.main(["--models", "anthropic:m", "--out", str(out_dir)])
        assert not out_dir.exists() or not list(out_dir.glob("exp-*.json"))

    def test_malformed_model_spec_is_rejected(self, out_dir, capsys):
        assert run_experiment.main(["--models", "no-colon", "--out", str(out_dir)]) == 1
        assert "provider:model-id" in capsys.readouterr().err

    def test_unknown_provider_is_rejected(self, out_dir, capsys):
        assert run_experiment.main(["--models", "nope:m", "--out", str(out_dir)]) == 1
        assert "Provider availability" in capsys.readouterr().err


class TestExperimentRun:
    def test_runs_with_the_mock_provider(self, out_dir, capsys):
        assert run_experiment.main(
            ["--models", "mock:demo", "--out", str(out_dir), "--no-cache"]
        ) == 0
        artifact = load_artifact(out_dir)
        assert artifact["models"][0]["model"]["label"] == "mock:demo"
        assert artifact["exploratory"] is True

    def test_sample_is_the_ten_case_stratified_set(self, out_dir):
        run_experiment.main(["--models", "mock:demo", "--out", str(out_dir), "--no-cache"])
        artifact = load_artifact(out_dir)
        assert artifact["sample"]["case_count"] == 10
        assert artifact["sample"]["is_complete"] is True

    def test_mock_results_are_labelled_not_real(self, out_dir):
        run_experiment.main(["--models", "mock:demo", "--out", str(out_dir), "--no-cache"])
        artifact = load_artifact(out_dir)
        assert artifact["models"][0]["model"]["is_real_model_output"] is False

    def test_several_models_are_evaluated_on_the_same_sample(self, out_dir):
        run_experiment.main(
            ["--models", "mock:a", "mock:b", "--out", str(out_dir), "--no-cache"]
        )
        artifact = load_artifact(out_dir)
        assert len(artifact["models"]) == 2
        assert artifact["sample"]["case_count"] == 10

    def test_adjudication_runs_both_conditions(self, out_dir):
        run_experiment.main(
            [
                "--models", "mock:demo", "--behaviour", "abstractive",
                "--adjudicate", "--out", str(out_dir), "--no-cache",
            ]
        )
        model = load_artifact(out_dir)["models"][0]
        assert set(model["post_adjudication"]) == {"informed", "blind"}
        assert model["adjudication_comparison"] is not None

    def test_repeated_runs_are_recorded(self, out_dir):
        run_experiment.main(
            [
                "--models", "mock:demo", "--repeat-model", "mock:demo",
                "--repeats", "3", "--out", str(out_dir), "--no-cache",
            ]
        )
        repeats = load_artifact(out_dir)["models"][0]["repeated_runs"]
        assert repeats["runs_per_case"] == 3
        assert repeats["cases"] == 10

    def test_review_set_is_written(self, out_dir):
        run_experiment.main(
            [
                "--models", "mock:demo", "--behaviour", "omission",
                "--out", str(out_dir), "--no-cache",
            ]
        )
        review = load_review(out_dir)
        assert review["entry_count"] > 0
        entry = review["entries"][0]
        for key in (
            "case_id", "source_note_excerpt", "deterministic_finding",
            "llm_judgment", "final_disposition", "selection_reason",
        ):
            assert key in entry

    def test_review_entries_carry_the_evidence_chain(self, out_dir):
        run_experiment.main(
            [
                "--models", "mock:demo", "--behaviour", "omission",
                "--out", str(out_dir), "--no-cache",
            ]
        )
        finding = load_review(out_dir)["entries"][0]["deterministic_finding"]
        assert finding["explanation"]
        assert finding["source_evidence"]
        assert finding["severity"]

    def test_run_is_reproducible(self, tmp_path):
        first, second = tmp_path / "a", tmp_path / "b"
        for directory in (first, second):
            run_experiment.main(
                [
                    "--models", "mock:demo", "--behaviour", "omission",
                    "--out", str(directory), "--no-cache",
                ]
            )
        a, b = load_artifact(first), load_artifact(second)
        assert a["sample"]["case_ids"] == b["sample"]["case_ids"]
        assert (
            a["models"][0]["pre_adjudication_metrics"]
            == b["models"][0]["pre_adjudication_metrics"]
        )
        assert a["models"][0]["failure_taxonomy"] == b["models"][0]["failure_taxonomy"]


class TestAnalyzeRuns:
    @pytest.fixture
    def populated(self, out_dir):
        run_experiment.main(
            [
                "--models", "mock:demo", "--behaviour", "abstractive",
                "--adjudicate", "--out", str(out_dir), "--no-cache", "--quiet",
            ]
        )
        return out_dir

    def test_reports_failure_modes(self, populated, capsys):
        assert analyze_runs.main(["--dir", str(populated)]) == 0
        output = capsys.readouterr().out
        assert "Recurring failure modes" in output
        assert "Per-model failure profile" in output

    def test_flags_potential_not_confirmed_false_positives(self, populated, capsys):
        """Terminology discipline: nothing is called a false positive until a
        person has verified it."""
        analyze_runs.main(["--dir", str(populated)])
        output = capsys.readouterr().out
        assert "Potential false positives" in output
        assert "POTENTIAL false positive" in output
        assert "None has been verified by a person" in output

    def test_warns_when_no_real_model_output_is_present(self, populated, capsys):
        analyze_runs.main(["--dir", str(populated)])
        output = capsys.readouterr().out
        assert "all results are from the mock provider" in output
        assert "Nothing here is evidence about how any real model behaves" in output

    def test_reports_the_anchoring_comparison(self, populated, capsys):
        analyze_runs.main(["--dir", str(populated)])
        assert "Blind vs informed adjudication" in capsys.readouterr().out

    def test_writes_machine_readable_analysis(self, populated, tmp_path):
        out_file = tmp_path / "analysis.json"
        analyze_runs.main(["--dir", str(populated), "--out", str(out_file)])
        payload = json.loads(out_file.read_text(encoding="utf-8"))
        assert "analysis" in payload and "review" in payload
        assert payload["review"]["verification_notice"]

    def test_missing_artifacts_are_reported(self, tmp_path, capsys):
        assert analyze_runs.main(["--dir", str(tmp_path / "empty")]) == 1
        assert "No experiment artifacts found" in capsys.readouterr().err

    def test_unreadable_artifact_is_skipped_not_fatal(self, populated, capsys):
        (populated / "exp-corrupt.json").write_text("{ broken", encoding="utf-8")
        assert analyze_runs.main(["--dir", str(populated)]) == 0
        assert "Skipping unreadable artifact" in capsys.readouterr().err


class TestChallengeSuite:
    @pytest.fixture(scope="class")
    @classmethod
    def outcomes(cls):
        from app.benchmark import generate_benchmark
        from app.experiments.challenges import run_challenges
        from app.experiments.selection import select_experiment_sample

        cases = generate_benchmark(case_count=100, seed=42)
        return run_challenges(select_experiment_sample(cases).cases)

    def test_suite_covers_every_category(self, outcomes):
        from app.experiments.challenges import TRANSFORMATIONS

        produced = {outcome.challenge.category for outcome in outcomes}
        # A transformation that never changes any summary produces no challenge,
        # which would silently reduce coverage.
        assert produced == set(TRANSFORMATIONS)

    def test_no_challenge_is_an_identity_transformation(self, outcomes):
        from app.benchmark import generate_benchmark

        by_id = {c.case_id: c for c in generate_benchmark(case_count=100, seed=42)}
        for outcome in outcomes:
            case = by_id[outcome.challenge.case_id]
            assert outcome.challenge.summary != case.reference_summary

    def test_undecided_challenges_are_not_counted_as_passes(self, outcomes):
        from app.experiments.challenges import summarize_outcomes

        summary = summarize_outcomes(outcomes)
        assert summary["decidable"] < summary["total_challenges"]
        assert summary["by_category"]["abstraction"]["undecided"] > 0

    def test_summary_disclaims_that_these_are_model_outputs(self, outcomes):
        from app.experiments.challenges import summarize_outcomes

        assert "not real model outputs" in summarize_outcomes(outcomes)["note"]

    def test_outcome_reports_are_machine_readable(self, outcomes):
        report = outcomes[0].report()
        assert set(report) >= {
            "challenge_id", "category", "expectation", "findings", "passed", "rationale"
        }
        json.dumps(report)
