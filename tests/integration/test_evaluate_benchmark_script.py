"""End-to-end tests for the evaluation CLI.

Exercises the documented command, offline throughout. The properties under
test are the ones a user relies on: the run is reproducible, it records what
produced it, nothing leaves the machine by default, and no fabricated model
result can be mistaken for a real one.
"""

from __future__ import annotations

import json

import pytest

from app.core.config import Settings
from scripts import evaluate_benchmark as script


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "runs"


def run_cli(args, run_dir):
    return script.main([*args, "--out", str(run_dir), "--no-cache"])


def load_run(run_dir):
    paths = list(run_dir.glob("run-*.json"))
    assert len(paths) == 1, f"expected one run file, found {len(paths)}"
    return json.loads(paths[0].read_text(encoding="utf-8"))


class TestArguments:
    def test_parses_the_documented_command(self):
        args = script.parse_args(["--model", "some-model", "--cases", "10"])
        assert args.model == "some-model"
        assert args.cases == 10

    def test_defaults_to_the_offline_mock_provider(self):
        assert script.parse_args([]).provider == "mock"

    def test_adjudication_is_off_by_default(self):
        assert script.parse_args([]).adjudicate is False


class TestEndToEnd:
    def test_runs_and_saves_a_result(self, run_dir, capsys):
        assert run_cli(["--cases", "5"], run_dir) == 0
        run = load_run(run_dir)
        assert run["cases_evaluated"] == 5
        assert run["run_id"].startswith("run-")
        assert "ClinEval AI - benchmark evaluation" in capsys.readouterr().out

    def test_correct_summaries_produce_no_findings(self, run_dir):
        run_cli(["--cases", "5", "--behaviour", "correct"], run_dir)
        assert load_run(run_dir)["findings"] == []

    @pytest.mark.parametrize(
        "behaviour,expected",
        [
            ("omission", "critical_omission"),
            ("negation_reversal", "negation_reversal"),
            ("medication_error", "medication_status_error"),
            ("temporal_reversal", "temporal_trend_reversal"),
            ("hallucination", "unsupported_addition"),
        ],
    )
    def test_defective_models_produce_the_expected_findings(
        self, run_dir, behaviour, expected
    ):
        run_cli(["--cases", "10", "--behaviour", behaviour], run_dir)
        assert expected in load_run(run_dir)["findings_summary"]["by_type"]

    def test_empty_output_is_handled(self, run_dir):
        assert run_cli(["--cases", "3", "--behaviour", "empty"], run_dir) == 0
        assert "empty_output" in load_run(run_dir)["findings_summary"]["by_type"]

    def test_run_records_reproducibility_metadata(self, run_dir):
        run_cli(["--cases", "3", "--seed", "7"], run_dir)
        run = load_run(run_dir)
        for key in (
            "run_id", "benchmark_version", "case_count", "model", "provider",
            "evaluation_version", "configuration", "generated_at", "evaluators",
        ):
            assert key in run, key
        assert run["generation"]["prompt_version"] == "v1"
        assert run["generation"]["prompt_hash"]
        assert run["configuration"]["seed"] == 7

    def test_run_reports_cost_tokens_and_latency(self, run_dir):
        run_cli(["--cases", "3"], run_dir)
        generation = load_run(run_dir)["generation_summary"]
        assert generation["tokens"]["total"] > 0
        assert generation["model_calls"] == 3
        # The mock has no real cost, so it is reported as unknown, never 0.0.
        assert generation["estimated_cost_usd"] is None

    def test_mock_output_is_never_labelled_real(self, run_dir):
        """No fabricated model result may be mistaken for a real one."""
        run_cli(["--cases", "3"], run_dir)
        run = load_run(run_dir)
        assert run["generation"]["is_real_model_output"] is False
        assert run["provider"] == "mock"

    def test_run_states_that_no_risk_score_is_produced(self, run_dir):
        run_cli(["--cases", "2"], run_dir)
        notes = " ".join(load_run(run_dir)["notes"]).lower()
        assert "no overall risk score" in notes
        assert "synthetic" in notes

    def test_identical_invocations_produce_identical_results(self, tmp_path):
        """Reproducibility: the same command twice must give the same findings
        and metrics."""
        first_dir, second_dir = tmp_path / "a", tmp_path / "b"
        run_cli(["--cases", "6", "--behaviour", "omission"], first_dir)
        run_cli(["--cases", "6", "--behaviour", "omission"], second_dir)

        first, second = load_run(first_dir), load_run(second_dir)
        assert first["metrics"] == second["metrics"]
        assert [f["finding_id"] for f in first["findings"]] == [
            f["finding_id"] for f in second["findings"]
        ]

    def test_a_small_case_count_is_supported(self, run_dir):
        assert run_cli(["--cases", "1"], run_dir) == 0

    def test_runs_against_a_saved_benchmark_file(self, tmp_path, run_dir):
        from scripts import generate_benchmark as generate_script

        benchmark_dir = tmp_path / "bench"
        generate_script.main(["--cases", "6", "--seed", "42", "--out", str(benchmark_dir)])
        cases_file = next(benchmark_dir.rglob("cases_*.json"))

        assert run_cli(["--benchmark", str(cases_file), "--cases", "4"], run_dir) == 0
        assert load_run(run_dir)["cases_evaluated"] == 4


class TestAdjudicationPath:
    def test_adjudication_records_are_saved(self, run_dir):
        run_cli(["--cases", "8", "--behaviour", "abstractive", "--adjudicate"], run_dir)
        adjudication = load_run(run_dir)["adjudication"]
        assert adjudication is not None
        assert adjudication["summary"]["findings_considered"] > 0
        assert adjudication["records"]

    def test_withdrawn_findings_are_retained_for_audit(self, run_dir):
        """Adjudication may remove a finding from the headline list, but the
        run must still show what was removed and why."""
        run_cli(["--cases", "8", "--behaviour", "abstractive", "--adjudicate"], run_dir)
        adjudication = load_run(run_dir)["adjudication"]
        if adjudication["summary"]["resolved_by_llm"]:
            assert adjudication["withdrawn_findings"]
            rules = {r["resolution_rule"] for r in adjudication["records"] if r["finding_withdrawn"]}
            assert rules == {"R2"}

    def test_metrics_are_labelled_pre_adjudication(self, run_dir):
        """A model's judgments must not silently move the headline metrics."""
        run_cli(["--cases", "5", "--behaviour", "abstractive", "--adjudicate"], run_dir)
        run = load_run(run_dir)
        assert run["metrics_are_pre_adjudication"] is True
        assert run["findings_are_post_adjudication"] is True

    def test_adjudication_metadata_labels_judgments_as_model_based(self, run_dir):
        run_cli(["--cases", "4", "--adjudicate"], run_dir)
        metadata = load_run(run_dir)["adjudication"]["metadata"]
        assert metadata["is_model_based_judgment"] is True
        assert metadata["prompt_version"] == "v1"

    def test_contradiction_findings_survive_a_lenient_judge(self, run_dir):
        """Rule R3 end-to-end: a model cannot erase a detected contradiction."""
        run_cli(
            ["--cases", "10", "--behaviour", "negation_reversal", "--adjudicate"], run_dir
        )
        run = load_run(run_dir)
        assert "negation_reversal" in run["findings_summary"]["by_type"]
        withdrawn_types = {
            f["finding_type"] for f in run["adjudication"]["withdrawn_findings"]
        }
        assert "negation_reversal" not in withdrawn_types

    def test_no_adjudication_section_when_disabled(self, run_dir):
        run_cli(["--cases", "3"], run_dir)
        assert load_run(run_dir)["adjudication"] is None


class TestFailureHandling:
    def test_unknown_provider_reports_availability(self, run_dir, capsys):
        assert run_cli(["--cases", "2", "--provider", "not_a_provider"], run_dir) == 1
        assert "Provider availability" in capsys.readouterr().err

    def test_unconfigured_remote_provider_is_refused(self, run_dir, capsys, monkeypatch):
        """Nothing leaves the machine without an explicit credential."""
        blank = Settings(
            anthropic_api_key=None, openai_api_key=None, google_api_key=None,
            local_base_url=None, _env_file=None,
        )
        monkeypatch.setattr("app.providers.registry.default_settings", blank)
        monkeypatch.setattr("app.providers.remote.default_settings", blank)

        assert run_cli(["--cases", "2", "--provider", "anthropic"], run_dir) == 1
        assert "CLINEVAL_ANTHROPIC_API_KEY" in capsys.readouterr().err

    def test_remote_provider_warns_before_sending(self, capsys, tmp_path, monkeypatch):
        configured = Settings(anthropic_api_key="test-key-not-real", _env_file=None)
        monkeypatch.setattr("app.providers.registry.default_settings", configured)
        monkeypatch.setattr("app.providers.remote.default_settings", configured)

        # Fails at the call, but the warning must already have been printed.
        script.main(
            ["--cases", "1", "--provider", "anthropic", "--model", "x",
             "--out", str(tmp_path), "--no-cache"]
        )
        output = capsys.readouterr().out
        assert "is remote" in output
        assert "synthetic" in output.lower()

    def test_zero_cases_is_reported(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(script, "load_cases", lambda args: [])
        assert script.main(["--out", str(tmp_path)]) == 1
        assert "No cases to evaluate" in capsys.readouterr().err

    def test_quiet_suppresses_the_report(self, run_dir, capsys):
        run_cli(["--cases", "2", "--quiet"], run_dir)
        assert capsys.readouterr().out == ""


class TestCaching:
    @staticmethod
    def _run_and_load(args, runs_dir):
        """Run the CLI and load exactly the file it just wrote.

        Run ids are random hex, so sorting filenames does not order runs by
        time. Identifying the new file by set difference is exact.
        """
        before = set(runs_dir.glob("run-*.json")) if runs_dir.exists() else set()
        script.main(args)
        created = set(runs_dir.glob("run-*.json")) - before
        assert len(created) == 1, f"expected one new run file, got {len(created)}"
        return json.loads(created.pop().read_text(encoding="utf-8"))

    def test_second_run_reuses_cached_output(self, tmp_path):
        runs_dir = tmp_path / "runs"
        args = ["--cases", "4", "--out", str(runs_dir)]

        first = self._run_and_load(args, runs_dir)
        assert first["generation_summary"]["model_calls"] == 4

        second = self._run_and_load(args, runs_dir)
        assert second["generation_summary"]["served_from_cache"] == 4
        assert second["generation_summary"]["model_calls"] == 0

    def test_force_regenerates(self, tmp_path):
        runs_dir = tmp_path / "runs"
        args = ["--cases", "3", "--out", str(runs_dir)]

        self._run_and_load(args, runs_dir)
        second = self._run_and_load([*args, "--force"], runs_dir)
        assert second["generation_summary"]["model_calls"] == 3
        assert second["generation_summary"]["served_from_cache"] == 0
