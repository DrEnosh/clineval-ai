"""Tests for the evaluator self-validation script.

The script is the project's answer to "how do you know the evaluator works?",
so its own failure paths matter: it must fail loudly when the evaluator
regresses, and it must not report a skipped injection as a pass.
"""

from __future__ import annotations

import pytest

from app.core.enums import FindingType
from scripts import validate_evaluators as script


class TestArguments:
    def test_parses_documented_arguments(self):
        args = script.parse_args(["--cases", "20", "--seed", "7"])
        assert args.cases == 20 and args.seed == 7

    def test_defaults_are_sensible(self):
        args = script.parse_args([])
        assert args.min_detection_rate == script.DEFAULT_MIN_DETECTION_RATE
        assert args.max_false_positive_rate == script.DEFAULT_MAX_FALSE_POSITIVE_RATE


class TestEndToEnd:
    def test_validation_passes_on_the_current_evaluators(self, capsys):
        exit_code = script.main(["--cases", "12", "--seed", "42"])
        output = capsys.readouterr().out
        assert "OVERALL: PASS" in output
        assert exit_code == 0

    def test_report_shows_zero_false_positives(self, capsys):
        script.main(["--cases", "12", "--seed", "42"])
        output = capsys.readouterr().out
        assert "Findings against correct summaries : 0" in output

    def test_report_lists_every_injection(self, capsys):
        script.main(["--cases", "12", "--seed", "42"])
        output = capsys.readouterr().out
        for name in script.INJECTIONS:
            assert name in output

    def test_report_shows_detected_missed_and_skipped(self, capsys):
        script.main(["--cases", "12", "--seed", "42"])
        output = capsys.readouterr().out
        assert "Det" in output and "Miss" in output and "Skip" in output

    def test_runs_against_a_saved_benchmark_file(self, tmp_path, capsys):
        from scripts import generate_benchmark as generate_script

        generate_script.main(["--cases", "10", "--seed", "42", "--out", str(tmp_path)])
        cases_file = next(tmp_path.rglob("cases_*.json"))
        assert script.main(["--cases", "10", "--cases-file", str(cases_file)]) == 0
        assert "OVERALL: PASS" in capsys.readouterr().out


class TestReportingLogic:
    def _outcome(self, detected: int, missed: int, skipped: int) -> script.InjectionOutcome:
        return script.InjectionOutcome(
            name="test_injection", expected=FindingType.CONTRADICTION,
            detected=detected, missed=missed, skipped=skipped,
        )

    def test_detection_rate_ignores_skipped_cases(self):
        outcome = self._outcome(detected=8, missed=2, skipped=90)
        assert outcome.applicable == 10
        assert outcome.detection_rate == pytest.approx(0.8)

    def test_detection_rate_is_none_when_nothing_applied(self):
        assert self._outcome(0, 0, 50).detection_rate is None

    def test_an_injection_that_never_applied_is_not_a_pass(self, capsys):
        """A skipped injection validated nothing, so it must not be reported
        as success - that would be the self-flattery this harness prevents."""
        passed = script.print_report(
            case_count=10, false_positives=0,
            outcomes={"never_applies": self._outcome(0, 0, 10)},
            min_detection_rate=0.9, max_false_positive_rate=0.02,
        )
        assert passed is False
        assert "SKIPPED" in capsys.readouterr().out

    def test_false_positives_fail_the_report(self, capsys):
        passed = script.print_report(
            case_count=10, false_positives=5,
            outcomes={"ok": self._outcome(10, 0, 0)},
            min_detection_rate=0.9, max_false_positive_rate=0.02,
        )
        assert passed is False
        assert "OVERALL: FAIL" in capsys.readouterr().out

    def test_low_detection_rate_fails_the_report(self, capsys):
        passed = script.print_report(
            case_count=10, false_positives=0,
            outcomes={"weak": self._outcome(5, 5, 0)},
            min_detection_rate=0.9, max_false_positive_rate=0.02,
        )
        assert passed is False
        assert "FAIL" in capsys.readouterr().out

    def test_a_clean_report_passes(self, capsys):
        passed = script.print_report(
            case_count=10, false_positives=0,
            outcomes={"good": self._outcome(10, 0, 0)},
            min_detection_rate=0.9, max_false_positive_rate=0.02,
        )
        assert passed is True
        assert "OVERALL: PASS" in capsys.readouterr().out

    def test_missed_cases_are_listed(self, capsys):
        outcome = self._outcome(1, 1, 0)
        outcome.missed_cases = ["SYN_0001"]
        script.print_report(
            case_count=2, false_positives=0, outcomes={"partial": outcome},
            min_detection_rate=0.9, max_false_positive_rate=0.02,
        )
        assert "SYN_0001" in capsys.readouterr().out

    def test_zero_cases_does_not_divide_by_zero(self, capsys):
        script.print_report(
            case_count=0, false_positives=0, outcomes={},
            min_detection_rate=0.9, max_false_positive_rate=0.02,
        )
        assert "Cases: 0" in capsys.readouterr().out


class TestRegressionDetection:
    def test_a_broken_evaluator_is_caught(self, monkeypatch, capsys):
        """The harness must fail when the evaluator stops detecting - this is
        the guarantee the whole script exists to provide."""
        from app.evaluation.runner import EvaluationRunner
        from app.models.schemas import CaseResult

        def blind_evaluate_case(self, case, ai_output, evaluation_run_id="adhoc"):
            return CaseResult(evaluation_run_id=evaluation_run_id, case_id=case.case_id)

        monkeypatch.setattr(EvaluationRunner, "evaluate_case", blind_evaluate_case)
        assert script.main(["--cases", "8", "--seed", "42"]) == 1
        assert "OVERALL: FAIL" in capsys.readouterr().out
