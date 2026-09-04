"""Tests for the evaluator registry and the benchmark runner."""

from __future__ import annotations

import pytest

from app.core.enums import CaseType, DataProvenance, FindingType, Severity
from app.evaluation import registry
from app.evaluation.base import DeterministicEvaluator, EvaluatorResult
from app.evaluation.runner import EvaluationRunner, summarize_findings
from app.evaluation.severity import SeverityPolicy
from app.models.schemas import AIOutput, BenchmarkCase


@pytest.fixture(autouse=True)
def clean_registry():
    """Each test starts from the default registration set."""
    for name in list(registry.available()):
        registry.unregister(name)
    registry.register_defaults()
    yield
    for name in list(registry.available()):
        registry.unregister(name)
    registry.register_defaults()


def _case(case_id: str = "CASE_1") -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id, case_type=CaseType.NORMAL, benchmark_version="v-test",
        provenance=DataProvenance.SYNTHETIC_TEMPLATE, age_group="45-54", sex="female",
        encounter_type="Office visit", clinical_note="Note.", reference_summary="Summary.",
    )


def _output(case_id: str = "CASE_1", text: str = "A summary.") -> AIOutput:
    return AIOutput(case_id=case_id, model_id="test-model", summary_text=text)


class CrashingEvaluator(DeterministicEvaluator):
    name = "crashing"
    version = "0.1.0"

    def evaluate(self, case, ai_output):
        raise RuntimeError("evaluator exploded")


class NoisyEvaluator(DeterministicEvaluator):
    name = "noisy"
    version = "0.1.0"

    def evaluate(self, case, ai_output):
        result = self._empty_result(case)
        result.findings.append(
            self.finding(
                case, FindingType.NEGATION_REVERSAL, "Reversed a negative.", index=0,
                details={"fact_importance": "critical"},
            )
        )
        return result


class TestRegistry:
    def test_defaults_register_the_five_deterministic_evaluators(self):
        assert set(registry.available()) == {
            "critical_retention", "medication_consistency", "negation_consistency",
            "temporal_consistency", "fact_support",
        }

    def test_create_returns_a_fresh_instance(self):
        first = registry.create("critical_retention")
        assert first is not registry.create("critical_retention")
        assert first.name == "critical_retention"

    def test_unknown_evaluator_raises_a_helpful_error(self):
        with pytest.raises(KeyError, match="Available"):
            registry.create("nonexistent")

    def test_create_all_instantiates_everything(self):
        assert len(registry.create_all()) == len(registry.available())

    def test_create_all_accepts_a_subset(self):
        evaluators = registry.create_all(["critical_retention"])
        assert [e.name for e in evaluators] == ["critical_retention"]

    def test_registering_a_duplicate_name_is_refused(self):
        """Two evaluators must not silently claim the same slot."""
        with pytest.raises(ValueError, match="already registered"):
            registry.register("critical_retention", CrashingEvaluator)

    def test_replace_allows_a_deliberate_override(self):
        registry.register("critical_retention", CrashingEvaluator, replace=True)
        assert isinstance(registry.create("critical_retention"), CrashingEvaluator)

    def test_a_new_evaluator_needs_no_change_to_existing_ones(self):
        registry.register("noisy", NoisyEvaluator)
        assert "noisy" in registry.available()
        assert registry.create("critical_retention").name == "critical_retention"

    def test_unregister_removes_an_evaluator(self):
        registry.unregister("fact_support")
        assert "fact_support" not in registry.available()

    def test_unregister_is_idempotent(self):
        registry.unregister("absent")


class TestRunnerCaseLevel:
    def test_runs_every_evaluator_over_a_case(self):
        runner = EvaluationRunner()
        result = runner.evaluate_case(_case(), _output())
        assert result.case_id == "CASE_1"
        assert result.metrics

    def test_severity_is_applied_to_every_finding(self):
        """Evaluators detect; the runner classifies. No finding may escape
        without a severity."""
        runner = EvaluationRunner(evaluators=[NoisyEvaluator()])
        result = runner.evaluate_case(_case(), _output())
        assert all(finding.severity is not None for finding in result.findings)

    def test_severity_policy_escalates_for_critical_facts(self):
        runner = EvaluationRunner(evaluators=[NoisyEvaluator()])
        finding = runner.evaluate_case(_case(), _output()).findings[0]
        assert finding.severity is Severity.CRITICAL

    def test_a_custom_policy_changes_the_classification(self):
        policy = SeverityPolicy(
            base_severities={FindingType.NEGATION_REVERSAL: Severity.INFO},
            escalate_for_critical_facts=False,
        )
        runner = EvaluationRunner(evaluators=[NoisyEvaluator()], severity_policy=policy)
        assert runner.evaluate_case(_case(), _output()).findings[0].severity is Severity.INFO

    def test_a_crashing_evaluator_does_not_end_the_run(self):
        """One bad case must not lose the rest of the benchmark."""
        runner = EvaluationRunner(evaluators=[CrashingEvaluator(), NoisyEvaluator()])
        result = runner.evaluate_case(_case(), _output())
        assert len(result.findings) == 1

    def test_a_crashing_evaluator_reports_its_error(self):
        runner = EvaluationRunner(evaluators=[CrashingEvaluator()])
        evaluator_result = runner._run_evaluator(CrashingEvaluator(), _case(), _output())
        assert not evaluator_result.succeeded
        assert "evaluator exploded" in evaluator_result.error

    def test_unrecognised_fact_importance_does_not_crash(self):
        class OddEvaluator(DeterministicEvaluator):
            name = "odd"
            version = "0.1.0"

            def evaluate(self, case, ai_output):
                result = self._empty_result(case)
                result.findings.append(
                    self.finding(
                        case, FindingType.OTHER, "x", index=0,
                        details={"fact_importance": "not-a-real-importance"},
                    )
                )
                return result

        runner = EvaluationRunner(evaluators=[OddEvaluator()])
        assert runner.evaluate_case(_case(), _output()).findings[0].severity is not None


class TestRunnerBenchmarkLevel:
    def test_evaluates_every_case_with_an_output(self):
        cases = [_case("CASE_1"), _case("CASE_2")]
        outputs = {"CASE_1": _output("CASE_1"), "CASE_2": _output("CASE_2")}
        result = EvaluationRunner().evaluate_benchmark(cases, outputs, model_id="m")
        assert result.cases_evaluated == 2

    def test_cases_without_output_are_skipped_not_failed(self):
        """A missing generation is a harness problem, not a model failure;
        scoring it as one would silently penalize the model."""
        cases = [_case("CASE_1"), _case("CASE_2")]
        result = EvaluationRunner().evaluate_benchmark(
            cases, {"CASE_1": _output("CASE_1")}, model_id="m"
        )
        assert result.cases_evaluated == 1
        assert result.configuration["cases_skipped_no_output"] == ["CASE_2"]

    def test_result_records_reproducibility_information(self):
        result = EvaluationRunner().evaluate_benchmark(
            [_case()], {"CASE_1": _output()}, model_id="model-a"
        )
        assert result.model_id == "model-a"
        assert result.benchmark_version == "v-test"
        assert result.evaluation_version
        assert result.evaluators
        assert result.configuration["severity_policy"]

    def test_run_id_is_generated_when_absent(self):
        result = EvaluationRunner().evaluate_benchmark([_case()], {"CASE_1": _output()}, model_id="m")
        assert result.evaluation_run_id.startswith("run-")

    def test_explicit_run_id_is_honoured(self):
        result = EvaluationRunner().evaluate_benchmark(
            [_case()], {"CASE_1": _output()}, model_id="m", evaluation_run_id="RUN_X"
        )
        assert result.evaluation_run_id == "RUN_X"
        assert result.case_results[0].evaluation_run_id == "RUN_X"

    def test_macro_averages_are_reported_alongside_micro(self):
        result = EvaluationRunner().evaluate_benchmark(
            [_case()], {"CASE_1": _output()}, model_id="m"
        )
        assert "macro_averages" in result.configuration

    def test_empty_benchmark_is_handled(self):
        result = EvaluationRunner().evaluate_benchmark([], {}, model_id="m")
        assert result.cases_evaluated == 0
        assert result.benchmark_version == "unknown"

    def test_result_serializes_to_json(self):
        result = EvaluationRunner().evaluate_benchmark(
            [_case()], {"CASE_1": _output()}, model_id="m"
        )
        assert result.model_dump(mode="json")["cases_evaluated"] == 1


class TestFindingSummary:
    def test_counts_by_type_and_severity(self):
        runner = EvaluationRunner(evaluators=[NoisyEvaluator()])
        result = runner.evaluate_benchmark([_case()], {"CASE_1": _output()}, model_id="m")
        summary = summarize_findings(result)
        assert summary["by_type"]["negation_reversal"] == 1
        assert summary["by_severity"]["CRITICAL"] == 1

    def test_empty_result_summarizes_to_nothing(self):
        result = EvaluationRunner(evaluators=[]).evaluate_benchmark(
            [_case()], {"CASE_1": _output()}, model_id="m"
        )
        assert summarize_findings(result) == {"by_type": {}, "by_severity": {}}
