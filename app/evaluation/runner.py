"""Benchmark-level evaluation runner.

Drives evaluators over a benchmark, applies the severity policy, and
aggregates metrics into an `EvaluationResult`.

    benchmark -> for each case -> evaluate -> collect findings -> aggregate

Two responsibilities the runner owns and no evaluator does:

**Severity assignment.** Evaluators detect; the runner applies the policy to
every finding before it leaves. A finding reaching a caller without a severity
would mean the policy was skipped, so the runner guarantees against it.

**Isolation of failures.** An evaluator that raises on one case must not lose
the other ninety-nine. Exceptions are caught, recorded on that evaluator's
result for that case, and the run continues. A crashed evaluator produces a
visible error, never a silently missing metric.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.core.enums import FactImportance, Severity
from app.evaluation import metrics as metrics_module
from app.evaluation.base import EVALUATION_VERSION, BaseEvaluator, EvaluatorResult
from app.evaluation.registry import create_all, register_defaults
from app.evaluation.severity import SeverityPolicy, load_severity_policy
from app.models.schemas import (
    AIOutput,
    BenchmarkCase,
    CaseResult,
    EvaluationFinding,
    EvaluationResult,
    MetricValue,
)


class EvaluationRunner:
    """Runs a set of evaluators over a benchmark and aggregates the results."""

    def __init__(
        self,
        evaluators: list[BaseEvaluator] | None = None,
        severity_policy: SeverityPolicy | None = None,
        evaluation_version: str = EVALUATION_VERSION,
    ):
        if evaluators is None:
            register_defaults()
            evaluators = create_all()
        self.evaluators = evaluators
        self.severity_policy = severity_policy or load_severity_policy()
        self.evaluation_version = evaluation_version

    # -- case level ---------------------------------------------------------

    def evaluate_case(
        self, case: BenchmarkCase, ai_output: AIOutput, evaluation_run_id: str = "adhoc"
    ) -> CaseResult:
        """Run every evaluator over one case."""
        findings: list[EvaluationFinding] = []
        case_metrics: dict[str, MetricValue] = {}

        for evaluator in self.evaluators:
            result = self._run_evaluator(evaluator, case, ai_output)
            case_metrics.update(result.metrics)
            findings.extend(result.findings)

        for finding in findings:
            self._apply_severity(finding)

        return CaseResult(
            evaluation_run_id=evaluation_run_id,
            case_id=case.case_id,
            metrics=case_metrics,
            findings=findings,
        )

    def _run_evaluator(
        self, evaluator: BaseEvaluator, case: BenchmarkCase, ai_output: AIOutput
    ) -> EvaluatorResult:
        try:
            return evaluator.evaluate(case, ai_output)
        except Exception as error:  # noqa: BLE001 - one bad case must not end the run
            return EvaluatorResult(
                evaluator_name=evaluator.name,
                evaluator_version=evaluator.version,
                case_id=case.case_id,
                error=f"{type(error).__name__}: {error}",
            )

    def _apply_severity(self, finding: EvaluationFinding) -> None:
        """Classify a finding's severity from the policy, never from the evaluator."""
        raw_importance = finding.details.get("fact_importance")
        importance = None
        if isinstance(raw_importance, str):
            try:
                importance = FactImportance(raw_importance)
            except ValueError:
                importance = None

        is_critical_value = bool(finding.details.get("is_critical_value"))
        finding.severity = self.severity_policy.assign(
            finding.finding_type,
            fact_importance=importance,
            is_critical_value=is_critical_value,
        )

    # -- benchmark level ----------------------------------------------------

    def evaluate_benchmark(
        self,
        cases: list[BenchmarkCase],
        outputs: dict[str, AIOutput],
        model_id: str,
        benchmark_version: str | None = None,
        evaluation_run_id: str | None = None,
        configuration: dict | None = None,
    ) -> EvaluationResult:
        """Evaluate every case for which an output exists.

        Cases without an output are skipped rather than scored as failures:
        absence of a generation is a harness problem, not a model failure, and
        counting it as one would silently penalize the model. The count of
        skipped cases is recorded in `configuration`.
        """
        run_id = evaluation_run_id or f"run-{uuid.uuid4().hex[:12]}"
        case_results: list[CaseResult] = []
        skipped: list[str] = []

        for case in cases:
            output = outputs.get(case.case_id)
            if output is None:
                skipped.append(case.case_id)
                continue
            case_results.append(self.evaluate_case(case, output, run_id))

        aggregated = metrics_module.aggregate([result.metrics for result in case_results])
        macro = metrics_module.macro_average([result.metrics for result in case_results])

        version = benchmark_version or (cases[0].benchmark_version if cases else "unknown")
        all_findings = [finding for result in case_results for finding in result.findings]

        return EvaluationResult(
            evaluation_run_id=run_id,
            benchmark_version=version,
            model_id=model_id,
            evaluation_version=self.evaluation_version,
            generated_at=datetime.now(timezone.utc),
            cases_evaluated=len(case_results),
            evaluators=[
                {"name": evaluator.name, "version": evaluator.version}
                for evaluator in self.evaluators
            ],
            metrics=aggregated,
            case_results=case_results,
            findings=all_findings,
            configuration={
                **(configuration or {}),
                "macro_averages": macro,
                "cases_skipped_no_output": skipped,
                "severity_policy": self.severity_policy.model_dump(mode="json"),
            },
        )


def summarize_findings(result: EvaluationResult) -> dict[str, dict[str, int]]:
    """Count findings by type and by severity, for reporting."""
    by_type: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for finding in result.findings:
        by_type[finding.finding_type.value] = by_type.get(finding.finding_type.value, 0) + 1
        key = finding.severity.value if finding.severity else "UNCLASSIFIED"
        by_severity[key] = by_severity.get(key, 0) + 1
    return {
        "by_type": dict(sorted(by_type.items())),
        "by_severity": {
            level.value: by_severity.get(level.value, 0)
            for level in Severity
            if by_severity.get(level.value, 0)
        },
    }
