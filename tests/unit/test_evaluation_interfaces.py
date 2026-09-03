"""Tests for the evaluation engine contract.

There are no concrete evaluators yet (Phase 4), so these tests verify the
interface behaves correctly for a stand-in implementation: the abstract base
cannot be instantiated, results are validated, and the pipeline dispatches
correctly - including skipping consistency evaluators, which need multiple
runs rather than a single output.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.enums import CaseType, DataProvenance, FailureType, Severity
from app.evaluation.base import (
    BaseEvaluator,
    ConsistencyEvaluator,
    EvaluationDimensionResult,
    EvaluationPipeline,
)
from app.models.schemas import AIOutput, BenchmarkCase, Failure


@pytest.fixture
def case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="CASE_TEST", case_type=CaseType.NORMAL, benchmark_version="v1",
        provenance=DataProvenance.MANUAL_SYNTHETIC, age_group="45-54", sex="female",
        encounter_type="Office visit", clinical_note="Note.", reference_summary="Summary.",
    )


@pytest.fixture
def ai_output() -> AIOutput:
    return AIOutput(case_id="CASE_TEST", model_id="model-a", summary_text="Generated summary.")


class StubEvaluator(BaseEvaluator):
    name = "stub"
    version = "0.1.0"

    def evaluate(self, case, ai_output):
        return EvaluationDimensionResult(
            dimension=self.name, evaluator_version=self.version, score=1.0,
        )


class StubFailingEvaluator(BaseEvaluator):
    name = "stub_failing"
    version = "0.1.0"

    def evaluate(self, case, ai_output):
        return EvaluationDimensionResult(
            dimension=self.name, evaluator_version=self.version, score=0.5,
            failures=[
                Failure(
                    type=FailureType.CRITICAL_OMISSION, severity=Severity.HIGH,
                    explanation="Stub failure.", evidence="Stub evidence.",
                    case_id=case.case_id,
                )
            ],
        )


class StubConsistencyEvaluator(ConsistencyEvaluator):
    name = "stub_consistency"
    version = "0.1.0"

    def evaluate_repeated(self, case, ai_outputs):
        return EvaluationDimensionResult(
            dimension=self.name, evaluator_version=self.version,
            score=1.0 if len({o.summary_text for o in ai_outputs}) == 1 else 0.0,
        )


class TestEvaluationDimensionResult:
    def test_rejects_score_outside_zero_to_one(self):
        with pytest.raises(ValidationError):
            EvaluationDimensionResult(dimension="d", evaluator_version="0.1.0", score=1.5)
        with pytest.raises(ValidationError):
            EvaluationDimensionResult(dimension="d", evaluator_version="0.1.0", score=-0.1)

    def test_accepts_boundary_scores(self):
        assert EvaluationDimensionResult(dimension="d", evaluator_version="0.1.0", score=0.0).score == 0.0
        assert EvaluationDimensionResult(dimension="d", evaluator_version="0.1.0", score=1.0).score == 1.0


class TestBaseEvaluator:
    def test_abstract_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseEvaluator()

    def test_stub_evaluator_returns_versioned_result(self, case, ai_output):
        result = StubEvaluator().evaluate(case, ai_output)
        assert result.dimension == "stub"
        assert result.evaluator_version == "0.1.0"


class TestConsistencyEvaluator:
    def test_single_output_evaluate_is_rejected(self, case, ai_output):
        with pytest.raises(NotImplementedError):
            StubConsistencyEvaluator().evaluate(case, ai_output)

    def test_identical_repeated_runs_score_one(self, case):
        outputs = [
            AIOutput(case_id=case.case_id, model_id="m", summary_text="same", run_index=i)
            for i in range(3)
        ]
        assert StubConsistencyEvaluator().evaluate_repeated(case, outputs).score == 1.0

    def test_divergent_repeated_runs_score_zero(self, case):
        outputs = [
            AIOutput(case_id=case.case_id, model_id="m", summary_text=f"different {i}", run_index=i)
            for i in range(3)
        ]
        assert StubConsistencyEvaluator().evaluate_repeated(case, outputs).score == 0.0


class TestEvaluationPipeline:
    def test_runs_every_registered_evaluator(self, case, ai_output):
        pipeline = EvaluationPipeline([StubEvaluator(), StubFailingEvaluator()])
        results = pipeline.run(case, ai_output)
        assert [r.dimension for r in results] == ["stub", "stub_failing"]

    def test_collects_failures_from_results(self, case, ai_output):
        results = EvaluationPipeline([StubFailingEvaluator()]).run(case, ai_output)
        assert results[0].failures[0].severity is Severity.HIGH

    def test_skips_consistency_evaluators_in_single_output_run(self, case, ai_output):
        pipeline = EvaluationPipeline([StubEvaluator(), StubConsistencyEvaluator()])
        results = pipeline.run(case, ai_output)
        assert [r.dimension for r in results] == ["stub"]

    def test_empty_pipeline_returns_no_results(self, case, ai_output):
        assert EvaluationPipeline([]).run(case, ai_output) == []
