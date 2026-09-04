"""Tests for the evaluation engine contract.

Phase 3 replaced the Phase 1 stub contract: evaluators now return
`EvaluatorResult` (findings plus metrics) rather than a bare dimension score,
and the `EvaluationRunner` supersedes the old pipeline. These tests cover the
contract itself - the concrete evaluators are tested in
tests/evaluation/test_evaluators.py.
"""

from __future__ import annotations

import pytest

from app.core.enums import CaseType, DataProvenance, FindingType, Severity
from app.evaluation.base import (
    EVALUATION_VERSION,
    BaseEvaluator,
    ConsistencyEvaluator,
    DeterministicEvaluator,
    EvaluatorResult,
    LLMEvaluator,
)
from app.evaluation.matching import FactMatcher
from app.models.schemas import AIOutput, BenchmarkCase


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


class StubEvaluator(DeterministicEvaluator):
    name = "stub"
    version = "0.1.0"

    def evaluate(self, case, ai_output):
        return self._empty_result(case)


class StubFindingEvaluator(DeterministicEvaluator):
    name = "stub_finding"
    version = "0.1.0"

    def evaluate(self, case, ai_output):
        result = self._empty_result(case)
        result.findings.append(
            self.finding(
                case, FindingType.CRITICAL_OMISSION, "Stub finding.", index=0,
                source_evidence="Source says X.", output_evidence="Output says nothing.",
            )
        )
        return result


class StubConsistencyEvaluator(ConsistencyEvaluator):
    name = "stub_consistency"
    version = "0.1.0"

    def evaluate_repeated(self, case, ai_outputs):
        result = self._empty_result(case)
        agree = len({output.summary_text for output in ai_outputs}) == 1
        result.details["agreement"] = 1.0 if agree else 0.0
        return result


class TestBaseEvaluator:
    def test_abstract_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseEvaluator()

    def test_evaluator_result_is_versioned_and_identified(self, case, ai_output):
        result = StubEvaluator().evaluate(case, ai_output)
        assert result.evaluator_name == "stub"
        assert result.evaluator_version == "0.1.0"
        assert result.case_id == case.case_id
        assert result.succeeded

    def test_evaluator_result_reports_operational_errors(self, case):
        result = EvaluatorResult(
            evaluator_name="x", evaluator_version="1", case_id=case.case_id, error="boom"
        )
        assert not result.succeeded

    def test_findings_are_stamped_with_the_evaluator(self, case, ai_output):
        finding = StubFindingEvaluator().evaluate(case, ai_output).findings[0]
        assert finding.evaluator_name == "stub_finding"
        assert finding.evaluator_version == "0.1.0"
        assert finding.case_id == case.case_id

    def test_finding_ids_are_deterministic(self, case, ai_output):
        first = StubFindingEvaluator().evaluate(case, ai_output).findings[0]
        second = StubFindingEvaluator().evaluate(case, ai_output).findings[0]
        assert first.finding_id == second.finding_id == "CASE_TEST-stub_finding-000"

    def test_findings_carry_both_sides_of_the_evidence(self, case, ai_output):
        finding = StubFindingEvaluator().evaluate(case, ai_output).findings[0]
        assert finding.source_evidence and finding.output_evidence

    def test_evaluator_accepts_a_shared_matcher(self, case, ai_output):
        matcher = FactMatcher()
        assert StubEvaluator(matcher=matcher).matcher is matcher

    def test_default_matcher_is_created(self):
        assert isinstance(StubEvaluator().matcher, FactMatcher)


class TestOutputGuard:
    @pytest.mark.parametrize("text", ["", "   ", "\n\t "])
    def test_empty_output_is_flagged_once(self, case, text):
        output = AIOutput(case_id=case.case_id, model_id="m", summary_text=text)
        result = StubEvaluator().guard_output(case, output)
        assert result is not None
        assert [f.finding_type for f in result.findings] == [FindingType.EMPTY_OUTPUT]
        assert result.details["empty_output"] is True

    def test_non_empty_output_passes_through(self, case, ai_output):
        assert StubEvaluator().guard_output(case, ai_output) is None


class TestConsistencyEvaluator:
    def test_single_output_evaluate_is_rejected(self, case, ai_output):
        with pytest.raises(NotImplementedError, match="multiple runs"):
            StubConsistencyEvaluator().evaluate(case, ai_output)

    def test_identical_repeated_runs_agree(self, case):
        outputs = [
            AIOutput(case_id=case.case_id, model_id="m", summary_text="same", run_index=i)
            for i in range(3)
        ]
        result = StubConsistencyEvaluator().evaluate_repeated(case, outputs)
        assert result.details["agreement"] == 1.0

    def test_divergent_repeated_runs_disagree(self, case):
        outputs = [
            AIOutput(case_id=case.case_id, model_id="m", summary_text=f"different {i}", run_index=i)
            for i in range(3)
        ]
        result = StubConsistencyEvaluator().evaluate_repeated(case, outputs)
        assert result.details["agreement"] == 0.0

    def test_consistency_evaluators_declare_their_shape(self):
        assert StubConsistencyEvaluator.requires_multiple_runs is True


class TestEvaluatorKinds:
    def test_deterministic_and_model_based_are_distinguishable(self):
        """The runner and result schema separate rule-based findings from
        model-based judgments before any LLM judge exists."""
        assert DeterministicEvaluator.is_model_based is False
        assert LLMEvaluator.is_model_based is True

    def test_no_llm_evaluator_is_implemented_yet(self):
        """Phase 3 is the deterministic foundation; LLM adjudication is Phase 4."""
        from app.evaluation import registry

        registry.register_defaults()
        for name in registry.available():
            assert registry.create(name).is_model_based is False

    def test_evaluation_version_is_declared(self):
        assert EVALUATION_VERSION
