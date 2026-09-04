"""Evaluation engine interfaces.

Defines the contract every evaluator implements. Phase 1 established these as
stubs; Phase 3 fills them in for the deterministic evaluators and keeps the
LLM-based ones abstract.

Design rules that hold across every evaluator:

- **One evaluator, one clinically meaningful dimension.** Each is
  independently testable and independently swappable.
- **Ground truth comes from the case, never from the output.** An evaluator
  reads `case.structured_case` for truth and `ai_output.summary_text` for the
  claim under test. It never lets the output define what is correct.
- **Detect, do not sentence.** Evaluators emit findings with
  `severity=None`; the severity policy classifies them afterwards. This keeps
  the most contestable judgment in the system in one configurable place.
- **Never crash on unexpected text.** A model can emit anything - empty
  strings, markup, prose in another language. An evaluator returns findings;
  it does not raise. `EvaluatorResult.error` carries operational failures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from app.core.enums import FindingType
from app.evaluation.matching import FactMatcher
from app.models.schemas import AIOutput, BenchmarkCase, EvaluationFinding, MetricValue

EVALUATION_VERSION = "0.5.0"


class EvaluatorResult(BaseModel):
    """One evaluator's output for one case."""

    evaluator_name: str
    evaluator_version: str
    case_id: str
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    findings: list[EvaluationFinding] = Field(default_factory=list)
    details: dict = Field(default_factory=dict)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class BaseEvaluator(ABC):
    """Base class for all evaluation dimensions.

    Subclasses set `name` and `version` and implement `evaluate`. `version` is
    recorded on every finding, so a result stays interpretable after the
    evaluator's logic changes.
    """

    name: str = "base"
    version: str = EVALUATION_VERSION

    def __init__(self, matcher: FactMatcher | None = None):
        self.matcher = matcher or FactMatcher()

    @abstractmethod
    def evaluate(self, case: BenchmarkCase, ai_output: AIOutput) -> EvaluatorResult:
        """Evaluate one AI output against one benchmark case."""
        raise NotImplementedError

    # -- helpers shared by concrete evaluators ------------------------------

    def _empty_result(self, case: BenchmarkCase, error: str | None = None) -> EvaluatorResult:
        return EvaluatorResult(
            evaluator_name=self.name, evaluator_version=self.version,
            case_id=case.case_id, error=error,
        )

    def finding(
        self,
        case: BenchmarkCase,
        finding_type: FindingType,
        explanation: str,
        index: int,
        **kwargs,
    ) -> EvaluationFinding:
        """Build a finding stamped with this evaluator's identity.

        Finding ids are deterministic - `<case>-<evaluator>-<index>` - so the
        same evaluation run produces the same ids twice, which regression
        tests and stored results both depend on.
        """
        return EvaluationFinding(
            finding_id=f"{case.case_id}-{self.name}-{index:03d}",
            case_id=case.case_id,
            evaluator_name=self.name,
            evaluator_version=self.version,
            finding_type=finding_type,
            explanation=explanation,
            **kwargs,
        )

    def guard_output(
        self, case: BenchmarkCase, ai_output: AIOutput
    ) -> EvaluatorResult | None:
        """Handle empty or whitespace-only output uniformly.

        Returns a result carrying an EMPTY_OUTPUT finding when there is
        nothing to evaluate, or None when evaluation should proceed. Every
        evaluator calls this first so that an empty summary produces one clear
        finding per evaluator rather than a flood of spurious omissions.
        """
        if ai_output.summary_text and ai_output.summary_text.strip():
            return None

        result = self._empty_result(case)
        result.findings.append(
            self.finding(
                case, FindingType.EMPTY_OUTPUT,
                "The model produced no summary text, so nothing could be evaluated.",
                index=0, confidence=1.0,
            )
        )
        result.details["empty_output"] = True
        return result


class LLMEvaluator(BaseEvaluator):
    """Marker base for model-based evaluators.

    Deliberately unimplemented in Phase 3. It exists so the registry, runner,
    and result schema already distinguish deterministic findings from
    model-based judgments before any exist, and so adding an LLM judge in
    Phase 4 requires no change to those layers. A model-based judgment must
    never become the unquestioned source of truth.
    """

    is_model_based = True


class DeterministicEvaluator(BaseEvaluator):
    """Marker base for rule-based evaluators.

    Every Phase 3 evaluator is one of these: same input, same findings, every
    time, with no network call and no model.
    """

    is_model_based = False


class ConsistencyEvaluator(BaseEvaluator):
    """Base for evaluators that compare several runs of the same case.

    Repeatability cannot be judged from one generation, so this dimension
    takes a list of outputs. `evaluate` is deliberately unusable: calling the
    single-output method on a multi-run dimension is a programming error, and
    raising says so rather than silently returning a meaningless result.

    No concrete implementation exists yet - the models under evaluation arrive
    in Phase 4, and repeatability is only meaningful once a stochastic model
    is actually generating. The contract is defined now so the registry and
    runner already distinguish the two shapes.
    """

    is_model_based = False
    requires_multiple_runs = True

    @abstractmethod
    def evaluate_repeated(
        self, case: BenchmarkCase, ai_outputs: list[AIOutput]
    ) -> EvaluatorResult:
        """Evaluate agreement across repeated generations of one case."""
        raise NotImplementedError

    def evaluate(self, case: BenchmarkCase, ai_output: AIOutput) -> EvaluatorResult:
        raise NotImplementedError(
            f"{type(self).__name__} evaluates repeatability and needs multiple runs; "
            f"call evaluate_repeated() with a list of outputs."
        )
