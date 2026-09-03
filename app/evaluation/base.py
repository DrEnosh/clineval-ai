"""Evaluation engine interfaces.

This module defines the *contract* every evaluation dimension must implement.
No scoring or NLP logic lives here yet - concrete implementations (fact
extraction, negation detection, temporal comparison, etc.) land in later
phases (see docs/ARCHITECTURE.md, Phase 4/5). Defining the interfaces first
lets the benchmark, database, and API layers be built and tested against a
stable contract before any single evaluator exists.

Design notes:
- One evaluator = one clinically meaningful dimension (factuality, hallucination,
  negation, temporal, medication, consistency, ...). Each is independently
  testable and independently swappable, per the "modular architecture"
  principle.
- `evaluate()` never raises for an evaluation *disagreement* (e.g. a
  hallucination found) - that's a normal result. It raises only for
  operational failures (bad input shape), which callers should handle
  gracefully and surface as a clear error rather than a silently wrong score.
- An evaluator returns evidence-backed `Failure` records, not a bare number,
  in keeping with the project's "explain failures, not merely produce
  scores" philosophy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from app.models.schemas import AIOutput, BenchmarkCase, Failure


class EvaluationDimensionResult(BaseModel):
    """The output of a single evaluator running against a single case.

    `score` is dimension-local (0.0-1.0, higher is better) and is combined
    across dimensions later by the scoring layer (Phase 6) - it is not itself
    a clinical safety judgment.
    """

    dimension: str
    evaluator_version: str
    score: float = Field(ge=0.0, le=1.0)
    failures: list[Failure] = Field(default_factory=list)
    details: dict = Field(default_factory=dict)


class BaseEvaluator(ABC):
    """Base class for all evaluation dimensions.

    Subclasses must set `name` and `version` and implement `evaluate`.
    `version` is recorded on every result so an evaluation run can be
    reproduced or audited later even after the evaluator's logic changes.
    """

    name: str
    version: str

    @abstractmethod
    def evaluate(
        self, case: BenchmarkCase, ai_output: AIOutput
    ) -> EvaluationDimensionResult:
        """Evaluate one AI output for one benchmark case.

        Implementations must not use `ai_output` to redefine ground truth:
        gold-standard facts always come from `case`, never from the output
        being evaluated.
        """
        raise NotImplementedError


class FactualityEvaluator(BaseEvaluator):
    """Classifies each gold-standard fact as retained / incorrectly
    represented / missing / contradicted / newly introduced when compared
    against the AI summary."""


class CriticalRetentionEvaluator(BaseEvaluator):
    """Computes critical-fact retention: how many of `case.critical_facts`
    are represented, omitted, or contradicted in the AI summary."""


class HallucinationEvaluator(BaseEvaluator):
    """Flags content in the AI summary unsupported by the source note or
    reference facts, split into unsupported additions vs. explicit
    contradictions."""


class NegationEvaluator(BaseEvaluator):
    """Detects negation reversals: a source fact with polarity NEGATED that
    the AI summary asserts as affirmed (or vice versa)."""


class TemporalConsistencyEvaluator(BaseEvaluator):
    """Checks whether the AI summary preserves the direction and values of
    `case.temporal_facts` (e.g. does not confuse a historical value with the
    current one, does not reverse a trend)."""


class MedicationConsistencyEvaluator(BaseEvaluator):
    """Compares `case.medication_facts` against the AI summary for invented,
    omitted, wrong-drug, wrong-dose, or wrong-status medication errors."""


class ConsistencyEvaluator(BaseEvaluator):
    """Evaluates agreement across multiple `AIOutput` runs for the same case
    (same model, repeated generations), for models with stochastic output.
    """

    @abstractmethod
    def evaluate_repeated(
        self, case: BenchmarkCase, ai_outputs: list[AIOutput]
    ) -> EvaluationDimensionResult:
        """Evaluate consistency across `ai_outputs` (>= 2 runs of the same
        case/model). `evaluate()` (single-output) is not meaningful for this
        dimension; callers should use this method instead.
        """
        raise NotImplementedError

    def evaluate(
        self, case: BenchmarkCase, ai_output: AIOutput
    ) -> EvaluationDimensionResult:
        raise NotImplementedError(
            "ConsistencyEvaluator requires multiple runs; use evaluate_repeated()."
        )


class EvaluationPipeline:
    """Runs a configured set of evaluators over a case and collects results.

    Deliberately thin: it owns no scoring logic (that's the scoring layer,
    Phase 6) and no evaluator implementations of its own. It exists so the
    API/CLI layers have one stable place to say "run these evaluators over
    this case" without depending on the evaluator list directly.
    """

    def __init__(self, evaluators: list[BaseEvaluator]):
        self._evaluators = evaluators

    @property
    def evaluators(self) -> list[BaseEvaluator]:
        return list(self._evaluators)

    def run(
        self, case: BenchmarkCase, ai_output: AIOutput
    ) -> list[EvaluationDimensionResult]:
        results = []
        for evaluator in self._evaluators:
            if isinstance(evaluator, ConsistencyEvaluator):
                continue
            results.append(evaluator.evaluate(case, ai_output))
        return results
