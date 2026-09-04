"""Negation consistency.

Evaluates whether documented negatives survived summarization with their
polarity intact.

Four outcomes, deliberately distinguished:

- **preserved** - the negative is stated, still negated.
- **omitted** - the negative is absent. Information is lost, but no false
  assertion is created, so this is not a reversal.
- **reversed** - the source denies it, the output asserts it. This is a
  *contradiction*, not an omission: "no history of diabetes" becoming
  "patient has diabetes" fabricates a diagnosis.
- **unsupported positive assertion** - the output affirms a finding the case
  never records at all.

Collapsing omission and reversal into one number would hide the difference
between a summary that said less and a summary that said something false.
"""

from __future__ import annotations

from app.core.enums import FactImportance, FindingType, SupportStatus
from app.evaluation import metrics
from app.evaluation.base import DeterministicEvaluator, EvaluatorResult
from app.evaluation.support import analyze_entity
from app.models.schemas import AIOutput, BenchmarkCase


class NegationConsistencyEvaluator(DeterministicEvaluator):
    """Checks that negated facts stay negated."""

    name = "negation_consistency"
    version = "0.3.0"

    def evaluate(self, case: BenchmarkCase, ai_output: AIOutput) -> EvaluatorResult:
        negations = case.negations

        empty = self.guard_output(case, ai_output)
        if empty is not None:
            # An empty summary asserts nothing, so it reverses nothing. Every
            # negation is omitted, and consistency is vacuously preserved -
            # reported honestly rather than scored as failure.
            empty.metrics = {
                "negation_consistency": metrics.compute(
                    "negation_consistency", len(negations), len(negations)
                ),
                "negation_reversal_rate": metrics.compute(
                    "negation_reversal_rate", 0, len(negations)
                ),
            }
            return empty

        result = self._empty_result(case)
        structured = case.structured_case
        index = structured.entity_index() if structured else {}
        output = ai_output.summary_text

        preserved = reversed_count = omitted = 0
        counter = 0

        for fact in negations:
            entity = index.get(fact.source_ref or "")
            if entity is None:
                result.findings.append(
                    self.finding(
                        case, FindingType.MISSING_ANNOTATION,
                        f"Negation {fact.fact_id} could not be traced to a structured entity.",
                        index=counter, fact_id=fact.fact_id, source_ref=fact.source_ref,
                        source_evidence=fact.content,
                    )
                )
                counter += 1
                continue

            support = analyze_entity(entity, output, self.matcher, structured)
            if support is None:
                continue

            shared = {
                "fact_id": fact.fact_id,
                "source_ref": fact.source_ref,
                "source_evidence": support.source_evidence or fact.content,
                "output_evidence": support.output_evidence,
                "match_level": support.match_level,
                "confidence": support.confidence,
            }
            importance = fact.importance

            if support.status is SupportStatus.CONTRADICTED:
                reversed_count += 1
                result.findings.append(
                    self.finding(
                        case, FindingType.NEGATION_REVERSAL,
                        f"Negation reversed: {support.reason}.",
                        index=counter,
                        details={
                            "criticality_rule": fact.criticality_rule,
                            "fact_importance": importance.value,
                            "entity_kind": support.entity_kind,
                        },
                        **shared,
                    )
                )
                counter += 1
                continue

            if support.status is SupportStatus.OMITTED:
                omitted += 1
                preserved += 1  # No false assertion was created.
                if importance is FactImportance.CRITICAL:
                    result.findings.append(
                        self.finding(
                            case, FindingType.NEGATION_OMISSION,
                            f"A clinically important negative is absent from the summary: "
                            f"'{fact.content}'.",
                            index=counter,
                            details={
                                "criticality_rule": fact.criticality_rule,
                                "fact_importance": importance.value,
                            },
                            **shared,
                        )
                    )
                    counter += 1
                continue

            preserved += 1

        total = len(negations)
        result.metrics = {
            "negation_consistency": metrics.compute("negation_consistency", preserved, total),
            "negation_reversal_rate": metrics.compute("negation_reversal_rate", reversed_count, total),
        }
        result.details = {
            "negations": total,
            "preserved": preserved,
            "reversed": reversed_count,
            "omitted": omitted,
        }
        return result
