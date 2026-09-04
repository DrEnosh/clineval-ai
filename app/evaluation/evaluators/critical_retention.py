"""Critical information retention.

For every fact the benchmark marked critical, decide whether the output
retained, omitted, or contradicted it - and say which fact, by id.

Works from `fact.source_ref` back to the structured entity, so retention is
judged against structured clinical content rather than against the annotation's
content string. A fact whose `source_ref` does not resolve is reported as a
MISSING_ANNOTATION finding rather than silently skipped: a benchmark defect
should be visible in the results, not absorbed by them.
"""

from __future__ import annotations

from app.core.enums import FactImportance, FindingType, SupportStatus
from app.evaluation import metrics
from app.evaluation.base import DeterministicEvaluator, EvaluatorResult
from app.evaluation.support import analyze_entity
from app.models.schemas import AIOutput, BenchmarkCase


class CriticalRetentionEvaluator(DeterministicEvaluator):
    """Measures retention of critical facts, with per-fact evidence."""

    name = "critical_retention"
    version = "0.3.0"

    def evaluate(self, case: BenchmarkCase, ai_output: AIOutput) -> EvaluatorResult:
        empty = self.guard_output(case, ai_output)
        critical_facts = case.critical_facts

        if empty is not None:
            # Every critical fact is lost, and the metrics must say so rather
            # than being left undefined - an empty summary retained nothing.
            empty.metrics = {
                "critical_information_retention": metrics.compute(
                    "critical_information_retention", 0, len(critical_facts)
                ),
                "critical_omission_rate": metrics.compute(
                    "critical_omission_rate", len(critical_facts), len(critical_facts)
                ),
                "critical_contradiction_rate": metrics.compute(
                    "critical_contradiction_rate", 0, len(critical_facts)
                ),
            }
            return empty

        result = self._empty_result(case)
        structured = case.structured_case
        index = structured.entity_index() if structured else {}
        output = ai_output.summary_text

        retained = omitted = contradicted = 0
        counter = 0

        for fact in critical_facts:
            entity = index.get(fact.source_ref or "")
            if entity is None:
                result.findings.append(
                    self.finding(
                        case, FindingType.MISSING_ANNOTATION,
                        f"Critical fact {fact.fact_id} could not be traced to a structured "
                        f"entity, so retention could not be assessed for it.",
                        index=counter, fact_id=fact.fact_id, source_ref=fact.source_ref,
                        source_evidence=fact.content, confidence=1.0,
                    )
                )
                counter += 1
                continue

            support = analyze_entity(entity, output, self.matcher, structured)
            if support is None:
                result.findings.append(
                    self.finding(
                        case, FindingType.MISSING_ANNOTATION,
                        f"No analyzer exists for entity type {type(entity).__name__}, so "
                        f"critical fact {fact.fact_id} was not assessed.",
                        index=counter, fact_id=fact.fact_id, source_ref=fact.source_ref,
                        source_evidence=fact.content,
                    )
                )
                counter += 1
                continue

            if support.status in {SupportStatus.SUPPORTED, SupportStatus.PARTIAL}:
                retained += 1
                continue

            if support.status is SupportStatus.CONTRADICTED:
                contradicted += 1
                finding_type = FindingType.CONTRADICTION
            else:
                omitted += 1
                finding_type = FindingType.CRITICAL_OMISSION

            result.findings.append(
                self.finding(
                    case, finding_type,
                    f"Critical fact not retained: {support.reason}.",
                    index=counter,
                    fact_id=fact.fact_id,
                    source_ref=fact.source_ref,
                    source_evidence=support.source_evidence or fact.content,
                    output_evidence=support.output_evidence,
                    match_level=support.match_level,
                    confidence=support.confidence,
                    details={
                        "criticality_rule": fact.criticality_rule,
                        "entity_kind": support.entity_kind,
                        "support_status": support.status.value,
                        "fact_importance": FactImportance.CRITICAL.value,
                    },
                )
            )
            counter += 1

        total = len(critical_facts)
        result.metrics = {
            "critical_information_retention": metrics.compute(
                "critical_information_retention", retained, total
            ),
            "critical_omission_rate": metrics.compute("critical_omission_rate", omitted, total),
            "critical_contradiction_rate": metrics.compute(
                "critical_contradiction_rate", contradicted, total
            ),
        }
        result.details = {
            "critical_facts": total,
            "retained": retained,
            "omitted": omitted,
            "contradicted": contradicted,
        }
        return result
