"""Temporal consistency.

Evaluates whether trajectories survived summarization: both endpoints of a
trend, the direction of change, and the current-versus-historical distinction.

Three distinct failures, kept separate because they mean different things
clinically:

- **information loss** - the trend is flattened to its current value. Context
  is gone; nothing false is asserted.
- **value confusion** - the *historical* value is reported without the current
  one. This actively misinforms: a receiving clinician reads a stale
  creatinine as today's.
- **trend reversal** - the direction is stated backwards. A rising creatinine
  described as falling inverts the clinical picture.

Also covers historical-versus-current confusion for conditions and
medications, and reporting a superseded statement from a resolved conflict as
though it were current.
"""

from __future__ import annotations

from app.core.enums import (
    ConditionStatus,
    FactImportance,
    FindingType,
    MedicationStatus,
    SupportStatus,
)
from app.evaluation import metrics
from app.evaluation.base import DeterministicEvaluator, EvaluatorResult
from app.evaluation.support import (
    analyze_condition,
    analyze_conflict,
    analyze_medication,
    analyze_trend,
)
from app.models.schemas import AIOutput, BenchmarkCase

_ERROR_TO_FINDING = {
    "information_loss": FindingType.TEMPORAL_INFORMATION_LOSS,
    "value_confusion": FindingType.TEMPORAL_VALUE_CONFUSION,
    "trend_reversal": FindingType.TEMPORAL_TREND_REVERSAL,
}


class TemporalConsistencyEvaluator(DeterministicEvaluator):
    """Checks trends, historical/current framing, and conflict resolution."""

    name = "temporal_consistency"
    version = "0.5.0"

    def evaluate(self, case: BenchmarkCase, ai_output: AIOutput) -> EvaluatorResult:
        structured = case.structured_case
        trends = structured.lab_trends if structured else []

        empty = self.guard_output(case, ai_output)
        if empty is not None:
            empty.metrics = {
                "temporal_consistency": metrics.compute("temporal_consistency", 0, len(trends)),
                "temporal_information_loss_rate": metrics.compute(
                    "temporal_information_loss_rate", len(trends), len(trends)
                ),
            }
            return empty

        result = self._empty_result(case)
        output = ai_output.summary_text
        counter = 0

        consistent = information_loss = 0

        for trend in trends:
            support = analyze_trend(trend, output, self.matcher, structured)
            shared = {
                "source_ref": trend.entity_id,
                "source_evidence": support.source_evidence,
                "output_evidence": support.output_evidence,
                "match_level": support.match_level,
                "confidence": support.confidence,
            }

            if support.status is SupportStatus.SUPPORTED:
                consistent += 1
                continue

            error = support.details.get("error")
            if support.status is SupportStatus.OMITTED:
                finding_type = FindingType.TEMPORAL_INFORMATION_LOSS
                information_loss += 1
            else:
                finding_type = _ERROR_TO_FINDING.get(
                    str(error), FindingType.TEMPORAL_INFORMATION_LOSS
                )
                if finding_type is FindingType.TEMPORAL_INFORMATION_LOSS:
                    information_loss += 1

            importance = (
                FactImportance.CRITICAL if trend.clinically_significant
                else FactImportance.NON_CRITICAL
            )
            result.findings.append(
                self.finding(
                    case, finding_type, support.reason.capitalize() + ".",
                    index=counter,
                    details={
                        "analyte": trend.lab_name,
                        "expected_trend": trend.trend.value,
                        "clinically_significant": trend.clinically_significant,
                        "fact_importance": importance.value,
                    },
                    **shared,
                )
            )
            counter += 1

        counter = self._check_historical_framing(case, structured, output, result, counter)
        self._check_conflicts(case, structured, output, result, counter)

        total = len(trends)
        result.metrics = {
            "temporal_consistency": metrics.compute("temporal_consistency", consistent, total),
            "temporal_information_loss_rate": metrics.compute(
                "temporal_information_loss_rate", information_loss, total
            ),
        }
        result.details = {
            "trends": total,
            "consistent": consistent,
            "information_loss": information_loss,
        }
        return result

    def _check_historical_framing(
        self, case: BenchmarkCase, structured, output: str, result: EvaluatorResult, counter: int
    ) -> int:
        """Flag resolved conditions and past medications presented as current."""
        if structured is None:
            return counter

        for condition in structured.conditions:
            if condition.status is ConditionStatus.ACTIVE:
                continue
            support = analyze_condition(condition, output, self.matcher, structured)
            if support.status is SupportStatus.CONTRADICTED:
                result.findings.append(
                    self.finding(
                        case, FindingType.HISTORICAL_CURRENT_CONFUSION,
                        support.reason.capitalize() + ".",
                        index=counter, source_ref=condition.entity_id,
                        source_evidence=support.source_evidence,
                        output_evidence=support.output_evidence,
                        match_level=support.match_level, confidence=support.confidence,
                        details={"entity_kind": "condition", "status": condition.status.value},
                    )
                )
                counter += 1

        for medication in structured.medications:
            if medication.status is not MedicationStatus.HISTORICAL:
                continue
            support = analyze_medication(medication, output, self.matcher, structured)
            if support.status is SupportStatus.CONTRADICTED:
                result.findings.append(
                    self.finding(
                        case, FindingType.HISTORICAL_CURRENT_CONFUSION,
                        support.reason.capitalize() + ".",
                        index=counter, source_ref=medication.entity_id,
                        source_evidence=support.source_evidence,
                        output_evidence=support.output_evidence,
                        match_level=support.match_level, confidence=support.confidence,
                        details={"entity_kind": "medication", "drug": medication.drug_name},
                    )
                )
                counter += 1
        return counter

    def _check_conflicts(
        self, case: BenchmarkCase, structured, output: str, result: EvaluatorResult, counter: int
    ) -> int:
        """Flag a superseded statement reported as the current one."""
        if structured is None:
            return counter

        for conflict in structured.conflicts:
            support = analyze_conflict(conflict, output, self.matcher, structured)
            if support.status is not SupportStatus.CONTRADICTED:
                continue
            result.findings.append(
                self.finding(
                    case, FindingType.CONFLICT_RESOLUTION_ERROR,
                    support.reason.capitalize() + ".",
                    index=counter, source_ref=conflict.entity_id,
                    source_evidence=support.source_evidence,
                    output_evidence=support.output_evidence,
                    match_level=support.match_level, confidence=support.confidence,
                    details={
                        "topic": conflict.topic,
                        "fact_importance": FactImportance.CRITICAL.value,
                    },
                )
            )
            counter += 1
        return counter
