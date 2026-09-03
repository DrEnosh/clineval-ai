"""Derives gold-standard annotations from a `StructuredCase`.

The third independent rendering. Like the summary renderer, this reads only
the structured case - never the note, never the summary. Every fact it emits
carries:

- `source_ref`: the `entity_id` it was derived from, so the integrity checker
  can prove the annotation describes real structured content.
- `criticality_rule`: the id of the rule that assigned its importance, so
  "why is this critical?" always has a stated answer.

Because both the note and the annotations derive from the same structured
object, they cannot disagree. That is the property hand-annotation cannot
guarantee and the reason this pipeline exists.
"""

from __future__ import annotations

from app.benchmark import criticality
from app.core.enums import (
    ConditionStatus,
    FactImportance,
    FactPolarity,
    FactType,
    MedicationStatus,
    Severity,
)
from app.models.schemas import ClinicalFact, MedicationFact, RiskFlag, TemporalFact
from app.models.structured import StructuredCase


class AnnotationSet:
    """The complete gold-standard layer derived from one structured case."""

    def __init__(self) -> None:
        self.critical_facts: list[ClinicalFact] = []
        self.non_critical_facts: list[ClinicalFact] = []
        self.medication_facts: list[MedicationFact] = []
        self.temporal_facts: list[TemporalFact] = []
        self.negations: list[ClinicalFact] = []
        self.contradiction_facts: list[ClinicalFact] = []
        self.risk_flags: list[RiskFlag] = []


def _sentence_case(text: str) -> str:
    """Upper-case the first character only.

    `str.capitalize()` would lower-case the rest, turning "no history of GERD"
    into "no history of gerd" and corrupting clinical abbreviations.
    """
    return text[:1].upper() + text[1:] if text else text


class _IdAllocator:
    def __init__(self, case_id: str) -> None:
        self._case_id = case_id
        self._counters: dict[str, int] = {}

    def next(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{self._case_id}-{prefix}{self._counters[prefix]:03d}"


def _place(annotations: AnnotationSet, fact: ClinicalFact) -> None:
    """File a fact into the critical or non-critical bucket by its importance."""
    if fact.importance is FactImportance.CRITICAL:
        annotations.critical_facts.append(fact)
    else:
        annotations.non_critical_facts.append(fact)


def _importance(rule_id: str | None) -> FactImportance:
    return FactImportance.CRITICAL if rule_id else FactImportance.NON_CRITICAL


def annotate(case: StructuredCase) -> AnnotationSet:
    """Derive the full gold-standard annotation layer for `case`."""
    annotations = AnnotationSet()
    ids = _IdAllocator(case.case_id)

    _annotate_diagnoses(case, annotations, ids)
    _annotate_conditions(case, annotations, ids)
    _annotate_labs(case, annotations, ids)
    _annotate_trends(case, annotations, ids)
    _annotate_vitals(case, annotations, ids)
    _annotate_medications(case, annotations, ids)
    _annotate_allergies(case, annotations, ids)
    _annotate_findings(case, annotations, ids)
    _annotate_conflicts(case, annotations, ids)
    _annotate_procedures(case, annotations, ids)
    _annotate_risk_flags(case, annotations, ids)

    return annotations


def _annotate_diagnoses(case: StructuredCase, annotations: AnnotationSet, ids: _IdAllocator) -> None:
    for diagnosis in case.diagnoses:
        rule_id = criticality.classify_diagnosis(diagnosis)
        _place(
            annotations,
            ClinicalFact(
                fact_id=ids.next("F"),
                type=FactType.CONDITION,
                content=(
                    f"{'Primary diagnosis' if diagnosis.is_primary else 'Diagnosis'}: "
                    f"{diagnosis.name}"
                ),
                importance=_importance(rule_id),
                source_ref=diagnosis.entity_id,
                criticality_rule=rule_id,
            ),
        )


def _annotate_conditions(case: StructuredCase, annotations: AnnotationSet, ids: _IdAllocator) -> None:
    """Only non-active conditions are annotated here.

    Active conditions already appear as diagnoses; annotating them twice would
    double-count them in retention metrics. Resolved and historical conditions
    have no diagnosis entry, and the current-vs-historical distinction is
    exactly what a summarizer must preserve, so they get their own facts.
    """
    for condition in case.conditions:
        if condition.status is ConditionStatus.ACTIVE:
            continue
        _place(
            annotations,
            ClinicalFact(
                fact_id=ids.next("F"),
                type=FactType.CONDITION,
                content=(
                    f"{condition.name} is {condition.status.value} "
                    f"({condition.onset_description or 'no longer active'}), not a current problem"
                ),
                importance=FactImportance.NON_CRITICAL,
                source_ref=condition.entity_id,
            ),
        )


def _annotate_labs(case: StructuredCase, annotations: AnnotationSet, ids: _IdAllocator) -> None:
    for lab in case.labs:
        rule_id = criticality.classify_lab(lab)
        _place(
            annotations,
            ClinicalFact(
                fact_id=ids.next("F"),
                type=FactType.LAB,
                content=f"{lab.describe()} ({lab.timing_description})",
                importance=_importance(rule_id),
                source_ref=lab.entity_id,
                criticality_rule=rule_id,
                attributes={
                    "value": lab.format_value(),
                    "unit": lab.unit,
                    "abnormal_flag": lab.abnormal_flag.value,
                    "timing": lab.timing_description,
                },
            ),
        )


def _annotate_trends(case: StructuredCase, annotations: AnnotationSet, ids: _IdAllocator) -> None:
    for trend in case.lab_trends:
        rule_id = criticality.classify_lab_trend(trend)
        annotations.temporal_facts.append(
            TemporalFact(
                fact_id=ids.next("T"),
                content=trend.describe(),
                importance=_importance(rule_id),
                earlier_value=f"{trend.format_value(trend.earlier_value)} {trend.unit}".strip(),
                later_value=f"{trend.format_value(trend.later_value)} {trend.unit}".strip(),
                unit=trend.unit,
                trend=trend.trend,
                source_ref=trend.entity_id,
                criticality_rule=rule_id,
                attributes={
                    "earlier_ref": trend.earlier_ref,
                    "later_ref": trend.later_ref,
                    "earlier_timing": trend.earlier_timing,
                    "later_timing": trend.later_timing,
                },
            )
        )


def _annotate_vitals(case: StructuredCase, annotations: AnnotationSet, ids: _IdAllocator) -> None:
    for vital in case.vitals:
        rule_id = criticality.classify_vital(vital)
        importance = _importance(rule_id)
        # An abnormal-but-not-panic vital is worth recording, but no rule makes
        # it critical; normal vitals stay non-critical too.
        _place(
            annotations,
            ClinicalFact(
                fact_id=ids.next("F"),
                type=FactType.VITAL,
                content=f"{vital.describe()} ({vital.timing_description})",
                importance=importance,
                source_ref=vital.entity_id,
                criticality_rule=rule_id,
                attributes={"abnormal_flag": vital.abnormal_flag.value},
            ),
        )


def _annotate_medications(case: StructuredCase, annotations: AnnotationSet, ids: _IdAllocator) -> None:
    for med in case.medications:
        rule_id = criticality.classify_medication(med)
        content_parts = [med.describe()]
        if med.status is MedicationStatus.DISCONTINUED:
            content_parts.append("- DISCONTINUED")
        elif med.status is MedicationStatus.CHANGED_DOSE and med.prior_dose:
            content_parts.append(f"- dose increased from {med.prior_dose}")
        elif med.status is MedicationStatus.NEW:
            content_parts.append("- newly started")
        elif med.status is MedicationStatus.HISTORICAL:
            content_parts.append("- historical, not current")
        if med.change_timing:
            content_parts.append(f"({med.change_timing})")

        annotations.medication_facts.append(
            MedicationFact(
                fact_id=ids.next("M"),
                content=" ".join(content_parts),
                importance=_importance(rule_id),
                drug_name=med.drug_name,
                status=med.status,
                dose=med.dose,
                route=med.route,
                source_ref=med.entity_id,
                criticality_rule=rule_id,
                attributes={
                    "prior_dose": med.prior_dose or "",
                    "change_reason": med.change_reason or "",
                    "change_timing": med.change_timing or "",
                },
            )
        )


def _annotate_allergies(case: StructuredCase, annotations: AnnotationSet, ids: _IdAllocator) -> None:
    for allergy in case.allergies:
        rule_id = criticality.classify_allergy(allergy)
        _place(
            annotations,
            ClinicalFact(
                fact_id=ids.next("F"),
                type=FactType.ALLERGY,
                content=f"Allergy: {allergy.describe()} ({allergy.severity.value})",
                importance=_importance(rule_id),
                source_ref=allergy.entity_id,
                criticality_rule=rule_id,
            ),
        )


def _annotate_findings(case: StructuredCase, annotations: AnnotationSet, ids: _IdAllocator) -> None:
    for finding in case.findings:
        rule_id = criticality.classify_finding(finding)
        if finding.present:
            _place(
                annotations,
                ClinicalFact(
                    fact_id=ids.next("F"),
                    type=FactType.OBSERVATION,
                    content=_sentence_case(finding.description),
                    importance=_importance(rule_id),
                    source_ref=finding.entity_id,
                    criticality_rule=rule_id,
                ),
            )
            continue

        # Negated findings live in `negations` regardless of importance, and
        # are additionally filed as critical facts when a rule makes them so -
        # a reversed critical negative is the benchmark's highest-severity
        # failure, so it must count in critical retention too.
        negation = ClinicalFact(
            fact_id=ids.next("N"),
            type=FactType.OBSERVATION,
            content=_sentence_case(finding.description),
            importance=_importance(rule_id),
            polarity=FactPolarity.NEGATED,
            source_ref=finding.entity_id,
            criticality_rule=rule_id,
        )
        annotations.negations.append(negation)
        if negation.importance is FactImportance.CRITICAL:
            annotations.critical_facts.append(negation)


def _annotate_conflicts(case: StructuredCase, annotations: AnnotationSet, ids: _IdAllocator) -> None:
    for conflict in case.conflicts:
        rule_id = criticality.classify_conflict(conflict)
        fact = ClinicalFact(
            fact_id=ids.next("C"),
            type=FactType.OTHER,
            content=(
                f"Regarding {conflict.topic}: '{conflict.earlier_statement}' "
                f"({conflict.earlier_timing}) was superseded by '{conflict.later_statement}' "
                f"({conflict.later_timing}). {conflict.resolution}"
            ),
            importance=_importance(rule_id),
            source_ref=conflict.entity_id,
            criticality_rule=rule_id,
            attributes={
                "earlier_statement": conflict.earlier_statement,
                "later_statement": conflict.later_statement,
                "resolution": conflict.resolution,
            },
        )
        annotations.contradiction_facts.append(fact)
        annotations.critical_facts.append(fact)


def _annotate_procedures(case: StructuredCase, annotations: AnnotationSet, ids: _IdAllocator) -> None:
    for procedure in case.procedures:
        if procedure.performed:
            outcome = f": {procedure.outcome}" if procedure.outcome else ""
            _place(
                annotations,
                ClinicalFact(
                    fact_id=ids.next("F"),
                    type=FactType.PROCEDURE,
                    content=f"{procedure.name} performed ({procedure.timing_description}){outcome}",
                    importance=FactImportance.NON_CRITICAL,
                    source_ref=procedure.entity_id,
                ),
            )
        else:
            annotations.negations.append(
                ClinicalFact(
                    fact_id=ids.next("N"),
                    type=FactType.PROCEDURE,
                    content=f"{procedure.name} was not performed at this visit",
                    importance=FactImportance.NON_CRITICAL,
                    polarity=FactPolarity.NEGATED,
                    source_ref=procedure.entity_id,
                )
            )


def _annotate_risk_flags(case: StructuredCase, annotations: AnnotationSet, ids: _IdAllocator) -> None:
    """Promote the case's serious failure opportunities into risk flags.

    Only HIGH and CRITICAL opportunities become flags: a risk flag asserts
    that mishandling this case could matter clinically, and applying that
    label to every minor redundancy would drain it of meaning.
    """
    for opportunity in case.failure_opportunities:
        if opportunity.severity_if_missed not in {Severity.HIGH, Severity.CRITICAL}:
            continue
        annotations.risk_flags.append(
            RiskFlag(
                flag_id=ids.next("R"),
                description=opportunity.description,
                severity=opportunity.severity_if_missed,
            )
        )
