"""Renders a reference summary from a `StructuredCase`.

This renderer never reads the note. It selects content directly from the
structured case using explicit inclusion rules, then writes it in its own
phrasing. That independence is deliberate: if the summary were derived from
the note, the benchmark would be testing paraphrase rather than clinical
information retention, and a summarizer could score well by copying sentences.

Inclusion rules (documented in docs/BENCHMARK.md):

1. Always include: the primary diagnosis, every abnormal laboratory value,
   every clinically significant trend, every medication change, every severe
   allergy, and every conflict resolution.
2. Include, space permitting: active comorbidities, important relevant
   negatives, performed procedures with outcomes, and the follow-up plan.
3. Never include: routine normal findings, boilerplate review of systems,
   normal laboratory values that carry no trend, and anything not present in
   the structured case.
4. Always distinguish current from historical - a resolved condition or a
   discontinued drug is stated as such, never listed as current.
"""

from __future__ import annotations

from app.core.enums import (
    AbnormalFlag,
    AllergySeverity,
    ConditionStatus,
    FindingSignificance,
    MedicationStatus,
    TemporalTrend,
)
from app.models.structured import StructuredCase

_TREND_VERBS: dict[TemporalTrend, str] = {
    TemporalTrend.INCREASING: "rose",
    TemporalTrend.DECREASING: "fell",
    TemporalTrend.STABLE: "remained stable",
    TemporalTrend.UNKNOWN: "changed",
}


def _opening(case: StructuredCase) -> str:
    patient = case.patient
    primary = next((c for c in case.conditions if c.is_primary), None)
    active_others = [
        c for c in case.conditions
        if not c.is_primary and c.status is ConditionStatus.ACTIVE
    ]

    sentence = (
        f"{patient.age}-year-old {patient.sex} seen for "
        f"{case.encounter.encounter_type.lower()}"
    )
    if primary is not None:
        sentence += f" with {primary.name.lower()}"
    if active_others:
        names = ", ".join(c.name.lower() for c in active_others)
        sentence += f", alongside {names}"
    return sentence + "."


def _laboratory_sentences(case: StructuredCase) -> list[str]:
    sentences: list[str] = []

    trended_labs = {t.lab_name for t in case.lab_trends}
    for trend in case.lab_trends:
        if not trend.clinically_significant:
            continue
        verb = _TREND_VERBS[trend.trend]
        sentences.append(
            f"{trend.lab_name} {verb} from {trend.format_value(trend.earlier_value)} to "
            f"{trend.format_value(trend.later_value)} {trend.unit} between "
            f"{trend.earlier_timing} and {trend.later_timing}."
        )

    # Abnormal values whose specific value a trend sentence did not already
    # state. Suppressing by analyte *name* would drop a measurement that no
    # trend covers - a case with both a standalone creatinine and a creatinine
    # series would silently lose the standalone value from the reference
    # summary. Caught by evaluating the reference summary against itself.
    trended_values = {
        (trend.lab_name, value)
        for trend in case.lab_trends
        if trend.clinically_significant
        for value in (trend.earlier_value, trend.later_value)
    }
    standalone = [
        lab for lab in case.labs
        if lab.is_abnormal and (lab.name, lab.value) not in trended_values
    ]
    for lab in standalone:
        qualifier = "critically abnormal" if lab.is_critical_value else "abnormal"
        direction = "elevated" if lab.abnormal_flag in {AbnormalFlag.HIGH, AbnormalFlag.CRITICAL_HIGH} else "low"
        sentences.append(
            f"{lab.name} is {qualifier} at {lab.format_value()} {lab.unit} "
            f"({direction}, {lab.timing_description})."
        )
    return sentences


def _vitals_sentence(case: StructuredCase) -> str | None:
    abnormal = [v for v in case.vitals if v.is_abnormal]
    if not abnormal:
        return None
    parts = [f"{v.name.lower()} {v.value}{' ' + v.unit if v.unit else ''}" for v in abnormal]
    return "Abnormal vital signs: " + ", ".join(parts) + "."


def _medication_sentences(case: StructuredCase) -> list[str]:
    sentences: list[str] = []

    for med in case.medications:
        if med.status is MedicationStatus.DISCONTINUED:
            timing = f" {med.change_timing}" if med.change_timing else ""
            reason = f" ({med.change_reason})" if med.change_reason else ""
            sentences.append(f"{med.drug_name} was discontinued{timing}{reason}.")
        elif med.status is MedicationStatus.CHANGED_DOSE and med.prior_dose:
            timing = f" {med.change_timing}" if med.change_timing else ""
            sentences.append(
                f"{med.drug_name} was increased from {med.prior_dose} to {med.dose}{timing}."
            )
        elif med.status is MedicationStatus.NEW:
            sentences.append(f"{med.drug_name} {med.dose} was newly started.")
        elif med.status is MedicationStatus.HISTORICAL:
            sentences.append(
                f"{med.drug_name} was taken in the past and is not a current medication."
            )

    continued = [m for m in case.medications if m.status is MedicationStatus.CURRENT]
    if continued:
        names = ", ".join(f"{m.drug_name} {m.dose}" for m in continued if m.dose)
        if names:
            sentences.append(f"Continued on {names}.")
    return sentences


def _allergy_sentence(case: StructuredCase) -> str | None:
    if not case.allergies:
        return None
    severe = [a for a in case.allergies if a.severity is AllergySeverity.SEVERE]
    listed = severe or case.allergies
    parts = [f"{a.substance} ({a.reaction})" for a in listed]
    prefix = "Severe allergy" if severe else "Allergy"
    if len(parts) > 1:
        prefix = "Severe allergies" if severe else "Allergies"
    return f"{prefix}: {', '.join(parts)}."


def _negatives_sentence(case: StructuredCase) -> str | None:
    important = [
        f for f in case.findings
        if not f.present and f.significance in {FindingSignificance.CRITICAL, FindingSignificance.IMPORTANT}
    ]
    if not important:
        return None
    return "Documented negatives: " + "; ".join(f.description for f in important) + "."


def _historical_sentence(case: StructuredCase) -> str | None:
    resolved = [c for c in case.conditions if c.status is not ConditionStatus.ACTIVE]
    if not resolved:
        return None
    parts = [f"{c.name.lower()} ({c.onset_description or 'not active'})" for c in resolved]
    return "Historical, not active: " + ", ".join(parts) + "."


def _conflict_sentences(case: StructuredCase) -> list[str]:
    return [
        f"Regarding {conflict.topic}: {conflict.later_statement}. {conflict.resolution}"
        for conflict in case.conflicts
    ]


def _procedure_sentence(case: StructuredCase) -> str | None:
    performed = [p for p in case.procedures if p.performed]
    if not performed:
        return None
    parts = [f"{p.name}{' - ' + p.outcome if p.outcome else ''}" for p in performed]
    return "Studies: " + "; ".join(parts) + "."


def _not_performed_sentence(case: StructuredCase) -> str | None:
    """Explicitly carrying "not done" forward is what stops a downstream reader
    inferring a result was obtained."""
    not_performed = [p for p in case.procedures if not p.performed]
    if not not_performed:
        return None
    parts = [p.name for p in not_performed]
    return "Not performed at this visit: " + ", ".join(parts) + "."


def _plan_sentence(case: StructuredCase) -> str | None:
    if not case.follow_up:
        return None
    plan = case.follow_up[0]
    actions = ", ".join(plan.actions) if plan.actions else "clinical reassessment"
    return f"Plan: {actions}; follow up with {plan.provider} in {plan.interval}."


def render_reference_summary(case: StructuredCase) -> str:
    """Render a concise reference summary directly from the structured case."""
    sentences: list[str] = [_opening(case)]

    sentences.extend(_conflict_sentences(case))
    sentences.extend(_laboratory_sentences(case))

    vitals = _vitals_sentence(case)
    if vitals:
        sentences.append(vitals)

    sentences.extend(_medication_sentences(case))

    for optional in (
        _allergy_sentence(case),
        _historical_sentence(case),
        _negatives_sentence(case),
        _procedure_sentence(case),
        _not_performed_sentence(case),
        _plan_sentence(case),
    ):
        if optional:
            sentences.append(optional)

    return " ".join(sentences)
