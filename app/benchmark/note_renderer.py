"""Renders a source clinical note from a `StructuredCase`.

This is one of three independent renderings of the structured case (note,
reference summary, annotations). It reads only structured entities and emits
prose; it never consults the summary or the annotations, and nothing it writes
feeds back into the structured case.

The note is deliberately *noisier* than the summary: it carries section
headers, routine findings, redundancy where the case calls for it, and the
full laboratory list including historical values. That noise is the point -
it is what a summarizer has to see through.
"""

from __future__ import annotations

from random import Random

from app.core.enums import (
    AbnormalFlag,
    ConditionStatus,
    FailureModeCategory,
    MedicationStatus,
    TemporalTrend,
    TimelineEventType,
)
from app.models.structured import StructuredCase


def _sentence_case(text: str) -> str:
    """Upper-case the first character only.

    `str.capitalize()` would lower-case the rest, corrupting clinical
    abbreviations ("GERD" -> "gerd").
    """
    return text[:1].upper() + text[1:] if text else text


_TREND_VERBS: dict[TemporalTrend, str] = {
    TemporalTrend.INCREASING: "increased",
    TemporalTrend.DECREASING: "decreased",
    TemporalTrend.STABLE: "remained stable",
    TemporalTrend.UNKNOWN: "changed",
}

_STATUS_PHRASES: dict[MedicationStatus, str] = {
    MedicationStatus.CURRENT: "current",
    MedicationStatus.NEW: "newly started",
    MedicationStatus.DISCONTINUED: "DISCONTINUED",
    MedicationStatus.CHANGED_DOSE: "dose changed",
    MedicationStatus.HISTORICAL: "prior medication, not current",
}

_FLAG_LABELS: dict[AbnormalFlag, str] = {
    AbnormalFlag.LOW: "LOW",
    AbnormalFlag.HIGH: "HIGH",
    AbnormalFlag.CRITICAL_LOW: "CRITICAL LOW",
    AbnormalFlag.CRITICAL_HIGH: "CRITICAL HIGH",
}


def _opening_line(case: StructuredCase, rng: Random) -> str:
    patient = case.patient
    primary = next((c for c in case.conditions if c.is_primary), None)
    lead = rng.choice(
        [
            f"{patient.age}-year-old {patient.sex} presents for {case.encounter.encounter_type.lower()}.",
            f"{patient.age}-year-old {patient.sex} seen today for {case.encounter.encounter_type.lower()}.",
            f"This is a {patient.age}-year-old {patient.sex} presenting for "
            f"{case.encounter.encounter_type.lower()}.",
        ]
    )
    if primary is not None:
        onset = f" ({primary.onset_description})" if primary.onset_description else ""
        lead += f" Known history of {primary.name.lower()}{onset}."
    return lead


def _subjective_section(case: StructuredCase, rng: Random) -> list[str]:
    lines = [_opening_line(case, rng)]

    others = [
        c for c in case.conditions
        if not c.is_primary and c.status is ConditionStatus.ACTIVE
    ]
    if others:
        names = ", ".join(c.name.lower() for c in others)
        lines.append(f"Additional active problems include {names}.")

    resolved = [c for c in case.conditions if c.status is not ConditionStatus.ACTIVE]
    for condition in resolved:
        lines.append(
            f"History of {condition.name.lower()}, {condition.onset_description or 'resolved'}; "
            f"this is not an active problem."
        )

    negatives = [f for f in case.findings if not f.present]
    if negatives:
        lines.append(
            "Pertinent negatives: " + "; ".join(f.description for f in negatives) + "."
        )

    for conflict in case.conflicts:
        lines.append(
            f"On initial assessment, {conflict.earlier_statement}. "
            f"On reassessment later in the encounter, {conflict.later_statement}. "
            f"{conflict.resolution}"
        )

    return lines


def _medication_section(case: StructuredCase) -> list[str]:
    if not case.medications:
        return []
    lines = ["MEDICATIONS:"]
    for med in case.medications:
        entry = f"- {med.describe()} ({_STATUS_PHRASES[med.status]})"
        if med.status is MedicationStatus.CHANGED_DOSE and med.prior_dose:
            entry += f"; increased from {med.prior_dose}"
        if med.change_timing:
            entry += f" {med.change_timing}"
        if med.change_reason:
            entry += f", {med.change_reason}"
        lines.append(entry)
    return lines


def _allergy_section(case: StructuredCase) -> list[str]:
    if not case.allergies:
        return []
    lines = ["ALLERGIES:"]
    for allergy in case.allergies:
        lines.append(f"- {allergy.describe()} ({allergy.severity.value})")
    return lines


def _vitals_section(case: StructuredCase) -> list[str]:
    if not case.vitals:
        return []
    parts = []
    for vital in case.vitals:
        rendered = vital.describe()
        if vital.is_abnormal:
            rendered += f" ({_FLAG_LABELS[vital.abnormal_flag]})"
        parts.append(rendered)
    return ["VITAL SIGNS:", ", ".join(parts)]


def _labs_section(case: StructuredCase) -> list[str]:
    if not case.labs:
        return []
    lines = ["LABORATORY RESULTS:"]
    for lab in case.labs:
        entry = f"- {lab.describe()} ({lab.timing_description})"
        if lab.is_abnormal:
            entry += f" - {_FLAG_LABELS[lab.abnormal_flag]}"
            if lab.reference_low is not None and lab.reference_high is not None:
                entry += f" [ref {lab.reference_low:g}-{lab.reference_high:g} {lab.unit}]"
        lines.append(entry)
    for trend in case.lab_trends:
        verb = _TREND_VERBS.get(trend.trend, "changed")
        lines.append(
            f"{trend.lab_name} has {verb} from "
            f"{trend.format_value(trend.earlier_value)} {trend.unit} ({trend.earlier_timing}) "
            f"to {trend.format_value(trend.later_value)} {trend.unit} ({trend.later_timing})."
        )
    return lines


def _findings_section(case: StructuredCase) -> list[str]:
    positives = [f for f in case.findings if f.present]
    if not positives:
        return []
    joined = "; ".join(f.description for f in positives)
    return ["EXAMINATION AND FINDINGS:", _sentence_case(joined) + "."]


def _procedure_section(case: StructuredCase) -> list[str]:
    if not case.procedures:
        return []
    lines = ["PROCEDURES AND STUDIES:"]
    for procedure in case.procedures:
        if procedure.performed:
            outcome = f": {procedure.outcome}" if procedure.outcome else ""
            lines.append(f"- {procedure.name} ({procedure.timing_description}){outcome}")
        else:
            lines.append(f"- {procedure.name} was not performed at this visit.")
    return lines


def _course_section(case: StructuredCase) -> list[str]:
    """Narrative hospital/clinical course, only for cases with a real timeline."""
    narrative_types = {
        TimelineEventType.DETERIORATION,
        TimelineEventType.IMPROVEMENT,
        TimelineEventType.MEDICATION_CHANGE,
    }
    events = [e for e in case.timeline if e.event_type in narrative_types]
    if len(events) < 2:
        return []
    lines = ["CLINICAL COURSE:"]
    for event in events:
        lines.append(f"- {event.description}.")
    return lines


def _assessment_section(case: StructuredCase) -> list[str]:
    if not case.diagnoses:
        return []
    lines = ["ASSESSMENT:"]
    for index, diagnosis in enumerate(case.diagnoses, start=1):
        marker = " (primary)" if diagnosis.is_primary else ""
        lines.append(f"{index}. {diagnosis.name}{marker}")
    return lines


def _plan_section(case: StructuredCase) -> list[str]:
    if not case.follow_up:
        return []
    lines = ["PLAN:"]
    for plan in case.follow_up:
        actions = ", ".join(plan.actions) if plan.actions else "clinical reassessment"
        lines.append(f"- {actions.capitalize()}; follow up with {plan.provider} in {plan.interval}.")
    return lines


def _review_of_systems(case: StructuredCase, rng: Random) -> list[str]:
    """Boilerplate that lengthens a note without adding clinical content."""
    return [
        "REVIEW OF SYSTEMS:",
        "Constitutional, cardiovascular, respiratory, gastrointestinal, genitourinary, "
        "musculoskeletal, neurologic, and psychiatric systems reviewed and negative "
        "except as documented above.",
    ]


def render_note(case: StructuredCase, rng: Random) -> str:
    """Render the source clinical note.

    Section selection responds to the rendering hints features set on the
    draft: terse cases drop optional sections, verbose cases gain a review of
    systems, and duplicate-documentation cases restate one finding in a later
    section the way copy-forward documentation does.
    """
    sections: list[list[str]] = [
        ["CLINICAL NOTE"],
        [f"Encounter: {case.encounter.encounter_type} ({case.encounter.setting.replace('_', ' ')})"],
        ["SUBJECTIVE:", *_subjective_section(case, rng)],
    ]

    terse = bool(case.categories) and all(
        category in {FailureModeCategory.SHORT_NOTE, FailureModeCategory.STRAIGHTFORWARD}
        for category in case.categories
    )

    sections.append(_medication_section(case))
    sections.append(_allergy_section(case))
    sections.append(_vitals_section(case))
    sections.append(_labs_section(case))
    sections.append(_findings_section(case))
    sections.append(_procedure_section(case))
    sections.append(_course_section(case))

    if not terse:
        sections.append(_assessment_section(case))
    sections.append(_plan_section(case))

    if (
        FailureModeCategory.LONG_NOTE in case.categories
        or FailureModeCategory.HIGH_COMPLEXITY in case.categories
    ):
        sections.append(_review_of_systems(case, rng))

    if FailureModeCategory.DUPLICATE_DOCUMENTATION in case.categories and case.findings:
        repeated = case.findings[0]
        prefix = "As documented above, " if repeated.present else "Again noted, "
        sections.append(["INTERVAL NOTE:", f"{prefix}{repeated.description}."])

    body = "\n\n".join("\n".join(section) for section in sections if section)
    return body.strip()
