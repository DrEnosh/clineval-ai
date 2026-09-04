"""Controlled failure injection - evaluating the evaluator.

    TEST AND VALIDATION USE ONLY.

Takes a correct reference summary, injects one known defect, and records what
the evaluator *should* detect. Running this across the benchmark answers the
question that matters most about a detector: does it actually detect?

Seven injections, one per failure mode the deterministic evaluators claim to
cover:

| Injection | Defect | Expected finding |
|---|---|---|
| `remove_critical_fact` | Deletes a sentence carrying a critical fact | `critical_omission` |
| `reverse_negation` | Turns "no history of X" into "has X" | `negation_reversal` |
| `change_medication` | Substitutes a drug not in the case | `medication_hallucination` |
| `change_dose` | Alters a stated dose | `medication_dose_error` |
| `change_lab_value` | Alters a stated laboratory value | `contradiction` |
| `reverse_temporal_trend` | Swaps trend endpoints and direction words | `temporal_trend_reversal` |
| `add_unsupported_diagnosis` | Appends a condition absent from the case | `unsupported_addition` |

Each returns `None` when the case cannot support that injection - a case with
no medications cannot have a dose changed. Skipped injections are reported as
skipped, never as passes: counting an injection that never happened as a
detection success would be exactly the self-flattery this harness exists to
prevent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from random import Random

from app.benchmark import content
from app.core.enums import ConditionStatus, FindingType, MedicationStatus
from app.evaluation.normalization import normalize_text
from app.models.schemas import BenchmarkCase


@dataclass(frozen=True)
class Injection:
    """A corrupted summary plus what the evaluator is expected to find."""

    name: str
    corrupted_summary: str
    expected_finding: FindingType
    description: str
    original_fragment: str = ""
    replacement_fragment: str = ""
    # Additional finding types that also constitute correct detection of this
    # defect. One corruption can legitimately surface as more than one true
    # finding - a fabricated laboratory value both contradicts the source and
    # is an unsupported assertion - and demanding exactly one type would test
    # the evaluator's labelling taxonomy rather than its detection.
    also_acceptable: frozenset[FindingType] = frozenset()

    @property
    def acceptable_findings(self) -> frozenset[FindingType]:
        return frozenset({self.expected_finding}) | self.also_acceptable


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _first_number(text: str) -> str | None:
    match = re.search(r"\d+(?:\.\d+)?", text)
    return match.group(0) if match else None


# ---------------------------------------------------------------------------
# Injections
# ---------------------------------------------------------------------------


def remove_critical_fact(case: BenchmarkCase, rng: Random) -> Injection | None:
    """Delete the summary sentence carrying a critical fact.

    Targets a lab or allergy fact, whose entity has a distinctive term that
    can be located in the summary reliably.
    """
    structured = case.structured_case
    if structured is None:
        return None

    summary = case.reference_summary
    # Only target labs that the benchmark actually marked critical. Removing a
    # merely-abnormal non-key analyte would expect a critical_omission finding
    # the evaluator is right not to raise, and would score a correct evaluator
    # as having missed something.
    critical_refs = {fact.source_ref for fact in case.critical_facts}

    for lab in structured.labs:
        if lab.entity_id not in critical_refs:
            continue
        # A value also stated by a trend sentence survives the deletion, so
        # removing its standalone sentence would not actually omit it.
        trend_values = {
            value
            for trend in structured.lab_trends
            if trend.lab_name == lab.name
            for value in (trend.earlier_value, trend.later_value)
        }
        if lab.value in trend_values:
            continue

        for sentence in _sentences(summary):
            if normalize_text(lab.name) in normalize_text(sentence) and lab.format_value() in sentence:
                corrupted = re.sub(r"\s{2,}", " ", summary.replace(sentence, "").strip())
                if corrupted == summary or not corrupted:
                    continue
                return Injection(
                    name="remove_critical_fact",
                    corrupted_summary=corrupted,
                    expected_finding=FindingType.CRITICAL_OMISSION,
                    description=f"Removed the sentence reporting {lab.describe()}",
                    original_fragment=sentence,
                )
    return None


def reverse_negation(case: BenchmarkCase, rng: Random) -> Injection | None:
    """Flip a documented negative into a positive assertion."""
    structured = case.structured_case
    if structured is None:
        return None

    summary = case.reference_summary
    for finding in structured.relevant_negatives:
        description = finding.description
        if description not in summary:
            continue
        reversed_text = re.sub(
            r"^no history of\s+", "patient has a history of ", description, flags=re.IGNORECASE
        )
        if reversed_text == description:
            reversed_text = re.sub(r"^no\s+", "patient has ", description, flags=re.IGNORECASE)
        if reversed_text == description:
            continue
        return Injection(
            name="reverse_negation",
            corrupted_summary=summary.replace(description, reversed_text),
            expected_finding=FindingType.NEGATION_REVERSAL,
            description=f"Reversed the negative '{description}'",
            original_fragment=description,
            replacement_fragment=reversed_text,
        )
    return None


def change_medication(case: BenchmarkCase, rng: Random) -> Injection | None:
    """Replace a drug with one that is not in the case at all."""
    structured = case.structured_case
    if structured is None or not structured.medications:
        return None

    summary = case.reference_summary
    case_drugs = {m.drug_name for m in structured.medications}
    substitutes = sorted(set(content.all_drug_names()) - case_drugs)
    if not substitutes:
        return None
    substitute = substitutes[rng.randrange(len(substitutes))]

    for medication in structured.medications:
        if medication.drug_name in summary:
            return Injection(
                name="change_medication",
                corrupted_summary=summary.replace(medication.drug_name, substitute),
                expected_finding=FindingType.MEDICATION_HALLUCINATION,
                description=f"Replaced {medication.drug_name} with {substitute}",
                original_fragment=medication.drug_name,
                replacement_fragment=substitute,
            )
    return None


def change_dose(case: BenchmarkCase, rng: Random) -> Injection | None:
    """Alter a stated dose to a clearly different value."""
    structured = case.structured_case
    if structured is None:
        return None

    summary = case.reference_summary
    for medication in structured.medications:
        if not medication.dose or medication.drug_name not in summary:
            continue
        dose_number = _first_number(medication.dose)
        if dose_number is None or medication.dose not in summary:
            continue
        altered_number = f"{float(dose_number) * 4:g}"
        altered_dose = medication.dose.replace(dose_number, altered_number, 1)
        return Injection(
            name="change_dose",
            corrupted_summary=summary.replace(medication.dose, altered_dose),
            expected_finding=FindingType.MEDICATION_DOSE_ERROR,
            description=f"Changed {medication.drug_name} dose {medication.dose} -> {altered_dose}",
            original_fragment=medication.dose,
            replacement_fragment=altered_dose,
        )
    return None


def change_lab_value(case: BenchmarkCase, rng: Random) -> Injection | None:
    """Alter a stated laboratory value beyond tolerance."""
    structured = case.structured_case
    if structured is None:
        return None

    summary = case.reference_summary
    # Prefer a standalone value. Altering a trend endpoint damages the trend
    # as well, so the evaluator correctly reports temporal findings and a
    # fabricated value rather than a plain value contradiction - true findings,
    # but not the one this injection is written to test.
    trend_values = {
        (trend.lab_name, value)
        for trend in structured.lab_trends
        for value in (trend.earlier_value, trend.later_value)
    }
    standalone = [lab for lab in structured.labs if (lab.name, lab.value) not in trend_values]

    for lab in standalone + structured.labs:
        rendered = lab.format_value()
        if normalize_text(lab.name) not in normalize_text(summary) or rendered not in summary:
            continue
        altered = f"{lab.value * 2 + 1:g}"
        if altered == rendered:
            continue
        return Injection(
            name="change_lab_value",
            corrupted_summary=summary.replace(rendered, altered, 1),
            expected_finding=FindingType.CONTRADICTION,
            description=f"Changed {lab.name} {rendered} -> {altered} {lab.unit}",
            original_fragment=rendered,
            replacement_fragment=altered,
            # A fabricated value is genuinely an unsupported assertion too, and
            # altering a trend endpoint genuinely damages the trend.
            also_acceptable=frozenset(
                {
                    FindingType.UNSUPPORTED_ADDITION,
                    FindingType.TEMPORAL_INFORMATION_LOSS,
                    FindingType.TEMPORAL_VALUE_CONFUSION,
                }
            ),
        )
    return None


def reverse_temporal_trend(case: BenchmarkCase, rng: Random) -> Injection | None:
    """Swap a trend's endpoints and invert its direction word."""
    structured = case.structured_case
    if structured is None:
        return None

    summary = case.reference_summary
    for trend in structured.lab_trends:
        earlier = trend.format_value(trend.earlier_value)
        later = trend.format_value(trend.later_value)
        fragment = f"from {earlier} to {later}"
        if fragment not in summary:
            continue
        swapped = f"from {later} to {earlier}"
        corrupted = summary.replace(fragment, swapped, 1)
        corrupted = corrupted.replace("rose", "__TMP__").replace("fell", "rose").replace("__TMP__", "fell")
        return Injection(
            name="reverse_temporal_trend",
            corrupted_summary=corrupted,
            expected_finding=FindingType.TEMPORAL_TREND_REVERSAL,
            description=f"Reversed the {trend.lab_name} trend ({earlier} -> {later})",
            original_fragment=fragment,
            replacement_fragment=swapped,
        )
    return None


def add_unsupported_diagnosis(case: BenchmarkCase, rng: Random) -> Injection | None:
    """Append a diagnosis the case does not contain."""
    structured = case.structured_case
    if structured is None:
        return None

    corpus = structured.text_corpus()
    candidates = [
        template.name for template in content.CONDITION_TEMPLATES
        if normalize_text(template.name) not in corpus
    ]
    if not candidates:
        return None
    invented = candidates[rng.randrange(len(candidates))]
    return Injection(
        name="add_unsupported_diagnosis",
        corrupted_summary=f"{case.reference_summary} The patient also has {invented.lower()}.",
        expected_finding=FindingType.UNSUPPORTED_ADDITION,
        description=f"Added unsupported diagnosis '{invented}'",
        replacement_fragment=invented,
    )


def change_medication_status(case: BenchmarkCase, rng: Random) -> Injection | None:
    """Present a discontinued drug as if it were continuing."""
    structured = case.structured_case
    if structured is None:
        return None

    summary = case.reference_summary
    for medication in structured.medications:
        if medication.status is not MedicationStatus.DISCONTINUED:
            continue
        # Replace the whole sentence, not just the leading fragment. The
        # reference sentence carries the reason in parentheses ("... was
        # discontinued 1 month ago (discontinued due to an adverse effect)"),
        # so a fragment swap leaves the word "discontinued" in the sentence and
        # the corrupted text still reads as discontinued - the injection would
        # then be testing nothing.
        for sentence in _sentences(summary):
            if f"{medication.drug_name} was discontinued" not in sentence:
                continue
            replacement = f"{medication.drug_name} is continued at the current dose."
            return Injection(
                name="change_medication_status",
                corrupted_summary=summary.replace(sentence, replacement, 1),
                expected_finding=FindingType.MEDICATION_STATUS_ERROR,
                description=f"Presented discontinued {medication.drug_name} as current",
                original_fragment=sentence,
                replacement_fragment=replacement,
            )
    return None


def present_resolved_as_current(case: BenchmarkCase, rng: Random) -> Injection | None:
    """Strip the 'historical, not active' framing from a resolved condition."""
    structured = case.structured_case
    if structured is None:
        return None

    summary = case.reference_summary
    resolved = [c for c in structured.conditions if c.status is not ConditionStatus.ACTIVE]
    for condition in resolved:
        marker = "Historical, not active:"
        if marker not in summary:
            continue
        for sentence in _sentences(summary):
            if marker in sentence:
                replacement = f"Active problems include {condition.name.lower()}."
                return Injection(
                    name="present_resolved_as_current",
                    corrupted_summary=summary.replace(sentence, replacement, 1),
                    expected_finding=FindingType.HISTORICAL_CURRENT_CONFUSION,
                    description=f"Presented resolved {condition.name} as an active problem",
                    original_fragment=sentence,
                    replacement_fragment=replacement,
                )
    return None


INJECTIONS = {
    "remove_critical_fact": remove_critical_fact,
    "reverse_negation": reverse_negation,
    "change_medication": change_medication,
    "change_dose": change_dose,
    "change_lab_value": change_lab_value,
    "reverse_temporal_trend": reverse_temporal_trend,
    "add_unsupported_diagnosis": add_unsupported_diagnosis,
    "change_medication_status": change_medication_status,
    "present_resolved_as_current": present_resolved_as_current,
}


def inject(name: str, case: BenchmarkCase, seed: int = 0) -> Injection | None:
    """Apply one named injection to a case, or None if inapplicable."""
    if name not in INJECTIONS:
        raise KeyError(f"Unknown injection '{name}'. Available: {', '.join(sorted(INJECTIONS))}")
    return INJECTIONS[name](case, Random(f"{name}:{case.case_id}:{seed}"))


def inject_all(case: BenchmarkCase, seed: int = 0) -> dict[str, Injection | None]:
    """Apply every injection to a case; inapplicable ones map to None."""
    return {name: inject(name, case, seed) for name in sorted(INJECTIONS)}
