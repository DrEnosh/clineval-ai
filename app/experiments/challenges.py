"""Methodology challenge suite: stressing the Phase 2-4 assumptions.

The deterministic evaluator was validated in Phase 3 against the benchmark's
*own* reference summaries and controlled corruptions of them. That establishes
it is self-consistent. It does **not** establish that it survives contact with
clinical language written in a different register - which is exactly what a
real model produces.

This module writes summaries that are clinically equivalent to the reference
but phrased the way a clinician or a model actually would, and asserts what the
evaluator should conclude:

    register              A different but faithful way of writing the summary
    compression           Terse, drops routine content, keeps every critical fact
    abstraction           Numbers replaced by clinical characterization
    negation_forms        The same negative expressed several ways
    numeric_forms         The same value written several ways
    medication_language   Natural clinical phrasing for status changes
    long_note             The same content, buried in a long summary

**These are not model outputs.** They are hand-designed transformations, and
they are honest about that: they test whether the evaluator understands
clinical language, not how any particular model writes. Evidence about real
model behaviour requires real models (see docs/REAL_MODEL_EVALUATION.md).

Each challenge declares an expectation:

- `must_not_fire` - findings here are **false positives**. The summary is
  clinically faithful; flagging it is an evaluator error.
- `must_fire` - findings here are required. Missing them is a **false
  negative**.
- `undecided` - the correct answer is genuinely debatable. The suite records
  what the evaluator does without asserting it is right, because encoding a
  contested judgment as a passing test would launder an opinion into a fact.

The `abstraction` category is deliberately `undecided`. Whether "renal function
worsened" adequately conveys "creatinine rose from 1.4 to 1.9 mg/dL" is a
clinical judgment, not a technical one, and this project should not silently
pick a side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from app.core.enums import ConditionStatus, FindingType, MedicationStatus
from app.models.schemas import BenchmarkCase

Expectation = str  # "must_not_fire" | "must_fire" | "undecided"


@dataclass(frozen=True)
class Challenge:
    """One methodology probe against one case."""

    challenge_id: str
    category: str
    case_id: str
    summary: str
    expectation: Expectation
    rationale: str
    #: For `must_fire`, the finding types that are required.
    required_findings: tuple[FindingType, ...] = ()
    #: Finding types tolerated even under `must_not_fire`.
    tolerated_findings: tuple[FindingType, ...] = ()


@dataclass
class ChallengeOutcome:
    """What the evaluator actually did with one challenge."""

    challenge: Challenge
    finding_types: list[str] = field(default_factory=list)
    findings: list = field(default_factory=list)

    @property
    def unexpected(self) -> list[str]:
        """Findings that fired but should not have."""
        if self.challenge.expectation != "must_not_fire":
            return []
        tolerated = {f.value for f in self.challenge.tolerated_findings}
        return sorted({t for t in self.finding_types if t not in tolerated})

    @property
    def missing(self) -> list[str]:
        """Required findings that did not fire."""
        if self.challenge.expectation != "must_fire":
            return []
        fired = set(self.finding_types)
        return sorted(
            f.value for f in self.challenge.required_findings if f.value not in fired
        )

    @property
    def passed(self) -> bool:
        if self.challenge.expectation == "undecided":
            return True
        return not self.unexpected and not self.missing

    def report(self) -> dict:
        return {
            "challenge_id": self.challenge.challenge_id,
            "category": self.challenge.category,
            "case_id": self.challenge.case_id,
            "expectation": self.challenge.expectation,
            "findings": sorted(set(self.finding_types)),
            "unexpected_findings": self.unexpected,
            "missing_findings": self.missing,
            "passed": self.passed,
            "rationale": self.challenge.rationale,
        }


# ---------------------------------------------------------------------------
# Summary transformations
# ---------------------------------------------------------------------------


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def register_shift(case: BenchmarkCase) -> str:
    """Rewrite the reference summary in a different clinical register.

    Same facts, same values, different words and connectives - the most basic
    thing a real model does that a naive matcher breaks on.
    """
    summary = case.reference_summary
    substitutions = (
        ("seen for", "presented for"),
        ("rose from", "increased from"),
        ("fell from", "declined from"),
        ("was discontinued", "was stopped"),
        ("was increased from", "was uptitrated from"),
        ("was newly started", "was commenced"),
        ("Continued on", "Remains on"),
        ("Documented negatives:", "Pertinent negatives:"),
        ("Severe allergy:", "Significant allergy:"),
        ("Studies:", "Investigations:"),
        ("Plan:", "Ongoing plan:"),
        ("Historical, not active:", "Past history, currently inactive:"),
        ("is critically abnormal at", "is markedly abnormal at"),
        ("is abnormal at", "is deranged at"),
        ("Abnormal vital signs:", "Vital sign abnormalities:"),
        ("between", "from"),
    )
    for source, target in substitutions:
        summary = summary.replace(source, target)
    return summary


def compression(case: BenchmarkCase) -> str:
    """Keep every critical fact; drop routine content.

    This is what a good summarizer does, and it is the transformation most
    likely to produce a *legitimate* omission the evaluator should tolerate.
    Sentences carrying critical content are identified from the structured
    case, not guessed at.
    """
    structured = case.structured_case
    summary = case.reference_summary
    if structured is None:
        return summary

    # Anchors that must survive. The medication list is included in full, not
    # just the changes: an early version of this challenge dropped current
    # medications and the evaluator flagged them, which was *correct* -
    # dropping the antibiotic from a cellulitis summary is real information
    # loss, not harmless concision. The challenge was wrong, not the evaluator.
    anchors: list[str] = []
    for lab in structured.labs:
        if lab.is_abnormal:
            anchors.append(lab.format_value())
    for trend in structured.lab_trends:
        if trend.clinically_significant:
            anchors.append(trend.format_value(trend.later_value))
    for med in structured.medications:
        anchors.append(med.drug_name)
    for allergy in structured.allergies:
        anchors.append(allergy.substance)
    for finding in structured.relevant_negatives:
        anchors.append(finding.description)
    for conflict in structured.conflicts:
        anchors.append(conflict.later_statement[:30])

    kept = [
        sentence
        for index, sentence in enumerate(_sentences(summary))
        # Always keep the opening (demographics and primary problem), plus any
        # sentence carrying an anchor.
        if index == 0 or any(anchor and anchor in sentence for anchor in anchors)
    ]
    return " ".join(kept)


def abstraction(case: BenchmarkCase) -> str:
    """Replace numeric trends with clinical characterization.

    "Creatinine rose from 1.4 to 1.9 mg/dL" becomes "renal function has
    worsened". Whether this is an acceptable summary is a clinical judgment,
    so challenges built from it are `undecided`.
    """
    structured = case.structured_case
    summary = case.reference_summary
    if structured is None:
        return summary

    characterizations = {
        "Creatinine": "renal function has worsened",
        "eGFR": "renal function has worsened",
        "Hemoglobin": "the anemia has progressed",
        "BNP": "heart failure markers have risen",
        "White blood cell count": "the inflammatory response has increased",
        "Potassium": "potassium is now deranged",
        "Lactate": "perfusion markers have worsened",
        "Troponin I": "cardiac markers have risen",
        "Sodium": "sodium is now deranged",
    }

    for trend in structured.lab_trends:
        if not trend.clinically_significant:
            continue
        phrase = characterizations.get(trend.lab_name)
        if phrase is None:
            continue
        pattern = re.compile(
            rf"{re.escape(trend.lab_name)}\s+(?:rose|fell|remained stable|changed).*?\.(?=\s|$)",
            re.IGNORECASE,
        )
        summary = pattern.sub(f"Interval change: {phrase}.", summary, count=1)
    return summary


def negation_forms(case: BenchmarkCase) -> str:
    """Express the same negatives through several linguistic constructions.

    Real clinical text does not write "no X" every time. This probes whether
    the negation detector handles the forms clinicians actually use.
    """
    structured = case.structured_case
    summary = case.reference_summary
    if structured is None:
        return summary

    forms: tuple[Callable[[str], str], ...] = (
        lambda term: f"denies {term}",
        lambda term: f"negative for {term}",
        lambda term: f"{term} is absent",
        lambda term: f"without {term}",
        lambda term: f"there is no {term}",
    )

    negatives = structured.relevant_negatives
    if not negatives:
        return summary

    rewritten: list[str] = []
    for index, finding in enumerate(negatives):
        core = re.sub(
            r"^(?:no known |no history of |no recent |no )", "", finding.description, flags=re.I
        ).strip()
        rewritten.append(forms[index % len(forms)](core))

    # Replace the summary's own negatives block with the rewritten forms.
    pattern = re.compile(r"Documented negatives:[^.]*\.")
    replacement = "On review the patient " + "; ".join(rewritten) + "."
    if pattern.search(summary):
        return pattern.sub(replacement, summary, count=1)
    return f"{summary} {replacement}"


def numeric_forms(case: BenchmarkCase) -> str:
    """Write the same values in other valid notations.

    Trailing zeroes, arrow notation, and "up to" phrasing are all common in
    clinical text and none of them change the value.
    """
    structured = case.structured_case
    summary = case.reference_summary
    if structured is None:
        return summary

    for trend in structured.lab_trends:
        if not trend.clinically_significant:
            continue
        earlier = trend.format_value(trend.earlier_value)
        later = trend.format_value(trend.later_value)
        summary = summary.replace(
            f"from {earlier} to {later}", f"from {earlier} up to {later}"
        )

    # Decimal padding: 1.9 -> 1.90 is the same number.
    summary = re.sub(r"\b(\d+\.\d)\b(?=\s*(?:mg|mmol|mEq|g|ng|%))", r"\g<1>0", summary)
    return summary


def medication_language(case: BenchmarkCase) -> str:
    """Use natural clinical phrasing for medication status.

    "held", "d/c'd", "no longer on" are how clinicians write discontinuation.
    A status detector keyed only on "discontinued" would produce false
    positives on all of them.
    """
    structured = case.structured_case
    summary = case.reference_summary
    if structured is None:
        return summary

    phrasings = ("has been held", "was stopped", "is no longer being taken")
    stopped = [m for m in structured.medications if m.status is MedicationStatus.DISCONTINUED]
    for index, med in enumerate(stopped):
        summary = summary.replace(
            f"{med.drug_name} was discontinued",
            f"{med.drug_name} {phrasings[index % len(phrasings)]}",
            1,
        )

    for med in structured.medications:
        if med.status is MedicationStatus.CHANGED_DOSE and med.prior_dose:
            summary = summary.replace(
                f"{med.drug_name} was increased from {med.prior_dose} to {med.dose}",
                f"{med.drug_name} dose was stepped up from {med.prior_dose} to {med.dose}",
                1,
            )
    summary = summary.replace("Continued on", "Ongoing therapy includes")
    return summary


def historical_language(case: BenchmarkCase) -> str:
    """Express current-versus-historical distinctions in other words."""
    structured = case.structured_case
    summary = case.reference_summary
    if structured is None:
        return summary

    summary = summary.replace("Historical, not active:", "Resolved and no longer active:")
    summary = summary.replace(
        "was taken in the past and is not a current medication",
        "was used previously and is not part of the current regimen",
    )
    return summary


def long_note(case: BenchmarkCase) -> str:
    """Bury the same content in a much longer summary.

    Probes whether length alone induces errors - a context-window proxy that
    does not require a real model.
    """
    padding = (
        "The patient was reviewed on the ward round and remains comfortable. "
        "Nursing observations have been unremarkable overnight. "
        "The multidisciplinary team discussed the plan and the patient is agreeable. "
        "Social circumstances were reviewed and no additional support is required. "
        "Documentation was completed contemporaneously. "
    )
    return f"{padding * 3}{case.reference_summary} {padding * 3}"


#: Transformation name -> (function, expectation, rationale)
TRANSFORMATIONS: dict[str, tuple[Callable[[BenchmarkCase], str], Expectation, str]] = {
    "register": (
        register_shift,
        "must_not_fire",
        "Same facts and values in different words. Flagging this is a false positive.",
    ),
    "compression": (
        compression,
        "must_not_fire",
        "Routine content dropped, every critical fact kept. Concision is not an error.",
    ),
    "abstraction": (
        abstraction,
        "undecided",
        "Numbers replaced by clinical characterization. Whether this is acceptable "
        "is a clinical judgment; the suite records behaviour without asserting.",
    ),
    "negation_forms": (
        negation_forms,
        "must_not_fire",
        "The same negatives in other constructions. Polarity is preserved throughout.",
    ),
    "numeric_forms": (
        numeric_forms,
        "must_not_fire",
        "The same values in other valid notations. 1.90 is 1.9.",
    ),
    "medication_language": (
        medication_language,
        "must_not_fire",
        "Natural clinical phrasing for medication status. 'held' means discontinued.",
    ),
    "historical_language": (
        historical_language,
        "must_not_fire",
        "Current-versus-historical distinction expressed in other words.",
    ),
    "long_note": (
        long_note,
        "must_not_fire",
        "Identical content buried in padding. Length alone must not induce findings.",
    ),
}


def build_challenges(cases: list[BenchmarkCase]) -> list[Challenge]:
    """Build the full challenge suite for a set of cases.

    A transformation that leaves a case's summary unchanged produces no
    challenge: testing an identity transformation would inflate the pass rate
    without testing anything.
    """
    challenges: list[Challenge] = []
    for case in cases:
        for name, (transform, expectation, rationale) in TRANSFORMATIONS.items():
            summary = transform(case)
            if summary == case.reference_summary or not summary.strip():
                continue
            challenges.append(
                Challenge(
                    challenge_id=f"{case.case_id}-{name}",
                    category=name,
                    case_id=case.case_id,
                    summary=summary,
                    expectation=expectation,
                    rationale=rationale,
                )
            )
    return challenges


def run_challenges(cases: list[BenchmarkCase], runner=None) -> list[ChallengeOutcome]:
    """Evaluate every challenge and record what the evaluator concluded."""
    from app.evaluation.runner import EvaluationRunner
    from app.models.schemas import AIOutput

    runner = runner or EvaluationRunner()
    by_id = {case.case_id: case for case in cases}
    outcomes: list[ChallengeOutcome] = []

    for challenge in build_challenges(cases):
        case = by_id[challenge.case_id]
        output = AIOutput(
            case_id=case.case_id,
            model_id="methodology-challenge",
            summary_text=challenge.summary,
            raw_metadata={"provider_kind": "synthetic_test_provider", "is_real_model_output": False},
        )
        result = runner.evaluate_case(case, output)
        outcomes.append(
            ChallengeOutcome(
                challenge=challenge,
                finding_types=[f.finding_type.value for f in result.findings],
                findings=result.findings,
            )
        )
    return outcomes


def summarize_outcomes(outcomes: list[ChallengeOutcome]) -> dict:
    """Aggregate challenge results by category."""
    by_category: dict[str, dict] = {}
    for outcome in outcomes:
        entry = by_category.setdefault(
            outcome.challenge.category,
            {
                "total": 0, "passed": 0, "failed": 0, "undecided": 0,
                "unexpected_finding_types": {},
            },
        )
        entry["total"] += 1
        if outcome.challenge.expectation == "undecided":
            entry["undecided"] += 1
        elif outcome.passed:
            entry["passed"] += 1
        else:
            entry["failed"] += 1
        for finding_type in outcome.unexpected:
            entry["unexpected_finding_types"][finding_type] = (
                entry["unexpected_finding_types"].get(finding_type, 0) + 1
            )

    decidable = [o for o in outcomes if o.challenge.expectation != "undecided"]
    return {
        "total_challenges": len(outcomes),
        "decidable": len(decidable),
        "passed": sum(1 for o in decidable if o.passed),
        "failed": sum(1 for o in decidable if not o.passed),
        "pass_rate": (
            round(sum(1 for o in decidable if o.passed) / len(decidable), 4)
            if decidable else None
        ),
        "by_category": dict(sorted(by_category.items())),
        "note": (
            "Challenges are hand-designed transformations of the benchmark's own "
            "reference summaries, not real model outputs. They test whether the "
            "evaluator understands clinical language, not how any model writes."
        ),
    }
