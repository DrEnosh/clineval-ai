"""Benchmark integrity checking.

A benchmark whose ground truth is wrong produces confidently wrong evaluations,
which is worse than no benchmark at all. These checks are the safeguard: they
verify that every annotation traces to real structured content, that the three
renderings agree, and that nothing identifiable was generated.

Checks are pure functions returning `IntegrityIssue` records rather than
raising, so the audit script can report every problem in one pass instead of
stopping at the first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.benchmark import content
from app.core.enums import FactImportance, FactPolarity
from app.models.schemas import BenchmarkCase
from app.models.structured import (
    ClinicalFindingEntry,
    LabResultEntry,
    MedicationEntry,
    ProcedureEntry,
    StructuredCase,
)


@dataclass(frozen=True)
class IntegrityIssue:
    check: str
    case_id: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.check}] {self.case_id}: {self.detail}"


# Patterns that would indicate identifiable content. The generator cannot
# produce any of these, so a hit means something changed that needs review.
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("phone", re.compile(r"\b\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("calendar_date", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
    ("named_person", re.compile(r"\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+")),
    ("mrn", re.compile(r"\b(?:MRN|Medical Record Number)\b", re.IGNORECASE)),
)

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def _all_facts(case: BenchmarkCase) -> list:
    return [
        *case.critical_facts, *case.non_critical_facts, *case.medication_facts,
        *case.temporal_facts, *case.negations, *case.contradiction_facts,
    ]


# ---------------------------------------------------------------------------
# Check 1: every annotation refers to real structured content
# ---------------------------------------------------------------------------


def check_annotations_reference_source(case: BenchmarkCase) -> list[IntegrityIssue]:
    structured = case.structured_case
    if structured is None:
        return []
    index = structured.entity_index()
    issues: list[IntegrityIssue] = []
    for fact in _all_facts(case):
        if fact.source_ref is None:
            issues.append(
                IntegrityIssue(
                    "annotations_reference_source", case.case_id,
                    f"fact {fact.fact_id} has no source_ref",
                )
            )
        elif fact.source_ref not in index:
            issues.append(
                IntegrityIssue(
                    "annotations_reference_source", case.case_id,
                    f"fact {fact.fact_id} references unknown entity {fact.source_ref}",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Check 2: every critical fact exists in the structured source
# ---------------------------------------------------------------------------


def check_critical_facts_grounded(case: BenchmarkCase) -> list[IntegrityIssue]:
    structured = case.structured_case
    if structured is None:
        return []
    index = structured.entity_index()
    issues: list[IntegrityIssue] = []
    for fact in case.critical_facts:
        if fact.importance is not FactImportance.CRITICAL:
            issues.append(
                IntegrityIssue(
                    "critical_facts_grounded", case.case_id,
                    f"fact {fact.fact_id} is in critical_facts but marked {fact.importance.value}",
                )
            )
        if fact.source_ref not in index:
            issues.append(
                IntegrityIssue(
                    "critical_facts_grounded", case.case_id,
                    f"critical fact {fact.fact_id} is not grounded in the structured case",
                )
            )
        if fact.criticality_rule is None:
            issues.append(
                IntegrityIssue(
                    "critical_facts_grounded", case.case_id,
                    f"critical fact {fact.fact_id} carries no criticality rule id",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Check 3: medication annotations match real medications
# ---------------------------------------------------------------------------


def check_medication_annotations(case: BenchmarkCase) -> list[IntegrityIssue]:
    structured = case.structured_case
    if structured is None:
        return []
    index = structured.entity_index()
    note = case.clinical_note.lower()
    issues: list[IntegrityIssue] = []

    for fact in case.medication_facts:
        entity = index.get(fact.source_ref or "")
        if not isinstance(entity, MedicationEntry):
            issues.append(
                IntegrityIssue(
                    "medication_annotations", case.case_id,
                    f"medication fact {fact.fact_id} does not reference a medication entity",
                )
            )
            continue
        if entity.drug_name != fact.drug_name:
            issues.append(
                IntegrityIssue(
                    "medication_annotations", case.case_id,
                    f"medication fact {fact.fact_id} names {fact.drug_name} but entity is "
                    f"{entity.drug_name}",
                )
            )
        if entity.status is not fact.status:
            issues.append(
                IntegrityIssue(
                    "medication_annotations", case.case_id,
                    f"medication fact {fact.fact_id} status {fact.status.value} disagrees with "
                    f"entity status {entity.status.value}",
                )
            )
        if fact.drug_name.lower() not in note:
            issues.append(
                IntegrityIssue(
                    "medication_annotations", case.case_id,
                    f"medication fact {fact.fact_id} names {fact.drug_name}, absent from the note",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Check 4: temporal annotations match real timeline data
# ---------------------------------------------------------------------------


def check_temporal_annotations(case: BenchmarkCase) -> list[IntegrityIssue]:
    structured = case.structured_case
    if structured is None:
        return []
    index = structured.entity_index()
    issues: list[IntegrityIssue] = []

    for fact in case.temporal_facts:
        trend = index.get(fact.source_ref or "")
        if trend is None or not hasattr(trend, "earlier_ref"):
            issues.append(
                IntegrityIssue(
                    "temporal_annotations", case.case_id,
                    f"temporal fact {fact.fact_id} does not reference a trend entity",
                )
            )
            continue

        for ref_name in ("earlier_ref", "later_ref"):
            ref = getattr(trend, ref_name)
            referenced = index.get(ref)
            if not isinstance(referenced, LabResultEntry):
                issues.append(
                    IntegrityIssue(
                        "temporal_annotations", case.case_id,
                        f"trend {trend.entity_id} {ref_name}={ref} does not resolve to a lab result",
                    )
                )

        if fact.earlier_value == fact.later_value:
            issues.append(
                IntegrityIssue(
                    "temporal_annotations", case.case_id,
                    f"temporal fact {fact.fact_id} has identical earlier and later values",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Check 5: negation annotations match real negated content
# ---------------------------------------------------------------------------


def check_negation_annotations(case: BenchmarkCase) -> list[IntegrityIssue]:
    structured = case.structured_case
    if structured is None:
        return []
    index = structured.entity_index()
    issues: list[IntegrityIssue] = []

    for fact in case.negations:
        if fact.polarity is not FactPolarity.NEGATED:
            issues.append(
                IntegrityIssue(
                    "negation_annotations", case.case_id,
                    f"negation {fact.fact_id} is not marked NEGATED",
                )
            )
        entity = index.get(fact.source_ref or "")
        if isinstance(entity, ClinicalFindingEntry):
            if entity.present:
                issues.append(
                    IntegrityIssue(
                        "negation_annotations", case.case_id,
                        f"negation {fact.fact_id} references finding {entity.entity_id}, "
                        f"which is present=True",
                    )
                )
        elif isinstance(entity, ProcedureEntry):
            if entity.performed:
                issues.append(
                    IntegrityIssue(
                        "negation_annotations", case.case_id,
                        f"negation {fact.fact_id} references procedure {entity.entity_id}, "
                        f"which was performed",
                    )
                )
        else:
            issues.append(
                IntegrityIssue(
                    "negation_annotations", case.case_id,
                    f"negation {fact.fact_id} does not reference a negatable entity",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Check 6: the reference summary introduces nothing unsupported
# ---------------------------------------------------------------------------


def _supported_numbers(structured: StructuredCase) -> str:
    return structured.text_corpus()


def check_summary_supported(case: BenchmarkCase) -> list[IntegrityIssue]:
    """Verify the reference summary asserts nothing absent from the structured case.

    Three passes: drug names, condition names, and numeric values. Each is
    checked against the structured case rather than against the note, because
    the summary is generated independently of the note - the question is
    whether it invented anything, not whether it paraphrased well.
    """
    structured = case.structured_case
    if structured is None:
        return []

    summary = case.reference_summary
    lowered = summary.lower()
    issues: list[IntegrityIssue] = []

    # The structured corpus is the reference for all three passes. A drug or
    # condition named anywhere in the structured case is supported, including
    # inside a negated finding ("no history of osteoarthritis") - the question
    # is whether the renderer invented it, not whether it is an active problem.
    corpus = _supported_numbers(structured)

    for drug in content.all_drug_names():
        if drug.lower() in lowered and drug.lower() not in corpus:
            issues.append(
                IntegrityIssue(
                    "summary_supported", case.case_id,
                    f"summary names {drug}, which is absent from the structured case",
                )
            )

    for condition in content.all_condition_names():
        lowered_condition = condition.lower()
        if lowered_condition in lowered and lowered_condition not in corpus:
            issues.append(
                IntegrityIssue(
                    "summary_supported", case.case_id,
                    f"summary names condition '{condition}', absent from the structured case",
                )
            )

    for number in set(_NUMBER_PATTERN.findall(summary)):
        if number not in corpus:
            issues.append(
                IntegrityIssue(
                    "summary_supported", case.case_id,
                    f"summary contains the value '{number}', absent from the structured case",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Check 7 / 10: identity and privacy
# ---------------------------------------------------------------------------


def check_case_ids_unique(cases: Iterable[BenchmarkCase]) -> list[IntegrityIssue]:
    seen: set[str] = set()
    issues: list[IntegrityIssue] = []
    for case in cases:
        if case.case_id in seen:
            issues.append(
                IntegrityIssue("case_ids_unique", case.case_id, "duplicate case_id")
            )
        seen.add(case.case_id)
    return issues


def check_no_identifiable_information(case: BenchmarkCase) -> list[IntegrityIssue]:
    """Scan every free-text surface for patterns that would indicate PII.

    The generator has no source of real identifiers, so this asserts an
    invariant rather than sanitizing - a hit means something is wrong with the
    generator, not with the data.
    """
    haystack = "\n".join(
        [
            case.clinical_note,
            case.reference_summary,
            *[f.content for f in _all_facts(case)],
        ]
    )
    issues: list[IntegrityIssue] = []
    for label, pattern in _PII_PATTERNS:
        match = pattern.search(haystack)
        if match:
            issues.append(
                IntegrityIssue(
                    "no_identifiable_information", case.case_id,
                    f"text matches {label} pattern: '{match.group(0)}'",
                )
            )

    structured = case.structured_case
    if structured is not None:
        if not structured.patient.patient_id.startswith("SYN-PT-"):
            issues.append(
                IntegrityIssue(
                    "no_identifiable_information", case.case_id,
                    f"patient_id '{structured.patient.patient_id}' is not a synthetic surrogate",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Structural sanity
# ---------------------------------------------------------------------------


def check_case_structure(case: BenchmarkCase) -> list[IntegrityIssue]:
    issues: list[IntegrityIssue] = []
    if not case.clinical_note.strip():
        issues.append(IntegrityIssue("case_structure", case.case_id, "empty clinical note"))
    if not case.reference_summary.strip():
        issues.append(IntegrityIssue("case_structure", case.case_id, "empty reference summary"))
    if len(case.reference_summary) >= len(case.clinical_note):
        issues.append(
            IntegrityIssue(
                "case_structure", case.case_id,
                "reference summary is not shorter than the source note",
            )
        )

    fact_ids = [f.fact_id for f in _all_facts(case)]
    # Critical negations and contradictions are intentionally filed in two
    # buckets, so compare against the de-duplicated set of fact objects.
    unique_ids = set(fact_ids)
    counts = {fact_id: fact_ids.count(fact_id) for fact_id in unique_ids}
    for fact_id, count in counts.items():
        if count > 2:
            issues.append(
                IntegrityIssue(
                    "case_structure", case.case_id,
                    f"fact id {fact_id} appears {count} times",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Check 9: metadata matches content
# ---------------------------------------------------------------------------


def check_no_duplicate_measurements(case: BenchmarkCase) -> list[IntegrityIssue]:
    """Reject two different values for the same analyte at the same timepoint.

    A case reporting both "Sodium 125 today" and "Sodium 110 today" is
    incoherent, and an evaluator comparing a summary against it could not
    decide which value retention should be measured against.
    """
    structured = case.structured_case
    if structured is None:
        return []

    seen: dict[tuple[str, int], float] = {}
    issues: list[IntegrityIssue] = []
    for lab in structured.labs:
        key = (lab.name, lab.day_offset)
        if key in seen and seen[key] != lab.value:
            issues.append(
                IntegrityIssue(
                    "no_duplicate_measurements", case.case_id,
                    f"{lab.name} has conflicting values {seen[key]} and {lab.value} at the "
                    f"same timepoint ({lab.timing_description})",
                )
            )
        seen[key] = lab.value
    return issues


def check_metadata_matches(
    cases: list[BenchmarkCase], metadata: dict[str, Any]
) -> list[IntegrityIssue]:
    """Verify recorded metadata actually describes the cases it ships with."""
    from app.benchmark.synthesizer import build_metadata
    from app.core.benchmark_config import BenchmarkConfig

    issues: list[IntegrityIssue] = []
    if metadata.get("case_count") != len(cases):
        issues.append(
            IntegrityIssue(
                "metadata_matches", "-",
                f"metadata case_count={metadata.get('case_count')} but {len(cases)} cases present",
            )
        )

    config = BenchmarkConfig(version=metadata.get("benchmark_version", ""))
    recomputed = build_metadata(cases, metadata.get("seed", ""), config)

    for section in ("distribution", "counts", "cases_containing", "averages"):
        if metadata.get(section) != recomputed[section]:
            issues.append(
                IntegrityIssue(
                    "metadata_matches", "-",
                    f"metadata section '{section}' does not match the cases it describes",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

PER_CASE_CHECKS = (
    check_annotations_reference_source,
    check_critical_facts_grounded,
    check_medication_annotations,
    check_temporal_annotations,
    check_negation_annotations,
    check_summary_supported,
    check_no_identifiable_information,
    check_no_duplicate_measurements,
    check_case_structure,
)


def validate_case(case: BenchmarkCase) -> list[IntegrityIssue]:
    issues: list[IntegrityIssue] = []
    for check in PER_CASE_CHECKS:
        issues.extend(check(case))
    return issues


def validate_benchmark(
    cases: list[BenchmarkCase], metadata: dict[str, Any] | None = None
) -> list[IntegrityIssue]:
    """Run every integrity check over a whole benchmark."""
    issues: list[IntegrityIssue] = list(check_case_ids_unique(cases))
    for case in cases:
        issues.extend(validate_case(case))
    if metadata is not None:
        issues.extend(check_metadata_matches(cases, metadata))
    return issues
