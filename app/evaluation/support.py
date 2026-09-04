"""Entity-level support analysis.

Given one structured entity and a model's output text, decide whether the
output SUPPORTS, PARTIALLY supports, OMITS, or CONTRADICTS it - and carry the
evidence for that decision.

This is the shared engine underneath every evaluator. Working from the
*structured entity* rather than from an annotation's content string is the
whole point: a medication's status is compared against `MedicationStatus`, not
against the words "was discontinued", so the evaluator is testing clinical
content rather than phrasing.

Adding a new entity type means adding one analyzer function and registering
it. No evaluator changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from app.core.enums import (
    AbnormalFlag,
    ConditionStatus,
    MatchLevel,
    MedicationStatus,
    SupportStatus,
    TemporalTrend,
    ValueComparison,
)
from app.evaluation.matching import FactMatcher, MatchResult
from app.evaluation.normalization import (
    MEDICATION_STATUS_CUES,
    normalize_text,
    sentences_containing,
)
from app.evaluation.numeric import (
    ValueComparisonResult,
    compare_blood_pressure,
    compare_term_value,
    compare_values,
)
from app.models.structured import (
    AllergyEntry,
    ClinicalFindingEntry,
    ConditionEntry,
    ConflictEntry,
    DiagnosisEntry,
    LabResultEntry,
    LabTrendEntry,
    MedicationEntry,
    ProcedureEntry,
    StructuredEntity,
    VitalSignEntry,
)

# How far around a drug mention to look for status cues. One clause, roughly.
_STATUS_WINDOW = 120

_LEADING_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_UNIT_WORD = re.compile(r"[a-zA-Z]+")

# Units that mark a number as a dose rather than a duration or a count.
_DOSE_UNITS = ("mg", "mcg", "g", "units", "unit", "ml", "mmol", "meq", "puffs", "puff")
_DOSE_QUANTITY = re.compile(
    rf"(\d+(?:\.\d+)?)\s*({'|'.join(_DOSE_UNITS)})\b"
)


def dose_quantities_near(drug_name: str, output: str) -> list[float]:
    """Dose-unit quantities stated in `drug_name`'s own sentences."""
    return [
        float(match.group(1))
        for sentence in sentences_containing(drug_name, output)
        for match in _DOSE_QUANTITY.finditer(sentence)
    ]


@dataclass
class SupportResult:
    """Whether model output supports one structured entity, and why."""

    status: SupportStatus
    entity_id: str
    entity_kind: str
    reason: str
    source_evidence: str = ""
    output_evidence: str | None = None
    match_level: MatchLevel = MatchLevel.NONE
    confidence: float = 1.0
    value_comparison: ValueComparisonResult | None = None
    details: dict[str, object] = field(default_factory=dict)

    @property
    def is_problem(self) -> bool:
        return self.status in {SupportStatus.OMITTED, SupportStatus.CONTRADICTED}


def _result(
    status: SupportStatus,
    entity: StructuredEntity,
    kind: str,
    reason: str,
    source_evidence: str,
    match: MatchResult | None = None,
    **extra,
) -> SupportResult:
    return SupportResult(
        status=status,
        entity_id=entity.entity_id,
        entity_kind=kind,
        reason=reason,
        source_evidence=source_evidence,
        output_evidence=match.evidence if match else None,
        match_level=match.level if match else MatchLevel.NONE,
        confidence=match.confidence if match else 1.0,
        **extra,
    )


# ---------------------------------------------------------------------------
# Per-entity analyzers
# ---------------------------------------------------------------------------


def analyze_condition(
    entity: ConditionEntry, output: str, matcher: FactMatcher, case=None
) -> SupportResult:
    """Conditions: present and affirmed, or negated (contradiction), or absent.

    A resolved condition presented as current is a
    historical/current confusion, reported as CONTRADICTED because the output
    asserts something the source denies.
    """
    match = matcher.match(entity.name, output)
    source = f"{entity.name} ({entity.status.value})"

    if not match.matched:
        return _result(SupportStatus.OMITTED, entity, "condition", "not mentioned in output", source)

    if entity.status is ConditionStatus.ACTIVE:
        if match.negated:
            return _result(
                SupportStatus.CONTRADICTED, entity, "condition",
                f"source lists {entity.name} as active; output negates it "
                f"(cue: '{match.negation_cue}')",
                source, match,
            )
        return _result(SupportStatus.SUPPORTED, entity, "condition", "mentioned and affirmed", source, match)

    # Resolved or historical: the output must not present it as current.
    if _asserts_historical(entity.name, output) or match.negated:
        return _result(
            SupportStatus.SUPPORTED, entity, "condition",
            "correctly presented as not current", source, match,
        )
    return _result(
        SupportStatus.CONTRADICTED, entity, "condition",
        f"source lists {entity.name} as {entity.status.value}; output presents it without "
        f"any indication that it is not current",
        source, match, details={"error": "historical_current_confusion"},
    )


def _asserts_historical(term: str, output: str) -> bool:
    """True when `term`'s mention is framed as past rather than current."""
    historical_words = MEDICATION_STATUS_CUES["historical"] + (
        "resolved", "no longer active", "not active", "past",
    )
    return any(
        any(word in sentence for word in historical_words)
        for sentence in sentences_containing(term, output)
    )


def analyze_diagnosis(
    entity: DiagnosisEntry, output: str, matcher: FactMatcher, case=None
) -> SupportResult:
    match = matcher.match(entity.name, output)
    source = f"{'Primary diagnosis' if entity.is_primary else 'Diagnosis'}: {entity.name}"
    if not match.matched:
        return _result(SupportStatus.OMITTED, entity, "diagnosis", "not mentioned in output", source)
    if match.negated:
        return _result(
            SupportStatus.CONTRADICTED, entity, "diagnosis",
            f"source states the diagnosis {entity.name}; output negates it",
            source, match,
        )
    return _result(SupportStatus.SUPPORTED, entity, "diagnosis", "mentioned and affirmed", source, match)


def detect_medication_status(drug_name: str, output: str) -> str | None:
    """Infer how the output frames a drug, from cues in the drug's own sentence.

    Returns a `MedicationStatus` value, or None when the output names the drug
    with no status cue at all.

    Two things matter here. Cues are read only from sentences mentioning this
    drug, so a neighbouring sentence about a different drug cannot contribute
    its status. And the check order is significant: 'discontinued' before
    'current' because "no longer taking" contains "taking", and 'historical'
    before 'current' because "not a current medication" contains "current".
    """
    for sentence in sentences_containing(drug_name, output):
        for status in ("discontinued", "historical", "changed_dose", "new", "current"):
            if any(cue in sentence for cue in MEDICATION_STATUS_CUES[status]):
                return status
    return None


def analyze_medication(
    entity: MedicationEntry, output: str, matcher: FactMatcher, case=None
) -> SupportResult:
    """Medications: presence, then status, then dose.

    Checked in that order because the errors compound - a drug that is absent
    cannot have a wrong dose, and reporting both would double-count one
    failure.
    """
    match = matcher.match(entity.drug_name, output)
    source = f"{entity.describe()} [{entity.status.value}]"

    if not match.matched:
        return _result(SupportStatus.OMITTED, entity, "medication", "drug not mentioned in output", source)

    if match.negated and entity.status is not MedicationStatus.DISCONTINUED:
        return _result(
            SupportStatus.CONTRADICTED, entity, "medication",
            f"source lists {entity.drug_name} as {entity.status.value}; output negates it",
            source, match, details={"error": "status"},
        )

    reported_status = detect_medication_status(entity.drug_name, output)
    expected = entity.status.value

    if reported_status is not None and not _status_compatible(expected, reported_status):
        return _result(
            SupportStatus.CONTRADICTED, entity, "medication",
            f"source status is '{expected}'; output presents it as '{reported_status}'",
            source, match,
            details={"error": "status", "expected": expected, "reported": reported_status},
        )

    if reported_status is None and entity.is_change:
        return _result(
            SupportStatus.PARTIAL, entity, "medication",
            f"{entity.drug_name} is a {expected} medication but the output states no status",
            source, match,
            details={"error": "status", "expected": expected, "reported": None},
        )

    dose_result = _check_dose(entity, output)
    if dose_result is not None:
        return _result(
            SupportStatus.CONTRADICTED, entity, "medication",
            f"dose mismatch for {entity.drug_name}: {dose_result.describe()}",
            source, match,
            details={"error": "dose"}, value_comparison=dose_result,
        )

    return _result(
        SupportStatus.SUPPORTED, entity, "medication",
        "drug present with a compatible status and dose", source, match,
    )


def _status_compatible(expected: str, reported: str) -> bool:
    """Whether an output's framing of a drug is acceptable for its true status.

    Not strict equality. A CHANGED_DOSE drug described as 'current' is
    acceptable - it *is* current - so long as the dose is right; the dose check
    catches a summary that reports the old dose. A DISCONTINUED drug described
    as 'current' is never acceptable, which is the error that risks duplicate
    therapy.
    """
    if expected == reported:
        return True
    acceptable: dict[str, set[str]] = {
        "current": {"current", "changed_dose"},
        "new": {"new", "current"},
        "changed_dose": {"changed_dose", "current", "new"},
        "discontinued": {"discontinued"},
        "historical": {"historical", "discontinued"},
    }
    return reported in acceptable.get(expected, {expected})


def _check_dose(entity: MedicationEntry, output: str) -> ValueComparisonResult | None:
    """Return a comparison result only when the dose is demonstrably wrong.

    Compares against quantities that carry a *dose unit*, not against any
    number near the drug name. Without that restriction, "Insulin glargine was
    discontinued 2 weeks ago" reads the "2" as a dose and reports a 30-units
    prescription as wrong - a false positive found by oracle testing.

    Silence when the output states no dose at all: omitting a dose is not the
    same error as stating the wrong one, and the retention evaluator already
    accounts for missing content.
    """
    if not entity.dose:
        return None
    dose_value = _dose_number(entity.dose)
    if dose_value is None:
        return None

    candidates = dose_quantities_near(entity.drug_name, output)
    comparison = compare_values(
        dose_value, candidates, unit=_dose_unit(entity.dose),
        evidence=entity.describe(),
    )
    if comparison.comparison is ValueComparison.OMITTED:
        return None
    if comparison.comparison is ValueComparison.INCORRECT:
        # A dose change legitimately names the prior dose too. Only the
        # current dose being absent is an error; both being present is
        # correct reporting of the change.
        if entity.prior_dose and dose_value in comparison.candidates:
            return None
        return comparison
    return None


def _dose_number(dose: str) -> float | None:
    match = _LEADING_NUMBER.search(dose)
    return float(match.group(0)) if match else None


def _dose_unit(dose: str) -> str | None:
    match = _UNIT_WORD.search(dose)
    return match.group(0) if match else None


def _is_repeated_measurement(name: str, entities: list, attribute: str) -> bool:
    """True when a case holds more than one measurement of the same thing.

    A case with a creatinine series (admission, day 2, discharge) reports
    several values for one analyte. A summary that quotes one of them is
    correct, and the evaluator cannot attribute a stated value to a specific
    timepoint without temporal parsing it does not do. So for a series, a
    value that is not found is reported as OMITTED, never CONTRADICTED - the
    evaluator declines to accuse where it cannot attribute.
    """
    return sum(1 for entity in entities if getattr(entity, attribute) == name) > 1


def analyze_lab(
    entity: LabResultEntry, output: str, matcher: FactMatcher, case=None
) -> SupportResult:
    """Labs: the analyte must be named and this value must be present."""
    match = matcher.match(entity.name, output)
    source = f"{entity.describe()} ({entity.timing_description})"

    if not match.matched:
        return _result(SupportStatus.OMITTED, entity, "lab", "analyte not mentioned in output", source)

    from app.evaluation.numeric import values_near_term

    candidates = [item.value for item in values_near_term(entity.name, output)]
    comparison = compare_values(
        entity.value, candidates, unit=entity.unit, evidence=match.evidence or "",
    )

    if entity.value in candidates:
        return _result(
            SupportStatus.SUPPORTED, entity, "lab",
            f"{entity.name} reported with the correct value", source, match,
            value_comparison=comparison,
        )

    if not candidates:
        return _result(
            SupportStatus.PARTIAL, entity, "lab",
            f"{entity.name} is named but no value is stated", source, match,
            value_comparison=comparison,
        )

    if comparison.comparison is ValueComparison.APPROXIMATE:
        return _result(
            SupportStatus.SUPPORTED, entity, "lab",
            f"{entity.name} reported with an approximate value", source, match,
            value_comparison=comparison,
        )

    if case is not None and _is_repeated_measurement(entity.name, case.labs, "name"):
        # A series records several values for one analyte, and the evaluator
        # cannot attribute a stated value to a timepoint. But a value matching
        # *none* of the series is still a claim the source never makes, so it
        # is a contradiction of the analyte rather than an unattributable
        # omission.
        series_values = {lab.value for lab in case.labs if lab.name == entity.name}
        if not series_values & set(candidates):
            return _result(
                SupportStatus.CONTRADICTED, entity, "lab",
                f"output states a {entity.name} value that matches no measurement in this "
                f"case: {comparison.describe()}",
                source, match, value_comparison=comparison,
            )
        return _result(
            SupportStatus.OMITTED, entity, "lab",
            f"this {entity.name} measurement ({entity.timing_description}) is not stated; "
            f"the case records several values for this analyte",
            source, match, value_comparison=comparison,
        )

    return _result(
        SupportStatus.CONTRADICTED, entity, "lab",
        f"{entity.name} value mismatch: {comparison.describe()}", source, match,
        value_comparison=comparison,
    )


def analyze_vital(
    entity: VitalSignEntry, output: str, matcher: FactMatcher, case=None
) -> SupportResult:
    """Vitals: blood pressure compares as a pair, everything else as a value."""
    match = matcher.match(entity.name, output)
    source = f"{entity.describe()} ({entity.timing_description})"

    if not match.matched:
        return _result(SupportStatus.OMITTED, entity, "vital", "vital sign not mentioned", source)

    if "/" in entity.value:
        comparison = compare_blood_pressure(entity.value, output)
    else:
        numeric = _dose_number(entity.value)
        if numeric is None:
            return _result(
                SupportStatus.SUPPORTED, entity, "vital", "non-numeric vital, presence only", source, match
            )
        comparison = compare_term_value(entity.name, numeric, output, unit=entity.unit)

    if comparison.comparison in {ValueComparison.CORRECT, ValueComparison.APPROXIMATE}:
        return _result(
            SupportStatus.SUPPORTED, entity, "vital", "reported correctly", source, match,
            value_comparison=comparison,
        )

    if comparison.comparison is ValueComparison.OMITTED:
        return _result(
            SupportStatus.PARTIAL, entity, "vital", "named without a value", source, match,
            value_comparison=comparison,
        )

    if case is not None and _is_repeated_measurement(entity.name, case.vitals, "name"):
        return _result(
            SupportStatus.OMITTED, entity, "vital",
            f"this {entity.name} reading is not stated; the case records several readings",
            source, match, value_comparison=comparison,
        )

    return _result(
        SupportStatus.CONTRADICTED, entity, "vital",
        f"{entity.name} mismatch: {comparison.describe()}", source, match,
        value_comparison=comparison,
    )


def analyze_allergy(
    entity: AllergyEntry, output: str, matcher: FactMatcher, case=None
) -> SupportResult:
    match = matcher.match(entity.substance, output)
    source = f"Allergy: {entity.describe()} ({entity.severity.value})"
    if not match.matched:
        return _result(SupportStatus.OMITTED, entity, "allergy", "allergen not mentioned", source)
    if match.negated:
        return _result(
            SupportStatus.CONTRADICTED, entity, "allergy",
            f"source documents an allergy to {entity.substance}; output negates it",
            source, match,
        )
    return _result(SupportStatus.SUPPORTED, entity, "allergy", "allergen present", source, match)


def analyze_finding(
    entity: ClinicalFindingEntry, output: str, matcher: FactMatcher, case=None
) -> SupportResult:
    """Findings: polarity must be preserved.

    A negated finding asserted as positive is the benchmark's signature
    failure - "no history of diabetes" becoming "patient has diabetes".

    Negative findings are matched on their *core term* rather than their full
    wording. Matching "no history of diabetes" literally would mean a reversal
    ("patient has a history of diabetes") failed to match at all and was
    reported as a mere omission - which is exactly backwards, since a
    fabricated diagnosis is worse than a dropped negative, not milder. The core
    term is matched instead, and polarity is then read from the output's own
    negation context.
    """
    polarity = "present" if entity.present else "ABSENT"
    source = f"{entity.description} [{polarity}]"

    if entity.present:
        match = matcher.match(entity.description, output)
        if not match.matched:
            return _result(
                SupportStatus.OMITTED, entity, "finding", "finding not mentioned in output", source
            )
        if match.negated and not _self_negating(entity.description):
            return _result(
                SupportStatus.CONTRADICTED, entity, "finding",
                f"source asserts '{entity.description}'; output negates it", source, match,
            )
        return _result(SupportStatus.SUPPORTED, entity, "finding", "finding preserved", source, match)

    core = _negation_core(entity.description)
    match = matcher.match(core, output)

    if not match.matched:
        return _result(
            SupportStatus.OMITTED, entity, "finding",
            f"the negative finding '{entity.description}' is not mentioned in the output",
            source,
        )

    if match.negated:
        return _result(
            SupportStatus.SUPPORTED, entity, "finding", "negative finding preserved", source, match
        )

    return _result(
        SupportStatus.CONTRADICTED, entity, "finding",
        f"source states '{entity.description}'; output asserts '{core}' as present",
        source, match, details={"error": "negation_reversal"},
    )


_NEGATION_PREFIXES = (
    "no known history of ", "no history of ", "no evidence of ", "no known ",
    "negative for ", "without ", "denies ", "denied ", "not ", "no ",
)


def _negation_core(description: str) -> str:
    """Strip a negative finding's leading negation, leaving the clinical term.

    "no history of diabetes" -> "diabetes". Longest prefix wins, so
    "no known history of X" is not left as "known history of X".
    """
    normalized = normalize_text(description)
    for prefix in _NEGATION_PREFIXES:
        if normalized.startswith(prefix):
            remainder = normalized[len(prefix) :].strip()
            if remainder:
                return remainder
    return normalized


def _self_negating(description: str) -> bool:
    """True when a finding's own wording carries the negation."""
    normalized = normalize_text(description)
    return normalized.startswith(("no ", "not ", "without ", "denies ", "negative for "))


def analyze_procedure(
    entity: ProcedureEntry, output: str, matcher: FactMatcher, case=None
) -> SupportResult:
    """Procedures: a study not performed must not be reported as done."""
    match = matcher.match(entity.name, output)
    source = f"{entity.name} [{'performed' if entity.performed else 'NOT performed'}]"

    if not match.matched:
        # Omitting a study that was never done is harmless; omitting one that
        # was performed loses content.
        status = SupportStatus.OMITTED if entity.performed else SupportStatus.SUPPORTED
        reason = "procedure not mentioned" if entity.performed else "correctly absent (never performed)"
        return _result(status, entity, "procedure", reason, source)

    if entity.performed:
        if match.negated:
            return _result(
                SupportStatus.CONTRADICTED, entity, "procedure",
                f"source states {entity.name} was performed; output negates it", source, match,
            )
        return _result(SupportStatus.SUPPORTED, entity, "procedure", "procedure reported", source, match)

    if match.negated:
        return _result(
            SupportStatus.SUPPORTED, entity, "procedure",
            "correctly reported as not performed", source, match,
        )
    return _result(
        SupportStatus.CONTRADICTED, entity, "procedure",
        f"source states {entity.name} was NOT performed; output reports it without negation",
        source, match, details={"error": "hallucinated_procedure"},
    )


def analyze_trend(
    entity: LabTrendEntry, output: str, matcher: FactMatcher, case=None
) -> SupportResult:
    """Trends: both endpoints and the direction must survive.

    Three distinct failures, distinguished because they mean different things:
    losing an endpoint (information loss), reporting the historical value as
    current (value confusion), and stating the wrong direction (reversal).
    """
    match = matcher.match(entity.lab_name, output)
    source = entity.describe()

    if not match.matched:
        return _result(SupportStatus.OMITTED, entity, "trend", "analyte not mentioned", source)

    from app.evaluation.numeric import values_near_term

    candidates = [item.value for item in values_near_term(entity.lab_name, output)]
    has_earlier = any(value == entity.earlier_value for value in candidates)
    has_later = any(value == entity.later_value for value in candidates)
    linked_sentence = _linked_endpoint_sentence(output, entity)

    if linked_sentence is not None:
        direction = _direction_in(linked_sentence)
        if direction is not None and direction is not entity.trend:
            return _result(
                SupportStatus.CONTRADICTED, entity, "trend",
                f"source trend is {entity.trend.value}; output states {direction.value}",
                source, match, details={"error": "trend_reversal"},
            )
        return _result(SupportStatus.SUPPORTED, entity, "trend", "both endpoints preserved", source, match)

    if has_earlier and not has_later:
        return _result(
            SupportStatus.CONTRADICTED, entity, "trend",
            f"output states the historical {entity.lab_name} "
            f"({entity.format_value(entity.earlier_value)} {entity.unit}) but not the current value "
            f"({entity.format_value(entity.later_value)} {entity.unit})",
            source, match, details={"error": "value_confusion"},
        )

    if has_later and not has_earlier:
        return _result(
            SupportStatus.PARTIAL, entity, "trend",
            f"output states the current {entity.lab_name} but not the prior value, losing the trend",
            source, match, details={"error": "information_loss"},
        )

    return _result(
        SupportStatus.PARTIAL, entity, "trend",
        f"{entity.lab_name} is named but neither trend endpoint is stated",
        source, match, details={"error": "information_loss"},
    )


_TREND_WORDS: dict[TemporalTrend, tuple[str, ...]] = {
    TemporalTrend.INCREASING: ("rose", "increased", "increasing", "risen", "rising", "up from", "higher"),
    TemporalTrend.DECREASING: ("fell", "decreased", "decreasing", "fallen", "falling", "down from", "lower", "improved to"),
    TemporalTrend.STABLE: ("stable", "unchanged", "remained"),
}


# Maximum characters between a trend's two endpoint values for them to count
# as one statement. Covers "from 1.4 to 1.9" and "was 1.4 three months ago and
# is now 1.9"; excludes two values that merely co-occur in a long sentence.
_ENDPOINT_LINK_DISTANCE = 40


def _standalone_positions(sentence: str, value: str) -> list[tuple[int, int]]:
    """Spans where `value` appears as a whole numeric token.

    Token-bounded so that looking for "2" does not match inside "3.1" or
    "1.73m2" - though it does still match a genuine standalone "2" such as the
    one in "hospital day 2", which is why linkage distance is also required.
    """
    pattern = re.compile(rf"(?<![\d.]){re.escape(value)}(?![\d.])")
    return [(match.start(), match.end()) for match in pattern.finditer(sentence)]


def _linked_endpoint_sentence(output: str, entity: LabTrendEntry) -> str | None:
    """A sentence stating both of this trend's endpoints as one statement.

    Three constraints, each fixing a false positive found by oracle testing:

    - **Sentence-bounded**, because a fixed character window reached into the
      next sentence and read another analyte's direction word.
    - **Token-bounded**, because substring matching found the "2" of
      "hospital day 2" and treated it as a trend endpoint.
    - **Linkage-bounded**, because a case with a creatinine series holds
      several trends over one analyte, and co-occurrence alone attributed the
      first trend's sentence to all of them.

    Returning None means the output states no single-statement trend for this
    entity, which is information loss - not a reversal.
    """
    earlier = entity.format_value(entity.earlier_value)
    later = entity.format_value(entity.later_value)

    for sentence in sentences_containing(entity.lab_name, output):
        earlier_spans = _standalone_positions(sentence, earlier)
        later_spans = _standalone_positions(sentence, later)
        for earlier_start, earlier_end in earlier_spans:
            for later_start, later_end in later_spans:
                # Either order counts as one statement: "rose from 1.4 to 1.9"
                # and "rose to 1.9 from 1.4" both state this trend. Order alone
                # does not decide correctness - the direction word does - so
                # accepting both is what lets a genuine reversal ("fell from
                # 3.1 to 1.1" for a rising trend) be caught rather than
                # falling through as information loss.
                gap = min(
                    abs(later_start - earlier_end), abs(earlier_start - later_end)
                )
                if gap <= _ENDPOINT_LINK_DISTANCE:
                    return sentence
    return None


def _direction_in(sentence: str) -> TemporalTrend | None:
    """The direction word stated in an already-attributed sentence."""
    for trend, words in _TREND_WORDS.items():
        if any(word in sentence for word in words):
            return trend
    return None


def analyze_conflict(
    entity: ConflictEntry, output: str, matcher: FactMatcher, case=None
) -> SupportResult:
    """Conflicts: the later statement wins; reporting the earlier one is wrong."""
    source = (
        f"'{entity.earlier_statement}' ({entity.earlier_timing}) superseded by "
        f"'{entity.later_statement}' ({entity.later_timing})"
    )
    later = matcher.match(entity.later_statement, output)
    earlier = matcher.match(entity.earlier_statement, output)

    if later.matched and not later.negated:
        return _result(SupportStatus.SUPPORTED, entity, "conflict", "later statement preserved", source, later)
    if earlier.matched and not earlier.negated:
        return _result(
            SupportStatus.CONTRADICTED, entity, "conflict",
            f"output reports the superseded statement about {entity.topic} as current",
            source, earlier, details={"error": "conflict_resolution"},
        )
    return _result(
        SupportStatus.OMITTED, entity, "conflict",
        f"the resolved conflict about {entity.topic} is not represented", source,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

Analyzer = Callable[..., SupportResult]

# Order matters: MedicationEntry and the specialized lab types are checked
# before their bases would be.
_ANALYZERS: tuple[tuple[type, Analyzer], ...] = (
    (MedicationEntry, analyze_medication),
    (LabTrendEntry, analyze_trend),
    (LabResultEntry, analyze_lab),
    (VitalSignEntry, analyze_vital),
    (AllergyEntry, analyze_allergy),
    (ClinicalFindingEntry, analyze_finding),
    (ProcedureEntry, analyze_procedure),
    (DiagnosisEntry, analyze_diagnosis),
    (ConditionEntry, analyze_condition),
    (ConflictEntry, analyze_conflict),
)


def analyze_entity(
    entity: StructuredEntity,
    output: str,
    matcher: FactMatcher | None = None,
    case=None,
) -> SupportResult | None:
    """Analyze any supported entity, or None for an unsupported type.

    `case` is the entity's `StructuredCase`, which analyzers use to see an
    entity's siblings - a lab needs to know whether it belongs to a series
    before it can decide that a differing value is a contradiction rather than
    an unattributable omission. Optional so a single entity can still be
    analyzed in isolation in tests.

    Returning None rather than raising means a new entity type added to the
    structured model does not crash evaluation before its analyzer is written -
    it is simply not evaluated, and the fact-support evaluator reports the gap.
    """
    matcher = matcher or FactMatcher()
    for entity_type, analyzer in _ANALYZERS:
        if isinstance(entity, entity_type):
            return analyzer(entity, output, matcher, case)
    return None


def supported_entity_kinds() -> tuple[str, ...]:
    return tuple(entity_type.__name__ for entity_type, _ in _ANALYZERS)
