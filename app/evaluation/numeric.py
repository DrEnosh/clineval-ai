"""Clinical numeric value parsing and comparison.

Handles the forms clinical text actually uses:

    1.4 mg/dL          creatinine 1.4        creatinine of 1.4
    BP 168/94          168/94 mmHg           heart rate 102 bpm
    14 K/uL            SpO2 94%              temperature 101.2 F

Tolerance policy, stated explicitly because "close enough" is a clinical
judgment and should not be buried in code:

- **CORRECT** - values equal after normalization. 1.9 == 1.90 == 1.9000.
- **APPROXIMATE** - values differ but within `relative_tolerance` (default
  2%). This exists for rounding ("creatinine 1.9" vs "creatinine 1.94"),
  not for genuine disagreement. 2% is tight on purpose: on a creatinine of
  1.9 it permits +/-0.04, which covers rounding and nothing else.
- **INCORRECT** - values differ beyond tolerance. A summary reporting a
  potassium of 4.1 when the source says 5.7 is wrong, not approximate.
- **OMITTED** - no value for this analyte found in the output.

Why not a wider tolerance? Because the errors this platform is built to catch
are exactly the ones that look small numerically. Reporting 1.4 instead of 1.9
is a 26% difference that a loose tolerance would still flag, but reporting the
*previous* value instead of the current one is the failure mode that matters,
and it is caught by comparison against the right source value rather than by
tolerance at all.

Blood pressure is compared as an ordered pair; both components must match for
the reading to be CORRECT, because a summary that keeps the systolic and drops
or alters the diastolic has changed the reading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.enums import ValueComparison
from app.evaluation.normalization import normalize_text, split_sentences, surface_forms

DEFAULT_RELATIVE_TOLERANCE = 0.02

_NUMBER = r"\d+(?:\.\d+)?"
_BP_PATTERN = re.compile(rf"\b({_NUMBER})\s*/\s*({_NUMBER})\b")
_NUMBER_PATTERN = re.compile(rf"\b{_NUMBER}\b")

# How many characters after a term to search for its value. Wide enough for
# "creatinine has risen from 1.4 to 1.9 mg/dL", narrow enough not to capture
# the next sentence's numbers.
_VALUE_WINDOW = 90


@dataclass(frozen=True)
class ParsedValue:
    """A numeric value found in text, with the span it came from."""

    value: float
    raw: str
    context: str

    def format(self) -> str:
        return str(int(self.value)) if self.value == int(self.value) else f"{self.value:g}"


@dataclass
class ValueComparisonResult:
    """The outcome of comparing a source value against model output.

    Carries everything a finding needs to explain itself: both values, the
    unit, the tolerance applied, and the text the output value came from.
    """

    comparison: ValueComparison
    source_value: float | None = None
    output_value: float | None = None
    unit: str | None = None
    tolerance: float = DEFAULT_RELATIVE_TOLERANCE
    evidence: str = ""
    candidates: list[float] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return self.comparison is ValueComparison.INCORRECT

    def describe(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        if self.comparison is ValueComparison.OMITTED:
            return f"expected {_fmt(self.source_value)}{unit}, no value found in output"
        return (
            f"expected {_fmt(self.source_value)}{unit}, "
            f"output states {_fmt(self.output_value)}{unit}"
        )


def _fmt(value: float | None) -> str:
    if value is None:
        return "none"
    return str(int(value)) if value == int(value) else f"{value:g}"


def parse_numbers(text: str) -> list[ParsedValue]:
    """Every standalone number in `text`, in order of appearance."""
    return [
        ParsedValue(value=float(match.group(0)), raw=match.group(0), context=text)
        for match in _NUMBER_PATTERN.finditer(text)
    ]


def parse_blood_pressure(text: str) -> tuple[float, float] | None:
    """The first systolic/diastolic pair in `text`, if any."""
    match = _BP_PATTERN.search(text)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def values_near_term(term: str, text: str, window: int = _VALUE_WINDOW) -> list[ParsedValue]:
    """Numbers appearing after any surface form of `term`, within its sentence.

    Bounded to the sentence containing the term. Searching near the term is
    what stops a creatinine comparison from matching a potassium value
    elsewhere; bounding to the sentence is what stops it reading the *next*
    statement's values.
    """
    found: list[ParsedValue] = []
    seen: set[tuple[str, float]] = set()

    for sentence in split_sentences(text):
        normalized = normalize_text(sentence)
        for form in sorted(surface_forms(term), key=len, reverse=True):
            if not form or form not in normalized:
                continue
            start = normalized.find(form)
            while start != -1:
                segment = normalized[start + len(form) : start + len(form) + window]
                for match in _NUMBER_PATTERN.finditer(segment):
                    key = (normalized, float(match.group(0)))
                    if key in seen:
                        continue
                    seen.add(key)
                    found.append(
                        ParsedValue(
                            value=float(match.group(0)), raw=match.group(0),
                            context=normalized.strip(),
                        )
                    )
                start = normalized.find(form, start + 1)
    return found


def compare_values(
    source_value: float,
    candidates: list[float],
    unit: str | None = None,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    evidence: str = "",
) -> ValueComparisonResult:
    """Compare a source value against candidate values found in output.

    Picks the *closest* candidate rather than the first: a summary mentioning
    several values for one analyte (a trend, "from 1.4 to 1.9") should be
    judged against the one it got right, so that reporting both endpoints is
    not scored as an error. Whether reporting only the historical value is a
    temporal failure is a separate question, answered by the temporal
    evaluator against the correct endpoint.
    """
    if not candidates:
        return ValueComparisonResult(
            comparison=ValueComparison.OMITTED, source_value=source_value,
            unit=unit, tolerance=relative_tolerance, evidence=evidence,
        )

    closest = min(candidates, key=lambda value: abs(value - source_value))
    result = ValueComparisonResult(
        source_value=source_value, output_value=closest, unit=unit,
        tolerance=relative_tolerance, evidence=evidence, candidates=list(candidates),
        comparison=ValueComparison.NOT_COMPARABLE,
    )

    if closest == source_value:
        result.comparison = ValueComparison.CORRECT
        return result

    scale = abs(source_value) if source_value else 1.0
    if abs(closest - source_value) <= scale * relative_tolerance:
        result.comparison = ValueComparison.APPROXIMATE
    else:
        result.comparison = ValueComparison.INCORRECT
    return result


def compare_term_value(
    term: str,
    source_value: float,
    text: str,
    unit: str | None = None,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> ValueComparisonResult:
    """Find `term`'s value in `text` and compare it against `source_value`."""
    found = values_near_term(term, text)
    evidence = found[0].context if found else ""
    return compare_values(
        source_value, [item.value for item in found], unit=unit,
        relative_tolerance=relative_tolerance, evidence=evidence,
    )


def compare_blood_pressure(
    source_reading: str, text: str
) -> ValueComparisonResult:
    """Compare a systolic/diastolic reading as an ordered pair.

    Both components must match for CORRECT. A summary that keeps the systolic
    and alters the diastolic has changed the reading, so partial agreement is
    reported as INCORRECT rather than APPROXIMATE.
    """
    source_pair = parse_blood_pressure(source_reading)
    if source_pair is None:
        return ValueComparisonResult(comparison=ValueComparison.NOT_COMPARABLE, unit="mmHg")

    systolic, diastolic = source_pair
    normalized = normalize_text(text)
    result = ValueComparisonResult(
        comparison=ValueComparison.OMITTED, source_value=systolic,
        unit="mmHg", evidence=source_reading,
    )

    for match in _BP_PATTERN.finditer(normalized):
        output_pair = (float(match.group(1)), float(match.group(2)))
        result.output_value = output_pair[0]
        result.evidence = match.group(0)
        if output_pair == source_pair:
            result.comparison = ValueComparison.CORRECT
            return result
        result.comparison = ValueComparison.INCORRECT
    return result
