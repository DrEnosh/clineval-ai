"""Selecting the experiment's case sample.

The Phase 5 experiment runs on 10 cases, not 100. That is a deliberate
constraint: the first real-model experiment should be cheap enough to run
repeatedly and small enough that a human can read every output. Scaling comes
after the methodology is understood, not before.

The sample is **stratified, not random**. Ten random cases would probably miss
several failure modes entirely, and the point of the experiment is to see how
models behave across the failure modes the benchmark was built to probe. So one
case is chosen per required stratum:

    simple                  critical omission opportunity
    hallucination opportunity   negation
    temporal trend          medication change
    multimorbidity          high complexity
    conflict resolution     adversarial

Selection is deterministic given a benchmark and seed, so every model is
evaluated on exactly the same cases and a re-run reproduces the sample.

A stratum with no matching case is reported as **unfilled** rather than
quietly back-filled. Silently substituting a different case would mean the
experiment claims coverage it does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.core.enums import DifficultyTier, FailureModeCategory, FindingType
from app.models.schemas import BenchmarkCase

Predicate = Callable[[BenchmarkCase], bool]


def _has(category: FailureModeCategory) -> Predicate:
    return lambda case: category in case.categories


def _has_any(*categories: FailureModeCategory) -> Predicate:
    wanted = set(categories)
    return lambda case: bool(wanted & set(case.categories))


def _is_simple(case: BenchmarkCase) -> bool:
    return case.difficulty is DifficultyTier.EASY


def _is_adversarial(case: BenchmarkCase) -> bool:
    return case.difficulty is DifficultyTier.ADVERSARIAL


@dataclass(frozen=True)
class Stratum:
    """One required slice of the experiment sample."""

    name: str
    description: str
    predicate: Predicate


#: The strata the experiment must cover, in report order.
STRATA: tuple[Stratum, ...] = (
    Stratum("simple", "Straightforward documentation, few facts", _is_simple),
    Stratum(
        "critical_omission",
        "Carries a panic-range value a summary must not drop",
        _has(FailureModeCategory.CRITICAL_OMISSION),
    ),
    Stratum(
        "hallucination_opportunity",
        "Documents a gap a model may fill with invented content",
        _has(FailureModeCategory.HALLUCINATION_OPPORTUNITY),
    ),
    Stratum(
        "negation",
        "Contains a clinically important negative",
        _has(FailureModeCategory.NEGATION),
    ),
    Stratum(
        "temporal_trend",
        "Contains a laboratory trend with both endpoints",
        _has(FailureModeCategory.TEMPORAL_PROGRESSION),
    ),
    Stratum(
        "medication_change",
        "Contains a started, stopped, or dose-changed medication",
        _has_any(
            FailureModeCategory.MEDICATION_CHANGE,
            FailureModeCategory.MEDICATION_DISCONTINUATION,
            FailureModeCategory.MEDICATION_DOSE_CHANGE,
        ),
    ),
    Stratum(
        "multimorbidity",
        "Several concurrent chronic conditions",
        _has(FailureModeCategory.MULTIPLE_COMORBIDITIES),
    ),
    Stratum(
        "high_complexity",
        "Multi-day course with a value that rises then partly recovers",
        _has(FailureModeCategory.HIGH_COMPLEXITY),
    ),
    Stratum(
        "conflict_resolution",
        "An earlier statement superseded by a later reassessment",
        _has(FailureModeCategory.CONFLICTING_INFORMATION),
    ),
    Stratum(
        "adversarial",
        "Adversarial tier: combined negation, temporal, and medication traps",
        _is_adversarial,
    ),
)


@dataclass
class SampleSelection:
    """The chosen sample, with the stratum each case fills."""

    cases: list[BenchmarkCase] = field(default_factory=list)
    assignments: dict[str, str] = field(default_factory=dict)
    unfilled: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    @property
    def case_ids(self) -> list[str]:
        return [case.case_id for case in self.cases]

    @property
    def is_complete(self) -> bool:
        return not self.unfilled

    def describe(self) -> dict:
        return {
            "case_count": len(self.cases),
            "case_ids": self.case_ids,
            "strata": dict(self.assignments),
            "unfilled_strata": list(self.unfilled),
            "is_complete": self.is_complete,
        }


def select_experiment_sample(
    cases: list[BenchmarkCase], strata: tuple[Stratum, ...] = STRATA
) -> SampleSelection:
    """Choose one case per stratum, without reusing a case.

    Strata are filled in declaration order, and the first matching unused case
    wins. Cases arrive in a deterministic order from the generator, so the
    selection is reproducible without needing a random seed at all.

    Filling in order means an early stratum can consume a case a later one also
    wanted. That is acceptable and the strata are ordered accordingly - the
    narrow ones (conflict resolution, high complexity) sit late but match few
    cases, while broad ones like `simple` have many candidates.
    """
    selection = SampleSelection()
    used: set[str] = set()

    for stratum in strata:
        match = next(
            (
                case
                for case in cases
                if case.case_id not in used and stratum.predicate(case)
            ),
            None,
        )
        if match is None:
            selection.unfilled.append(stratum.name)
            continue
        used.add(match.case_id)
        selection.cases.append(match)
        selection.assignments[stratum.name] = match.case_id

    return selection


def summarize_sample(selection: SampleSelection) -> dict:
    """Descriptive statistics for the chosen sample, for the run record."""
    cases = selection.cases
    if not cases:
        return {"case_count": 0}

    difficulties: dict[str, int] = {}
    categories: dict[str, int] = {}
    for case in cases:
        if case.difficulty:
            difficulties[case.difficulty.value] = difficulties.get(case.difficulty.value, 0) + 1
        for category in case.categories:
            categories[category.value] = categories.get(category.value, 0) + 1

    return {
        **selection.describe(),
        "by_difficulty": dict(sorted(difficulties.items())),
        "by_category": dict(sorted(categories.items())),
        "critical_facts": sum(len(case.critical_facts) for case in cases),
        "medication_facts": sum(len(case.medication_facts) for case in cases),
        "temporal_facts": sum(len(case.temporal_facts) for case in cases),
        "negations": sum(len(case.negations) for case in cases),
        "average_note_characters": round(
            sum(len(case.clinical_note) for case in cases) / len(cases), 1
        ),
    }


#: Finding types grouped for the failure taxonomy reported per model.
FAILURE_TAXONOMY: dict[str, tuple[FindingType, ...]] = {
    "critical_omission": (FindingType.CRITICAL_OMISSION,),
    "non_critical_omission": (FindingType.NON_CRITICAL_OMISSION,),
    "unsupported_addition": (FindingType.UNSUPPORTED_ADDITION,),
    "contradiction": (FindingType.CONTRADICTION,),
    "negation_reversal": (FindingType.NEGATION_REVERSAL,),
    "negation_omission": (FindingType.NEGATION_OMISSION,),
    "medication_omission": (FindingType.MEDICATION_OMISSION,),
    "medication_hallucination": (FindingType.MEDICATION_HALLUCINATION,),
    "medication_status_error": (FindingType.MEDICATION_STATUS_ERROR,),
    "medication_dose_error": (FindingType.MEDICATION_DOSE_ERROR,),
    "temporal_error": (
        FindingType.TEMPORAL_INFORMATION_LOSS,
        FindingType.TEMPORAL_TREND_REVERSAL,
        FindingType.TEMPORAL_VALUE_CONFUSION,
    ),
    "historical_current_confusion": (FindingType.HISTORICAL_CURRENT_CONFUSION,),
    "numeric_error": (FindingType.VALUE_ERROR, FindingType.VALUE_APPROXIMATION),
    "conflict_resolution_error": (FindingType.CONFLICT_RESOLUTION_ERROR,),
    "operational": (
        FindingType.EMPTY_OUTPUT,
        FindingType.MALFORMED_OUTPUT,
        FindingType.MISSING_ANNOTATION,
    ),
}


def classify_failures(findings) -> dict[str, int]:
    """Count findings under the taxonomy, including a zero for absent groups.

    Reporting explicit zeros matters: "0 negation reversals" and "we did not
    look for negation reversals" are different claims, and a taxonomy that
    silently omits empty groups cannot distinguish them.
    """
    counts = {group: 0 for group in FAILURE_TAXONOMY}
    lookup = {
        finding_type: group
        for group, types in FAILURE_TAXONOMY.items()
        for finding_type in types
    }
    for finding in findings:
        group = lookup.get(finding.finding_type)
        if group is not None:
            counts[group] += 1
    return counts
