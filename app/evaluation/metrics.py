"""Metric definitions and aggregation.

Every metric declares its numerator, denominator, interpretation, edge-case
behavior, and limitations *in code*, so the documentation and the computation
cannot drift apart. `docs/EVALUATION.md` renders this registry rather than
restating it.

Two rules run through all of it:

**An undefined metric is reported as undefined.** Zero critical facts means
retention has no value - not 1.0 ("perfect"), not 0.0 ("failed"). `value` is
None, and aggregation skips it. Reporting 100% retention for a case with
nothing to retain would be the kind of flattering nonsense this project exists
to avoid.

**Aggregation pools counts; it does not average averages.** A benchmark-level
retention of 0.9 means 90% of all critical facts across all cases were
retained. Macro-averaging would let a case with one critical fact weigh as
heavily as a case with twenty. Both are computed - `micro` is the headline,
`macro` is reported alongside because a large disparity between them is
itself informative (it means performance varies with case size).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import MetricValue


@dataclass(frozen=True)
class MetricDefinition:
    """The full specification of one metric."""

    name: str
    numerator: str
    denominator: str
    interpretation: str
    edge_cases: str
    limitations: str
    higher_is_better: bool = True

    def compute(self, numerator: float, denominator: float) -> MetricValue:
        """Compute this metric, returning an undefined value when denominator is 0."""
        value = numerator / denominator if denominator else None
        return MetricValue(
            name=self.name,
            value=value,
            numerator=numerator,
            denominator=denominator,
            definition=f"{self.numerator} / {self.denominator}",
        )


METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    "critical_information_retention": MetricDefinition(
        name="critical_information_retention",
        numerator="critical facts retained in the output",
        denominator="critical facts annotated on the case",
        interpretation=(
            "The share of clinically critical facts the summary preserved. The headline "
            "retention measure."
        ),
        edge_cases=(
            "Undefined when a case has no critical facts. A fact counted as retained may "
            "have matched at PARTIAL level; check the finding's match_level."
        ),
        limitations=(
            "Retention means the fact was found, not that it was framed correctly. A "
            "correctly retained but misattributed fact scores as retained here and is "
            "caught by the temporal and medication evaluators instead."
        ),
    ),
    "critical_omission_rate": MetricDefinition(
        name="critical_omission_rate",
        numerator="critical facts absent from the output",
        denominator="critical facts annotated on the case",
        interpretation="The share of critical facts the summary dropped entirely.",
        edge_cases="Undefined when a case has no critical facts.",
        limitations=(
            "Counts absence, not consequence. Dropping one panic-range potassium and "
            "dropping one routine allergy both count as one omission; severity "
            "distinguishes them, this rate does not."
        ),
        higher_is_better=False,
    ),
    "critical_contradiction_rate": MetricDefinition(
        name="critical_contradiction_rate",
        numerator="critical facts the output contradicts",
        denominator="critical facts annotated on the case",
        interpretation=(
            "The share of critical facts the summary actively misstates. Distinct from "
            "omission and generally worse: silence loses information, contradiction "
            "creates false information."
        ),
        edge_cases="Undefined when a case has no critical facts.",
        limitations="Depends on negation detection, which is scope-based and not perfect.",
        higher_is_better=False,
    ),
    "medication_consistency": MetricDefinition(
        name="medication_consistency",
        numerator="medications present with a compatible status and correct dose",
        denominator="medications annotated on the case",
        interpretation=(
            "The share of the medication list the summary reproduced faithfully in drug, "
            "status, and dose."
        ),
        edge_cases=(
            "Undefined when a case has no medications. A drug named without any status "
            "cue counts as consistent unless it is a change (new, discontinued, dose "
            "change), where an absent status is a partial failure."
        ),
        limitations=(
            "Status detection is cue-based. A summary conveying discontinuation through "
            "phrasing not in the cue list would be scored as a status error - a false "
            "positive this evaluator can produce."
        ),
    ),
    "medication_omission_rate": MetricDefinition(
        name="medication_omission_rate",
        numerator="medications not mentioned in the output",
        denominator="medications annotated on the case",
        interpretation="The share of the medication list the summary dropped.",
        edge_cases="Undefined when a case has no medications.",
        limitations="Does not weight by clinical importance of the drug.",
        higher_is_better=False,
    ),
    "medication_status_error_rate": MetricDefinition(
        name="medication_status_error_rate",
        numerator="medications whose status the output misrepresents",
        denominator="medications mentioned in the output",
        interpretation=(
            "Among drugs the summary did mention, the share it framed wrongly - a "
            "discontinued drug presented as current, a historical drug as active."
        ),
        edge_cases=(
            "Denominator is mentioned drugs, not all drugs: a drug the summary omitted "
            "cannot have its status misstated, and including it would dilute the rate."
        ),
        limitations="Cue-based detection, as above.",
        higher_is_better=False,
    ),
    "negation_consistency": MetricDefinition(
        name="negation_consistency",
        numerator="negated facts whose polarity the output preserved",
        denominator="negated facts annotated on the case",
        interpretation=(
            "The share of documented negatives the summary kept negative. A summary need "
            "not repeat every negative, but any it does mention must stay negated."
        ),
        edge_cases=(
            "Undefined when a case has no negations. An omitted negation counts as "
            "preserved here and is reported separately as negation_omission - dropping "
            "'no chest pain' loses information but does not create a false assertion."
        ),
        limitations=(
            "Relies on scope-based negation detection, which does not handle double "
            "negation, subject scope, or hypothetical framing."
        ),
    ),
    "negation_reversal_rate": MetricDefinition(
        name="negation_reversal_rate",
        numerator="negated facts the output asserts as positive",
        denominator="negated facts annotated on the case",
        interpretation=(
            "The share of documented negatives the summary flipped. The benchmark's "
            "signature failure: 'no history of diabetes' becoming 'patient has diabetes'."
        ),
        edge_cases="Undefined when a case has no negations.",
        limitations="As above, bounded by negation-detection accuracy.",
        higher_is_better=False,
    ),
    "temporal_consistency": MetricDefinition(
        name="temporal_consistency",
        numerator="temporal facts whose endpoints and direction the output preserved",
        denominator="temporal facts annotated on the case",
        interpretation=(
            "The share of laboratory trends the summary reported without losing or "
            "reversing the trajectory."
        ),
        edge_cases=(
            "Undefined when a case has no temporal facts. A trend whose analyte is not "
            "mentioned at all counts as inconsistent, since the trend is lost either way."
        ),
        limitations=(
            "Direction detection is keyword-based over a window after the analyte name. "
            "A trend described across several sentences may be scored as information "
            "loss even where a reader would follow it."
        ),
    ),
    "temporal_information_loss_rate": MetricDefinition(
        name="temporal_information_loss_rate",
        numerator="temporal facts reported with only one endpoint",
        denominator="temporal facts annotated on the case",
        interpretation=(
            "The share of trends flattened to a single value. Distinguished from "
            "reversal because losing context and stating the opposite are different "
            "errors."
        ),
        edge_cases="Undefined when a case has no temporal facts.",
        limitations="Does not judge whether the retained endpoint was the clinically right one.",
        higher_is_better=False,
    ),
    "supported_fact_rate": MetricDefinition(
        name="supported_fact_rate",
        numerator="structured entities the output supports",
        denominator="structured entities analyzable on the case",
        interpretation=(
            "Across every clinical entity in the case - conditions, medications, labs, "
            "vitals, procedures, findings, allergies, trends - the share the summary "
            "represents faithfully."
        ),
        edge_cases=(
            "Entity types without an analyzer are excluded from the denominator rather "
            "than counted as failures."
        ),
        limitations=(
            "Treats every entity as equally important, which they are not. Use alongside "
            "critical_information_retention, which weights by the benchmark's own "
            "importance labels."
        ),
    ),
    "unsupported_fact_rate": MetricDefinition(
        name="unsupported_fact_rate",
        numerator="clinical assertions in the output absent from the structured case",
        denominator="clinical assertions detected in the output",
        interpretation=(
            "How much of what the summary asserts is not in the source - the "
            "hallucination measure."
        ),
        edge_cases=(
            "Undefined when no clinical assertions are detected, including for an empty "
            "output. Detection covers the benchmark's own vocabulary only, so a "
            "hallucinated term outside that vocabulary is not counted."
        ),
        limitations=(
            "Vocabulary-bounded, so this is a floor on hallucination, not a ceiling. It "
            "cannot detect invented content expressed in terms the content library does "
            "not contain."
        ),
        higher_is_better=False,
    ),
}


def get_definition(name: str) -> MetricDefinition:
    if name not in METRIC_DEFINITIONS:
        raise KeyError(
            f"Unknown metric '{name}'. Every metric must be declared in "
            f"app/evaluation/metrics.py so its definition ships with its value."
        )
    return METRIC_DEFINITIONS[name]


def compute(name: str, numerator: float, denominator: float) -> MetricValue:
    return get_definition(name).compute(numerator, denominator)


def aggregate(case_metrics: list[dict[str, MetricValue]]) -> dict[str, MetricValue]:
    """Pool per-case metrics into benchmark-level metrics.

    Micro-averaged: numerators and denominators are summed across cases, then
    divided. Cases where a metric was undefined contribute nothing to either,
    so they neither inflate nor deflate the result.
    """
    totals: dict[str, tuple[float, float]] = {}
    for metrics in case_metrics:
        for name, metric in metrics.items():
            if metric.denominator == 0:
                continue
            numerator, denominator = totals.get(name, (0.0, 0.0))
            totals[name] = (numerator + metric.numerator, denominator + metric.denominator)

    return {
        name: compute(name, numerator, denominator)
        for name, (numerator, denominator) in sorted(totals.items())
    }


def macro_average(case_metrics: list[dict[str, MetricValue]]) -> dict[str, float | None]:
    """Unweighted mean of per-case values, for comparison against the micro rate.

    A large gap between macro and micro means performance varies with how much
    is at stake in a case, which is worth surfacing rather than hiding.
    """
    collected: dict[str, list[float]] = {}
    for metrics in case_metrics:
        for name, metric in metrics.items():
            if metric.value is not None:
                collected.setdefault(name, []).append(metric.value)
    return {
        name: (sum(values) / len(values) if values else None)
        for name, values in sorted(collected.items())
    }
