"""The Phase 5 experiment: evaluating models on a fixed case sample.

Runs the full pipeline for each model over the *same* stratified sample, and
records everything needed to reproduce and audit it.

    sample -> model -> summary -> deterministic evaluation -> routing
                                                                 |
                          informed adjudication  <---------------+
                          blind adjudication     <---------------+

Three disciplines this module enforces:

**Pre- and post-adjudication are never mixed.** Deterministic metrics are
computed once, before any model judgment, and are reported unchanged.
Post-adjudication findings are reported separately. A single number combining
them would be neither reproducible nor deterministic.

**Small samples are labelled as such.** Every result carries
`exploratory: true` and counts alongside rates. Ten cases cannot support a
percentage presented as an estimate of real-world performance, and the report
says so rather than trusting the reader to remember.

**Nothing is fabricated.** A model that cannot be reached produces a recorded
failure, not an omission and not a substituted number.

The blind-versus-informed comparison exists because showing the judge the
deterministic finding might anchor it. Condition A includes the finding,
condition B withholds it, everything else is identical. Which is better is an
empirical question this module measures rather than assumes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.enums import AdjudicationStatus, Severity
from app.evaluation.llm import (
    AdjudicationPolicy,
    LLMAdjudicator,
    apply_records,
)
from app.evaluation.runner import EvaluationRunner, summarize_findings
from app.experiments.selection import classify_failures
from app.generation import OutputCache, SummaryGenerator, to_ai_outputs
from app.models.schemas import BenchmarkCase
from app.prompts.library import load_prompt
from app.providers.base import ModelProvider

EXPERIMENT_VERSION = "0.5.0"

#: Prompt used for each adjudication condition.
ADJUDICATION_CONDITIONS: dict[str, str] = {
    "informed": "adjudication",
    "blind": "adjudication_blind",
}


@dataclass
class ModelSpec:
    """One model under evaluation, described entirely by configuration."""

    label: str
    provider_name: str
    model: str
    provider: ModelProvider
    temperature: float = 0.0
    max_tokens: int = 1024
    runs_per_case: int = 1

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "provider": self.provider_name,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "runs_per_case": self.runs_per_case,
            **self.provider.describe(),
        }


@dataclass
class ModelResult:
    """Everything observed for one model on the sample."""

    spec: ModelSpec
    generation: dict[str, Any] = field(default_factory=dict)
    output_lengths: list[int] = field(default_factory=list)
    pre_adjudication_metrics: dict[str, Any] = field(default_factory=dict)
    pre_adjudication_findings: list = field(default_factory=list)
    failure_taxonomy: dict[str, int] = field(default_factory=dict)
    severity_counts: dict[str, int] = field(default_factory=dict)
    adjudication: dict[str, Any] = field(default_factory=dict)
    repeated_runs: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def average_output_length(self) -> float | None:
        if not self.output_lengths:
            return None
        return round(sum(self.output_lengths) / len(self.output_lengths), 1)


def _severity_counts(findings) -> dict[str, int]:
    counts = {level.value: 0 for level in Severity}
    for finding in findings:
        if finding.severity:
            counts[finding.severity.value] += 1
    return counts


def evaluate_model(
    spec: ModelSpec,
    cases: list[BenchmarkCase],
    cache: OutputCache | None = None,
    adjudicate: bool = False,
    judge: ModelSpec | None = None,
    policy: AdjudicationPolicy | None = None,
    force: bool = False,
) -> ModelResult:
    """Generate, evaluate, and (optionally) adjudicate one model on the sample."""
    result = ModelResult(spec=spec)

    generator = SummaryGenerator(
        provider=spec.provider,
        model=spec.model,
        cache=cache if cache is not None else OutputCache(),
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
    )
    responses = generator.generate_all(cases, force=force)
    outputs = to_ai_outputs(responses)

    for case_id, response in responses.items():
        if response.error:
            result.errors.append(f"{case_id}: {response.error}")
        else:
            result.output_lengths.append(len(response.summary_text))

    result.generation = {
        "metadata": generator.metadata(),
        "summary": generator.summary.report(),
        "average_output_characters": result.average_output_length,
        "summaries_produced": len(outputs),
    }

    runner = EvaluationRunner()
    evaluation = runner.evaluate_benchmark(
        cases, outputs, model_id=f"{spec.provider_name}:{spec.model}"
    )

    # Snapshot BEFORE adjudication. These are the deterministic numbers and
    # they are never revised by a model judgment.
    result.pre_adjudication_metrics = {
        name: metric.model_dump(mode="json") for name, metric in evaluation.metrics.items()
    }
    result.pre_adjudication_findings = list(evaluation.findings)
    result.failure_taxonomy = classify_failures(evaluation.findings)
    result.severity_counts = _severity_counts(evaluation.findings)

    if adjudicate and judge is not None:
        result.adjudication = _run_adjudication_conditions(
            evaluation.findings, outputs, judge, policy or AdjudicationPolicy(enabled=True)
        )

    return result


def _run_adjudication_conditions(
    findings, outputs, judge: ModelSpec, policy: AdjudicationPolicy
) -> dict[str, Any]:
    """Run both adjudication conditions over the same findings.

    Both conditions see identical findings and identical summaries; only the
    prompt differs. That isolation is the whole point - any difference in
    outcome is attributable to whether the judge saw the deterministic result.
    """
    summaries = {case_id: output.summary_text for case_id, output in outputs.items()}
    conditions: dict[str, Any] = {}

    for condition, prompt_name in ADJUDICATION_CONDITIONS.items():
        adjudicator = LLMAdjudicator(
            provider=judge.provider,
            model=judge.model,
            policy=policy,
            prompt=load_prompt(prompt_name, "v1"),
            temperature=judge.temperature,
        )
        records = adjudicator.adjudicate_findings(findings, summaries)
        kept, withdrawn = apply_records(findings, records)

        conditions[condition] = {
            "prompt": prompt_name,
            "metadata": adjudicator.metadata(),
            "summary": adjudicator.summary.report(),
            "records": [record.model_dump(mode="json") for record in records],
            "post_adjudication_finding_count": len(kept),
            "withdrawn_finding_count": len(withdrawn),
            "withdrawn_finding_ids": [f.finding_id for f in withdrawn],
        }

    conditions["comparison"] = compare_conditions(
        conditions["informed"], conditions["blind"]
    )
    return conditions


def compare_conditions(informed: dict, blind: dict) -> dict[str, Any]:
    """Compare the informed and blind adjudication conditions.

    Reports agreement on the judgment itself, not just on the final
    disposition: two conditions can reach the same disposition through
    different judgments, and collapsing them would hide the anchoring effect
    the experiment is designed to detect.
    """
    informed_judgments = {
        record["finding_id"]: record.get("judgment")
        for record in informed["records"]
        if record.get("judgment") is not None
    }
    blind_judgments = {
        record["finding_id"]: record.get("judgment")
        for record in blind["records"]
        if record.get("judgment") is not None
    }

    shared = sorted(set(informed_judgments) & set(blind_judgments))
    agreements = [
        finding_id
        for finding_id in shared
        if informed_judgments[finding_id] == blind_judgments[finding_id]
    ]
    disagreements = [
        {
            "finding_id": finding_id,
            "informed": informed_judgments[finding_id],
            "blind": blind_judgments[finding_id],
        }
        for finding_id in shared
        if informed_judgments[finding_id] != blind_judgments[finding_id]
    ]

    return {
        "findings_judged_in_both": len(shared),
        "judgment_agreements": len(agreements),
        "judgment_disagreements": len(disagreements),
        "judgment_agreement_rate": (
            round(len(agreements) / len(shared), 4) if shared else None
        ),
        "disagreement_detail": disagreements[:25],
        "withdrawal_informed": informed["withdrawn_finding_count"],
        "withdrawal_blind": blind["withdrawn_finding_count"],
        "withdrawal_delta": (
            blind["withdrawn_finding_count"] - informed["withdrawn_finding_count"]
        ),
        "adjudication_required_informed": informed["summary"]["adjudication_required"],
        "adjudication_required_blind": blind["summary"]["adjudication_required"],
        "interpretation_note": (
            "A large withdrawal_delta or a low judgment_agreement_rate suggests the "
            "deterministic finding is anchoring the judge. Neither condition is "
            "assumed better; this is a measurement, not a preference."
        ),
    }


def repeated_run_analysis(
    spec: ModelSpec,
    cases: list[BenchmarkCase],
    runs: int,
    cache: OutputCache | None = None,
) -> dict[str, Any]:
    """Generate each case several times and measure variation.

    Only meaningful for a stochastic model. At temperature 0 a well-behaved
    provider should be near-deterministic, and the measured variation is
    itself the result - reported rather than assumed.
    """
    runner = EvaluationRunner()
    per_case: dict[str, dict[str, Any]] = {}

    for run_index in range(runs):
        generator = SummaryGenerator(
            provider=spec.provider, model=spec.model,
            cache=cache if cache is not None else OutputCache(),
            temperature=spec.temperature, max_tokens=spec.max_tokens,
        )
        responses = generator.generate_all(cases, run_index=run_index)
        outputs = to_ai_outputs(responses)

        for case in cases:
            output = outputs.get(case.case_id)
            if output is None:
                continue
            entry = per_case.setdefault(
                case.case_id, {"summaries": [], "finding_counts": [], "finding_type_sets": []}
            )
            entry["summaries"].append(output.summary_text)
            case_result = runner.evaluate_case(case, output)
            entry["finding_counts"].append(len(case_result.findings))
            entry["finding_type_sets"].append(
                sorted({f.finding_type.value for f in case_result.findings})
            )

    identical_outputs = 0
    identical_findings = 0
    for entry in per_case.values():
        if len(set(entry["summaries"])) == 1:
            identical_outputs += 1
        if len({tuple(types) for types in entry["finding_type_sets"]}) == 1:
            identical_findings += 1

    return {
        "runs_per_case": runs,
        "cases": len(per_case),
        "cases_with_identical_output": identical_outputs,
        "cases_with_identical_findings": identical_findings,
        "output_stability_rate": (
            round(identical_outputs / len(per_case), 4) if per_case else None
        ),
        "finding_stability_rate": (
            round(identical_findings / len(per_case), 4) if per_case else None
        ),
        "temperature": spec.temperature,
        "note": (
            "Stability at temperature 0 reflects provider determinism, not model "
            "quality. A mock provider is trivially stable and proves nothing about "
            "a real model."
        ),
    }


def build_comparison(
    results: list[ModelResult], benchmark_version: str, sample: dict
) -> dict[str, Any]:
    """Machine-readable model comparison.

    Deliberately produces **no overall ranking**. Different models fail in
    different ways, and collapsing distinct failure profiles into one number
    would discard exactly the information a deployment decision needs.
    """
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_version": benchmark_version,
        "exploratory": True,
        "sample": sample,
        "models": [
            {
                "model": result.spec.describe(),
                "summaries_produced": result.generation.get("summaries_produced", 0),
                "average_output_characters": result.average_output_length,
                "pre_adjudication_metrics": result.pre_adjudication_metrics,
                "pre_adjudication_finding_count": len(result.pre_adjudication_findings),
                "failure_taxonomy": result.failure_taxonomy,
                "severity_counts": result.severity_counts,
                "post_adjudication": (
                    {
                        condition: {
                            "finding_count": data["post_adjudication_finding_count"],
                            "withdrawn": data["withdrawn_finding_count"],
                        }
                        for condition, data in result.adjudication.items()
                        if condition in ADJUDICATION_CONDITIONS
                    }
                    if result.adjudication
                    else None
                ),
                "adjudication_comparison": result.adjudication.get("comparison"),
                "repeated_runs": result.repeated_runs or None,
                "generation_errors": result.errors,
            }
            for result in results
        ],
        "notes": [
            "EXPLORATORY BENCHMARK RESULTS. The sample is small and synthetic; "
            "rates are reported alongside counts and are not estimates of "
            "real-world performance.",
            "Pre-adjudication metrics are deterministic and are not revised by any "
            "model judgment. Post-adjudication finding counts are reported "
            "separately and must not be combined with them.",
            "No overall ranking is produced. Models with different failure "
            "profiles are not ordered on a single axis.",
            "No risk score or deployment recommendation is produced.",
        ],
    }


def new_experiment_id() -> str:
    return f"exp-{uuid.uuid4().hex[:12]}"
