"""Run the Phase 5 real-model experiment.

    python -m scripts.run_experiment --models anthropic:<model-id> openai:<model-id>

Evaluates each model on the same stratified 10-case sample, runs deterministic
evaluation, then adjudicates ambiguous findings under both the informed and
blind conditions.

**This script will not fabricate a real-model result.** Asking for a provider
that has no credential is an error, not a silent fallback to the mock. The mock
provider is available explicitly (`--models mock:name`) for exercising the
apparatus, and every artifact it produces is labelled
`is_real_model_output: false`.

Model identifiers come entirely from the command line or configuration. None is
hard-coded, because provider model IDs change and a pinned name silently rots.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark import generate_benchmark, load_generated_benchmark
from app.core.config import settings
from app.evaluation.llm import load_adjudication_policy
from app.experiments.runner import (
    EXPERIMENT_VERSION,
    ModelSpec,
    build_comparison,
    evaluate_model,
    new_experiment_id,
    repeated_run_analysis,
)
from app.experiments.selection import select_experiment_sample, summarize_sample
from app.generation import OutputCache
from app.providers import ProviderError, build_provider, mock_summarizer
from app.providers.registry import describe_availability

DATA_NOTICE = (
    "SYNTHETIC DATA ONLY. This experiment runs on generated benchmark cases. "
    "No real patient information is used, and none must be introduced."
)

EXPLORATORY_NOTICE = (
    "EXPLORATORY RESULTS. A 10-case sample cannot support percentages read as "
    "estimates of real-world performance. Counts are reported alongside rates."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_experiment",
        description="Evaluate one or more models on the Phase 5 case sample.",
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Models as provider:model-id, e.g. anthropic:<model-id> mock:demo",
    )
    parser.add_argument("--benchmark", type=Path, default=None, help="Benchmark cases JSON")
    parser.add_argument("--pool", type=int, default=100, help="Cases to select the sample from")
    parser.add_argument("--seed", type=int, default=42, help="Benchmark seed")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)

    parser.add_argument("--adjudicate", action="store_true", help="Run LLM adjudication")
    parser.add_argument("--judge", default=None, help="Judge as provider:model-id")

    parser.add_argument(
        "--repeat-model", default=None,
        help="Label of one model to generate repeatedly, for stability analysis",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--repeat-temperature", type=float, default=None,
        help="Temperature for repeated runs (defaults to --temperature)",
    )

    parser.add_argument("--behaviour", default="correct", help="Mock provider behaviour")
    parser.add_argument("--out", type=Path, default=None, help="Directory for the artifact")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate, ignoring cache")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def parse_model_spec(
    text: str, args: argparse.Namespace, cases
) -> ModelSpec:
    """Turn 'provider:model-id' into a constructed ModelSpec.

    Raises rather than substituting a mock when a real provider is unavailable:
    a run that silently degraded to synthetic output would produce a result
    file indistinguishable from a real experiment at a glance.
    """
    provider_name, _, model = text.partition(":")
    provider_name = provider_name.strip()
    model = model.strip()
    if not provider_name or not model:
        raise ValueError(
            f"Model spec '{text}' must be provider:model-id, e.g. anthropic:<model-id>"
        )

    if provider_name == "mock":
        provider = mock_summarizer(cases, args.behaviour, model=model)
    else:
        provider = build_provider(provider_name)

    return ModelSpec(
        label=text,
        provider_name=provider_name,
        model=model,
        provider=provider,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def load_sample(args: argparse.Namespace):
    if args.benchmark:
        pool = load_generated_benchmark(args.benchmark)
    else:
        pool = generate_benchmark(case_count=args.pool, seed=args.seed)
    return select_experiment_sample(pool), pool


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_id = new_experiment_id()

    selection, _ = load_sample(args)
    if not selection.cases:
        print("No cases selected; cannot run the experiment.", file=sys.stderr)
        return 1

    if not args.quiet:
        print("=" * 78)
        print("ClinEval AI - Phase 5 model experiment")
        print("=" * 78)
        print(DATA_NOTICE)
        print(EXPLORATORY_NOTICE)
        print()
        if selection.unfilled:
            print(f"  WARNING: unfilled strata: {', '.join(selection.unfilled)}")
        print(f"  Sample: {len(selection)} cases  ({', '.join(selection.case_ids)})")
        print()

    # -- build model specs, refusing to substitute --------------------------
    specs: list[ModelSpec] = []
    for text in args.models:
        try:
            specs.append(parse_model_spec(text, args, selection.cases))
        except (ProviderError, KeyError, ValueError) as error:
            print(f"Error building model '{text}': {error}", file=sys.stderr)
            print(
                "\nNo result is written. This script does not fall back to a mock "
                "provider, because a fabricated result must never be mistaken for a "
                "real one.",
                file=sys.stderr,
            )
            print(
                f"Provider availability: {json.dumps(describe_availability(), indent=2)}",
                file=sys.stderr,
            )
            return 1

    judge_spec = None
    if args.adjudicate:
        judge_text = args.judge or f"mock:{settings.judge_model or 'mock-judge-v1'}"
        provider_name, _, judge_model = judge_text.partition(":")
        try:
            if provider_name == "mock":
                from app.providers import mock_judge

                judge_provider = mock_judge("SUPPORTED", 0.9)
            else:
                judge_provider = build_provider(provider_name)
        except (ProviderError, KeyError) as error:
            print(f"Error building judge '{judge_text}': {error}", file=sys.stderr)
            return 1
        judge_spec = ModelSpec(
            label=judge_text, provider_name=provider_name,
            model=judge_model or "mock-judge-v1", provider=judge_provider,
            temperature=0.0,
        )

    # -- run ----------------------------------------------------------------
    cache = OutputCache(enabled=not args.no_cache)
    policy = load_adjudication_policy().model_copy(update={"enabled": True})
    results = []

    for spec in specs:
        if not args.quiet:
            print(f"  Evaluating {spec.label} ...")
        result = evaluate_model(
            spec, selection.cases, cache=cache,
            adjudicate=args.adjudicate, judge=judge_spec, policy=policy,
            force=args.force,
        )
        if args.repeat_model and spec.label == args.repeat_model:
            repeat_spec = spec
            if args.repeat_temperature is not None:
                repeat_spec = ModelSpec(
                    **{**spec.__dict__, "temperature": args.repeat_temperature}
                )
            result.repeated_runs = repeated_run_analysis(
                repeat_spec, selection.cases, args.repeats, cache=cache
            )
        results.append(result)

    # -- artifact -----------------------------------------------------------
    comparison = build_comparison(
        results,
        benchmark_version=selection.cases[0].benchmark_version,
        sample=summarize_sample(selection),
    )
    comparison["experiment_id"] = experiment_id
    comparison["command"] = " ".join(argv or sys.argv[1:])
    comparison["adjudication_enabled"] = bool(judge_spec)
    comparison["judge"] = judge_spec.describe() if judge_spec else None

    out_dir = Path(args.out) if args.out else settings.runs_dir / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / f"{experiment_id}.json"
    artifact.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    review = build_review_set(results, selection.cases)
    review_path = out_dir / f"{experiment_id}-review.json"
    review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.quiet:
        print_report(comparison, artifact, review_path, review)
    return 0


def build_review_set(results, cases, limit_per_model: int = 12) -> dict:
    """Human inspection sample: the cases most worth a person's attention.

    Selected by three criteria, not by score: the highest-severity findings,
    findings where deterministic and LLM evaluation disagreed, and findings
    resting on a weak match (the most likely false positives). Each entry
    carries the full evidence chain so a reviewer can judge without opening the
    benchmark.
    """
    by_id = {case.case_id: case for case in cases}
    severity_rank = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3, "INFO": 4}
    entries = []

    for result in results:
        disagreed: dict[str, dict] = {}
        for condition in ("informed", "blind"):
            for record in result.adjudication.get(condition, {}).get("records", []):
                if record.get("status") in {
                    "adjudication_required", "resolved_by_llm", "failed"
                }:
                    disagreed.setdefault(record["finding_id"], {})[condition] = record

        ranked = sorted(
            result.pre_adjudication_findings,
            key=lambda f: (
                # Disagreements first, then severity, then weak matches.
                0 if f.finding_id in disagreed else 1,
                severity_rank.get(f.severity.value if f.severity else "INFO", 5),
                f.confidence,
            ),
        )

        for finding in ranked[:limit_per_model]:
            case = by_id.get(finding.case_id)
            adjudications = disagreed.get(finding.finding_id, {})
            entries.append(
                {
                    "model": result.spec.label,
                    "case_id": finding.case_id,
                    "selection_reason": (
                        "deterministic/LLM disagreement"
                        if adjudications
                        else "high severity"
                        if finding.severity and finding.severity.value in ("HIGH", "CRITICAL")
                        else "weak match - potential false positive"
                    ),
                    "source_note_excerpt": (case.clinical_note[:600] if case else None),
                    "model_output": None,  # filled below where available
                    "deterministic_finding": {
                        "finding_id": finding.finding_id,
                        "type": finding.finding_type.value,
                        "severity": finding.severity.value if finding.severity else None,
                        "explanation": finding.explanation,
                        "confidence": finding.confidence,
                        "match_level": (
                            finding.match_level.value if finding.match_level else None
                        ),
                        "source_evidence": finding.source_evidence,
                        "output_evidence": finding.output_evidence,
                    },
                    "llm_judgment": {
                        condition: {
                            "judgment": record.get("judgment"),
                            "model_reported_confidence": record.get(
                                "model_reported_confidence"
                            ),
                            "reason": record.get("model_reason"),
                        }
                        for condition, record in adjudications.items()
                    },
                    "final_disposition": {
                        condition: record.get("status")
                        for condition, record in adjudications.items()
                    }
                    or {"deterministic": "finding stands (not adjudicated)"},
                }
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_version": EXPERIMENT_VERSION,
        "entry_count": len(entries),
        "entries": entries,
        "notes": [
            "Selected for human review, not automatically labelled. A finding here "
            "is a POTENTIAL false positive or negative until a person verifies it.",
            EXPLORATORY_NOTICE,
        ],
    }


def _fmt(value) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_report(comparison: dict, artifact: Path, review_path: Path, review: dict) -> None:
    for model in comparison["models"]:
        spec = model["model"]
        print()
        print("-" * 78)
        print(f"MODEL: {spec['label']}   (real output: {spec.get('is_real_model_output', 'n/a')})")
        print("-" * 78)
        print(f"  summaries produced   : {model['summaries_produced']}")
        print(f"  avg output characters: {_fmt(model['average_output_characters'])}")
        if model["generation_errors"]:
            print(f"  generation errors    : {len(model['generation_errors'])}")

        print("\n  Pre-adjudication metrics (deterministic, unrevised):")
        for name, metric in sorted(model["pre_adjudication_metrics"].items()):
            counts = f"({metric['numerator']:g}/{metric['denominator']:g})"
            print(f"    {name:<34} {_fmt(metric['value']):>8}  {counts}")

        print(f"\n  Pre-adjudication findings: {model['pre_adjudication_finding_count']}")
        for group, count in model["failure_taxonomy"].items():
            if count:
                print(f"    {group:<34} {count}")
        severities = {k: v for k, v in model["severity_counts"].items() if v}
        if severities:
            print(f"    severity: {severities}")

        if model["post_adjudication"]:
            print("\n  Post-adjudication findings (reported separately):")
            for condition, data in model["post_adjudication"].items():
                print(
                    f"    {condition:<10} findings={data['finding_count']:<4} "
                    f"withdrawn={data['withdrawn']}"
                )
            comparison_block = model.get("adjudication_comparison") or {}
            if comparison_block:
                print("\n  Blind vs informed adjudication:")
                print(f"    judged in both        : {comparison_block['findings_judged_in_both']}")
                print(f"    judgment agreement    : {_fmt(comparison_block['judgment_agreement_rate'])}")
                print(f"    disagreements         : {comparison_block['judgment_disagreements']}")
                print(f"    withdrawal delta      : {comparison_block['withdrawal_delta']}")

        if model["repeated_runs"]:
            repeats = model["repeated_runs"]
            print(f"\n  Repeated runs ({repeats['runs_per_case']}x, temp {repeats['temperature']}):")
            print(f"    identical outputs  : {repeats['cases_with_identical_output']}/{repeats['cases']}")
            print(f"    identical findings : {repeats['cases_with_identical_findings']}/{repeats['cases']}")

    print()
    print("=" * 78)
    print(f"Comparison artifact : {artifact}")
    print(f"Review set          : {review_path}  ({review['entry_count']} entries)")
    print()
    for note in comparison["notes"]:
        print(f"  - {note}")


if __name__ == "__main__":
    raise SystemExit(main())
