"""Evaluate a model against the benchmark, end to end.

    python -m scripts.evaluate_benchmark --model mock --cases 10

The full pipeline: load the benchmark, generate summaries, run deterministic
evaluation, route ambiguous findings to LLM adjudication if enabled, reconcile
disagreements, and save a reproducible run.

Defaults are offline. With no `--provider`, the deterministic mock provider is
used and nothing leaves the machine. A remote provider requires both an
explicit flag and a configured credential, and the run prints what it is about
to send before sending it.

Adjudication is off unless `--adjudicate` is passed, so a run never starts
making paid external calls by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark import generate_benchmark, load_generated_benchmark
from app.core.config import settings
from app.evaluation.llm import (
    LLMAdjudicator,
    apply_records,
    load_adjudication_policy,
)
from app.evaluation.runner import EvaluationRunner, summarize_findings
from app.generation import OutputCache, SummaryGenerator, to_ai_outputs
from app.providers import BEHAVIOURS, ProviderError, build_provider, mock_summarizer
from app.providers.registry import describe_availability

DATA_WARNING = (
    "This platform is for SYNTHETIC and research data only. Do not evaluate "
    "real patient information with it. Benchmark content is sent to a model "
    "provider only when you configure a remote provider explicitly."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.evaluate_benchmark",
        description="Generate model summaries and evaluate them against the benchmark.",
    )
    parser.add_argument("--benchmark", type=Path, default=None, help="Benchmark cases JSON")
    parser.add_argument("--cases", type=int, default=10, help="Number of cases to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Seed when generating a benchmark")

    parser.add_argument("--provider", default="mock", help="Provider name (default: mock)")
    parser.add_argument("--model", default=None, help="Model name (from configuration)")
    parser.add_argument(
        "--behaviour", default="correct", choices=list(BEHAVIOURS),
        help="Mock provider behaviour (mock provider only)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)

    parser.add_argument("--adjudicate", action="store_true", help="Enable LLM adjudication")
    parser.add_argument("--judge-provider", default=None, help="Provider for the judge")
    parser.add_argument("--judge-model", default=None, help="Model for the judge")

    parser.add_argument("--no-cache", action="store_true", help="Bypass the output cache")
    parser.add_argument("--force", action="store_true", help="Regenerate, ignoring cached output")
    parser.add_argument("--out", type=Path, default=None, help="Directory to write the run to")
    parser.add_argument("--quiet", action="store_true", help="Print less")
    return parser.parse_args(argv)


def load_cases(args: argparse.Namespace):
    if args.benchmark:
        return load_generated_benchmark(args.benchmark)[: args.cases]
    return generate_benchmark(case_count=args.cases, seed=args.seed)


def build_summary_provider(args: argparse.Namespace, cases):
    """Construct the summarizing provider, with a clear message on failure."""
    if args.provider == "mock":
        return mock_summarizer(cases, args.behaviour, model=args.model or "mock-summarizer-v1")
    return build_provider(args.provider)


def _print_data_warning(provider_name: str, case_count: int) -> None:
    print("=" * 78)
    print("ClinEval AI - benchmark evaluation")
    print("=" * 78)
    print(DATA_WARNING)
    if provider_name != "mock":
        print()
        print(
            f"  NOTE: provider '{provider_name}' is remote. {case_count} synthetic "
            f"clinical notes will be sent to it."
        )
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = f"run-{uuid.uuid4().hex[:12]}"

    cases = load_cases(args)
    if not cases:
        print("No cases to evaluate.", file=sys.stderr)
        return 1

    if not args.quiet:
        _print_data_warning(args.provider, len(cases))

    # -- 1/2. provider and generation ---------------------------------------
    try:
        provider = build_summary_provider(args, cases)
    except (ProviderError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        print(
            f"\nProvider availability: "
            f"{json.dumps(describe_availability(), indent=2)}",
            file=sys.stderr,
        )
        return 1

    model = args.model or settings.summarizer_model or "mock-summarizer-v1"
    cache = OutputCache(enabled=not args.no_cache)
    generator = SummaryGenerator(
        provider=provider, model=model, cache=cache,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )
    responses = generator.generate_all(cases, force=args.force)
    outputs = to_ai_outputs(responses)

    # -- 3. deterministic evaluation ----------------------------------------
    runner = EvaluationRunner()
    result = runner.evaluate_benchmark(
        cases, outputs, model_id=f"{provider.kind.value}:{model}",
        evaluation_run_id=run_id,
        configuration={"generation": generator.metadata()},
    )

    # -- 4/5. adjudication ---------------------------------------------------
    records = []
    withdrawn = []
    adjudicator = None
    if args.adjudicate:
        policy = load_adjudication_policy().model_copy(update={"enabled": True})
        judge_provider_name = args.judge_provider or args.provider
        try:
            judge_provider = (
                _build_mock_judge() if judge_provider_name == "mock"
                else build_provider(judge_provider_name)
            )
        except (ProviderError, KeyError) as error:
            print(f"Error building judge provider: {error}", file=sys.stderr)
            return 1

        adjudicator = LLMAdjudicator(
            provider=judge_provider,
            model=args.judge_model or settings.judge_model or "mock-judge-v1",
            policy=policy,
        )
        summaries = {case_id: output.summary_text for case_id, output in outputs.items()}
        records = adjudicator.adjudicate_findings(result.findings, summaries)
        kept, withdrawn = apply_records(result.findings, records)
        result.findings = kept

    # -- 6/7. structured result and persistence ------------------------------
    run = build_run_document(
        run_id, args, cases, generator, result, adjudicator, records, withdrawn
    )
    out_dir = args.out or settings.runs_dir
    run_path = save_run(run, out_dir, run_id)

    # -- 8. report ------------------------------------------------------------
    if not args.quiet:
        print_report(run, run_path)
    return 0


def _build_mock_judge():
    from app.providers import mock_judge

    return mock_judge("SUPPORTED", 0.9)


def build_run_document(
    run_id, args, cases, generator, result, adjudicator, records, withdrawn
) -> dict:
    """Assemble everything needed to reproduce and audit this run."""
    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_version": result.benchmark_version,
        "case_count": len(cases),
        "cases_evaluated": result.cases_evaluated,
        "model": generator.model,
        "provider": generator.provider.kind.value,
        "evaluation_version": result.evaluation_version,
        "evaluators": result.evaluators,
        "configuration": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "behaviour": args.behaviour if args.provider == "mock" else None,
            "seed": args.seed,
            "adjudication_enabled": bool(adjudicator),
        },
        "generation": generator.metadata(),
        "generation_summary": generator.summary.report(),
        # Deterministic metrics, computed before adjudication and deliberately
        # not adjusted by it. Letting a model's judgments move these would make
        # the headline numbers partly model-derived, which is exactly what the
        # deterministic-first design exists to prevent.
        "metrics": {
            name: metric.model_dump(mode="json") for name, metric in result.metrics.items()
        },
        "metrics_are_pre_adjudication": True,
        "findings_summary": summarize_findings(result),
        "findings_are_post_adjudication": bool(adjudicator),
        "findings": [finding.model_dump(mode="json") for finding in result.findings],
        "adjudication": (
            {
                "metadata": adjudicator.metadata(),
                "summary": adjudicator.summary.report(),
                "records": [record.model_dump(mode="json") for record in records],
                "withdrawn_findings": [f.model_dump(mode="json") for f in withdrawn],
            }
            if adjudicator
            else None
        ),
        "notes": [
            DATA_WARNING,
            "Deterministic findings are authoritative. LLM judgments adjudicate "
            "only ambiguous findings and never silently override a deterministic "
            "result.",
            "Metrics are deterministic and pre-adjudication; the findings list is "
            "post-adjudication. Withdrawn findings are retained under "
            "adjudication.withdrawn_findings so nothing is silently dropped.",
            "model_reported_confidence is the model's own stated number and is "
            "not a calibrated probability.",
            "No overall risk score or deployment recommendation is produced.",
        ],
    }


def save_run(run: dict, out_dir: Path, run_id: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _fmt(value) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_report(run: dict, run_path: Path) -> None:
    print(f"Run          : {run['run_id']}")
    print(f"Benchmark    : {run['benchmark_version']}  ({run['cases_evaluated']} cases)")
    print(f"Model        : {run['provider']}:{run['model']}")
    print(f"Real output  : {run['generation'].get('is_real_model_output', 'n/a')}")

    print("\n-- Generation --------------------------------------------------------")
    generation = run["generation_summary"]
    print(f"  model calls        : {generation['model_calls']}")
    print(f"  served from cache  : {generation['served_from_cache']}")
    print(f"  failed             : {generation['failed']}")
    print(f"  total tokens       : {generation['tokens']['total']}")
    print(f"  estimated cost USD : {_fmt(generation['estimated_cost_usd'])}")
    print(f"  average latency ms : {_fmt(generation['average_latency_ms'])}")

    print("\n-- Deterministic metrics (pre-adjudication) --------------------------")
    for name, metric in sorted(run["metrics"].items()):
        value = metric["value"]
        counts = f"({metric['numerator']:g}/{metric['denominator']:g})"
        print(f"  {name:<34} {_fmt(value):>8}  {counts}")
    if run.get("adjudication"):
        print(
            "  These are the deterministic numbers and are NOT adjusted by "
            "adjudication.\n  Adjudication outcomes are reported separately below, so a "
            "model's\n  judgments never silently move the headline metrics."
        )

    label = (
        "Findings (after adjudication)" if run.get("adjudication") else "Findings"
    )
    print(f"\n-- {label} " + "-" * max(4, 68 - len(label)))
    summary = run["findings_summary"]
    if not summary["by_type"]:
        print("  none")
    for finding_type, count in summary["by_type"].items():
        print(f"  {finding_type:<34} {count:>4}")
    if summary["by_severity"]:
        print("  by severity: " + ", ".join(
            f"{level}={count}" for level, count in summary["by_severity"].items()
        ))

    adjudication = run.get("adjudication")
    if adjudication:
        print("\n-- LLM adjudication --------------------------------------------------")
        stats = adjudication["summary"]
        print(f"  findings considered   : {stats['findings_considered']}")
        print(f"  routed to LLM         : {stats['routed_to_llm']}")
        print(f"  skipped (high conf.)  : {stats['skipped_high_confidence']}")
        print(f"  adjudication rate     : {_fmt(stats['adjudication_rate'])}")
        print(f"  agreed                : {stats['agreed']}")
        print(f"  resolved by LLM (R2)  : {stats['resolved_by_llm']}")
        print(f"  ADJUDICATION_REQUIRED : {stats['adjudication_required']}")
        print(f"  failed                : {stats['failed']}")
        print(f"  disagreement rate     : {_fmt(stats['disagreement_rate'])}")
        print(f"  parse failures        : {stats['parse_failures']}")
        print(f"  estimated cost USD    : {_fmt(stats['estimated_cost_usd'])}")

    print(f"\nRun saved to {run_path}")
    print(
        "\nNote: deterministic findings are authoritative. No overall risk score or "
        "deployment recommendation is produced - that comes after the evaluation "
        "stack is validated."
    )


if __name__ == "__main__":
    raise SystemExit(main())
