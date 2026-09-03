"""Audit a generated benchmark and report dataset quality.

    python -m scripts.audit_benchmark

Exits non-zero if any integrity check fails, so it can gate CI. With no
arguments it audits the newest benchmark under data/synthetic/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark import load_generated_benchmark, validate_benchmark
from app.core.benchmark_config import load_benchmark_config
from app.core.config import settings
from app.models.schemas import BenchmarkCase


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.audit_benchmark",
        description="Report dataset quality for a generated benchmark and verify its integrity.",
    )
    parser.add_argument("--cases-file", type=Path, default=None, help="Path to a benchmark cases JSON")
    parser.add_argument("--metadata-file", type=Path, default=None, help="Path to its metadata JSON")
    parser.add_argument("--base-dir", type=Path, default=None, help="Directory to search for benchmarks")
    parser.add_argument("--max-issues", type=int, default=25, help="How many issues to print")
    return parser.parse_args(argv)


def find_latest_benchmark(base_dir: Path) -> tuple[Path, Path | None] | None:
    """Locate the most recently modified benchmark, with its metadata if present."""
    candidates = sorted(
        base_dir.glob("*/cases_seed*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        return None
    cases_path = candidates[0]
    metadata_path = cases_path.parent / cases_path.name.replace("cases_", "metadata_", 1)
    return cases_path, (metadata_path if metadata_path.exists() else None)


def _percent(part: int, whole: int) -> str:
    return f"{(100 * part / whole):.1f}%" if whole else "n/a"


def summarize(cases: list[BenchmarkCase]) -> dict:
    """Recompute descriptive statistics directly from the cases.

    Deliberately recomputed rather than read from the metadata file - an audit
    that trusted the metadata could not detect metadata that had drifted from
    the cases.
    """
    from app.benchmark.synthesizer import build_metadata

    config = load_benchmark_config()
    return build_metadata(cases, master_seed(cases), config)


def master_seed(cases: list[BenchmarkCase]) -> str:
    """Recover the run's master seed from a case's per-case seed.

    Per-case seeds are "<master>:<index>", so the master is the part before
    the colon - reporting "42:0" would wrongly suggest the run was seeded with
    a value that cannot be passed back to --seed.
    """
    if not cases:
        return "unknown"
    per_case = str(cases[0].generation_metadata.get("seed", "unknown"))
    return per_case.split(":", 1)[0] if ":" in per_case else per_case


def print_report(cases: list[BenchmarkCase], stats: dict, issues: list) -> None:
    total = len(cases)
    print("=" * 68)
    print("ClinEval AI - benchmark quality report")
    print("=" * 68)
    print(f"Benchmark version : {stats['benchmark_version']}")
    print(f"Generator version : {stats['generator_version']}")
    print(f"Seed              : {stats['seed']}")
    print(f"Total cases       : {total}")
    print(f"Real patient data : {'YES - INVESTIGATE' if stats['contains_real_patient_data'] else 'none (synthetic)'}")

    print("\n-- Cases by difficulty ---------------------------------------------")
    for tier, count in stats["distribution"]["by_difficulty"].items():
        print(f"  {tier:<16} {count:>4}  ({_percent(count, total)})")

    print("\n-- Cases by failure-mode category ----------------------------------")
    for category, count in stats["distribution"]["by_category"].items():
        print(f"  {category:<30} {count:>4}  ({_percent(count, total)})")

    print("\n-- Annotated fact counts -------------------------------------------")
    for key, value in stats["counts"].items():
        print(f"  {key:<30} {value:>5}")

    print("\n-- Cases containing ------------------------------------------------")
    for key, value in stats["cases_containing"].items():
        print(f"  {key:<30} {value:>4}  ({_percent(value, total)})")

    print("\n-- Averages --------------------------------------------------------")
    print(f"  {'note length (characters)':<30} {stats['averages']['note_characters']:>7}")
    print(f"  {'summary length (characters)':<30} {stats['averages']['summary_characters']:>7}")
    print(f"  {'annotated facts per case':<30} {stats['averages']['facts_per_case']:>7}")

    print("\n-- Integrity -------------------------------------------------------")
    if not issues:
        print("  All integrity checks PASSED.")
    else:
        print(f"  {len(issues)} issue(s) found:")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = args.base_dir or settings.benchmark_dir

    if args.cases_file:
        cases_path = args.cases_file
        metadata_path = args.metadata_file
    else:
        found = find_latest_benchmark(base_dir)
        if found is None:
            print(
                f"No generated benchmark found under {base_dir}.\n"
                f"Run: python -m scripts.generate_benchmark --cases 100 --seed 42",
                file=sys.stderr,
            )
            return 1
        cases_path, metadata_path = found

    try:
        cases = load_generated_benchmark(cases_path)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if not cases:
        print(f"Error: {cases_path} contains no cases.", file=sys.stderr)
        return 1

    metadata = None
    if metadata_path and Path(metadata_path).exists():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))

    issues = validate_benchmark(cases, metadata)
    stats = summarize(cases)

    print(f"Source: {cases_path}\n")
    print_report(cases, stats, issues)

    for issue in issues[: args.max_issues]:
        print(f"    {issue}")
    if len(issues) > args.max_issues:
        print(f"    ... and {len(issues) - args.max_issues} more")

    print()
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
