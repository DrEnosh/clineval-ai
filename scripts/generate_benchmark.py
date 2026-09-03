"""Generate a synthetic clinical benchmark.

    python -m scripts.generate_benchmark --cases 100 --seed 42

The same command with the same seed produces a byte-identical benchmark.
Output is written under data/synthetic/<benchmark-version>/ and an existing
benchmark is never overwritten without --force.

The curated Phase 1 seed set (10 hand-authored cases) is regenerated with
--curated; it is a separate artifact with separate provenance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark import (
    BENCHMARK_VERSION,
    build_metadata,
    generate_benchmark,
    generate_default_benchmark,
    save_benchmark,
    save_generated_benchmark,
    validate_benchmark,
)
from app.core.benchmark_config import load_benchmark_config
from app.core.config import settings
from app.core.enums import DifficultyTier, FailureModeCategory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.generate_benchmark",
        description="Generate a reproducible synthetic clinical benchmark.",
    )
    parser.add_argument("--cases", type=int, default=None, help="Number of cases (default: from config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: from config)")
    parser.add_argument(
        "--difficulty", choices=[t.value for t in DifficultyTier], default=None,
        help="Force every case to one difficulty tier",
    )
    parser.add_argument(
        "--categories", nargs="+", choices=[c.value for c in FailureModeCategory], default=None,
        help="Restrict the failure-mode draw pool to these categories",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to a benchmark config YAML")
    parser.add_argument("--out", type=Path, default=None, help="Output base directory")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing benchmark version")
    parser.add_argument(
        "--curated", action="store_true",
        help="Regenerate the 10-case hand-authored Phase 1 seed set instead",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Write output even if integrity checks fail (not recommended)",
    )
    return parser.parse_args(argv)


def _generate_curated(base_dir: Path) -> int:
    cases = generate_default_benchmark()
    out_path = base_dir / f"benchmark_{BENCHMARK_VERSION}.json"
    save_benchmark(cases, out_path)
    print(f"Wrote {len(cases)} curated synthetic cases to {out_path}")
    print(f"Curated seed set version: {BENCHMARK_VERSION}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = args.out or settings.benchmark_dir

    if args.curated:
        return _generate_curated(base_dir)

    config = load_benchmark_config(args.config)
    seed = args.seed if args.seed is not None else config.default_seed
    difficulty = DifficultyTier(args.difficulty) if args.difficulty else None
    categories = [FailureModeCategory(c) for c in args.categories] if args.categories else None

    cases = generate_benchmark(
        case_count=args.cases, seed=seed, config=config,
        difficulty=difficulty, categories=categories,
    )
    metadata = build_metadata(cases, seed, config)

    issues = validate_benchmark(cases, metadata)
    if issues:
        print(f"Integrity checks found {len(issues)} issue(s):", file=sys.stderr)
        for issue in issues[:20]:
            print(f"  {issue}", file=sys.stderr)
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more", file=sys.stderr)
        if not args.skip_validation:
            print("\nRefusing to write a benchmark that fails integrity checks.", file=sys.stderr)
            print("Re-run with --skip-validation to write it anyway.", file=sys.stderr)
            return 1

    try:
        cases_path, metadata_path = save_generated_benchmark(
            cases, metadata, base_dir, force=args.force
        )
    except FileExistsError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {len(cases)} synthetic cases to {cases_path}")
    print(f"Wrote metadata to {metadata_path}")
    print(f"Benchmark version: {metadata['benchmark_version']}  seed: {seed}")
    print()
    print("Difficulty distribution:")
    for tier, count in metadata["distribution"]["by_difficulty"].items():
        print(f"  {tier:<14} {count}")
    print()
    print("Annotated facts:")
    for key, value in metadata["counts"].items():
        print(f"  {key:<26} {value}")
    print()
    print(f"Integrity checks: {'PASSED' if not issues else f'{len(issues)} issue(s)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
