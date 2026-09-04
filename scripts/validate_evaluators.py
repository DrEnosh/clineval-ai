"""Validate the evaluators against controlled failure injection.

    python -m scripts.validate_evaluators

Two questions, both of which a detector must answer:

**Does it stay quiet when nothing is wrong?** The reference summary is correct
by construction, so every finding against it is a false positive. Reported as
the false-positive rate.

**Does it fire when something is wrong?** Each injection corrupts a correct
summary in one known way and declares what should be detected. Reported per
injection as detected / missed / skipped.

Injections that cannot apply to a case (a dose change needs a medication) are
reported as *skipped*, never as passes. Counting an injection that never
happened as a detection success would be precisely the self-flattery this
harness exists to prevent.

Exits non-zero if the false-positive rate or any detection rate breaches its
threshold, so it can gate CI.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark import generate_benchmark, load_generated_benchmark
from app.core.enums import FindingType
from app.evaluation.injection import INJECTIONS, inject
from app.evaluation.providers import ReferenceSummaryProvider
from app.evaluation.runner import EvaluationRunner
from app.models.schemas import AIOutput, BenchmarkCase

# Detection thresholds. An injection detected below this share of applicable
# cases is a failure of the evaluator, not of the benchmark.
DEFAULT_MIN_DETECTION_RATE = 0.90
DEFAULT_MAX_FALSE_POSITIVE_RATE = 0.02


@dataclass
class InjectionOutcome:
    name: str
    expected: FindingType
    detected: int = 0
    missed: int = 0
    skipped: int = 0
    missed_cases: list[str] = field(default_factory=list)

    @property
    def applicable(self) -> int:
        return self.detected + self.missed

    @property
    def detection_rate(self) -> float | None:
        return self.detected / self.applicable if self.applicable else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.validate_evaluators",
        description="Validate evaluators against controlled failure injection.",
    )
    parser.add_argument("--cases", type=int, default=50, help="Cases to validate against")
    parser.add_argument("--seed", type=int, default=42, help="Benchmark seed")
    parser.add_argument("--cases-file", type=Path, default=None, help="Use a saved benchmark")
    parser.add_argument(
        "--min-detection-rate", type=float, default=DEFAULT_MIN_DETECTION_RATE,
        help="Minimum acceptable detection rate per injection",
    )
    parser.add_argument(
        "--max-false-positive-rate", type=float, default=DEFAULT_MAX_FALSE_POSITIVE_RATE,
        help="Maximum acceptable findings per case against the reference oracle",
    )
    return parser.parse_args(argv)


def run_false_positive_check(
    cases: list[BenchmarkCase], runner: EvaluationRunner
) -> tuple[int, list]:
    """Evaluate the oracle: findings here are false positives by definition."""
    outputs = ReferenceSummaryProvider().generate_all(cases)
    result = runner.evaluate_benchmark(cases, outputs, model_id="reference-oracle")
    return len(result.findings), result.findings


def run_injection_checks(
    cases: list[BenchmarkCase], runner: EvaluationRunner
) -> dict[str, InjectionOutcome]:
    """Inject each known defect into each case and check it is detected."""
    outcomes: dict[str, InjectionOutcome] = {}

    for name in sorted(INJECTIONS):
        outcome: InjectionOutcome | None = None

        for case in cases:
            injection = inject(name, case)
            if injection is None:
                if outcome is None:
                    outcome = InjectionOutcome(name=name, expected=FindingType.OTHER)
                outcome.skipped += 1
                continue

            if outcome is None or outcome.expected is FindingType.OTHER:
                previous = outcome
                outcome = InjectionOutcome(name=name, expected=injection.expected_finding)
                if previous is not None:
                    outcome.skipped = previous.skipped

            output = AIOutput(
                case_id=case.case_id,
                model_id=f"injected-{name}",
                summary_text=injection.corrupted_summary,
                raw_metadata={"provider_kind": "synthetic_test_provider", "injection": name},
            )
            case_result = runner.evaluate_case(case, output)
            found = {finding.finding_type for finding in case_result.findings}

            if found & injection.acceptable_findings:
                outcome.detected += 1
            else:
                outcome.missed += 1
                outcome.missed_cases.append(case.case_id)

        outcomes[name] = outcome or InjectionOutcome(name=name, expected=FindingType.OTHER)

    return outcomes


def print_report(
    case_count: int,
    false_positives: int,
    outcomes: dict[str, InjectionOutcome],
    min_detection_rate: float,
    max_false_positive_rate: float,
) -> bool:
    """Print the validation report. Returns True when everything passed."""
    print("=" * 78)
    print("ClinEval AI - evaluator self-validation")
    print("=" * 78)
    print(f"Cases: {case_count}")
    print()

    print("-- False positives (reference summary, correct by construction) ------------")
    rate = false_positives / case_count if case_count else 0.0
    fp_ok = rate <= max_false_positive_rate
    print(f"  Findings against correct summaries : {false_positives}")
    print(f"  Findings per case                  : {rate:.4f}  (max {max_false_positive_rate})")
    print(f"  Result                             : {'PASS' if fp_ok else 'FAIL'}")
    print()

    print("-- Controlled failure injection --------------------------------------------")
    header = f"  {'Injection':<32}{'Expected':<30}{'Det':>5}{'Miss':>6}{'Skip':>6}{'Rate':>8}  Result"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_ok = fp_ok
    for name in sorted(outcomes):
        outcome = outcomes[name]
        detection_rate = outcome.detection_rate
        if detection_rate is None:
            verdict = "SKIPPED"
            rate_text = "n/a"
            # Not a pass: an injection that never applied validated nothing.
            all_ok = False
        else:
            passed = detection_rate >= min_detection_rate
            verdict = "PASS" if passed else "FAIL"
            rate_text = f"{detection_rate:.2%}"
            all_ok = all_ok and passed

        print(
            f"  {name:<32}{outcome.expected.value:<30}"
            f"{outcome.detected:>5}{outcome.missed:>6}{outcome.skipped:>6}"
            f"{rate_text:>8}  {verdict}"
        )

    print()
    for name in sorted(outcomes):
        outcome = outcomes[name]
        if outcome.missed_cases:
            shown = ", ".join(outcome.missed_cases[:8])
            more = f" (+{len(outcome.missed_cases) - 8} more)" if len(outcome.missed_cases) > 8 else ""
            print(f"  Missed by {name}: {shown}{more}")

    print()
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.cases_file:
        cases = load_generated_benchmark(args.cases_file)[: args.cases]
    else:
        cases = generate_benchmark(case_count=args.cases, seed=args.seed)

    runner = EvaluationRunner()
    false_positives, _ = run_false_positive_check(cases, runner)
    outcomes = run_injection_checks(cases, runner)

    passed = print_report(
        len(cases), false_positives, outcomes,
        args.min_detection_rate, args.max_false_positive_rate,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
