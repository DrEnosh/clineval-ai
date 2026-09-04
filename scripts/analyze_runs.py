"""Error analysis over stored experiment artifacts.

    python -m scripts.analyze_runs

Reads the experiment artifacts under `data/runs/experiments/` and reports
recurring failure modes, evaluator/adjudicator disagreement, model-specific
error profiles, and the cases most worth a human's attention.

**Nothing is labelled a false positive here.** The script surfaces *potential*
false positives and negatives - findings whose evidence is weak, or where the
deterministic evaluator and the LLM disagreed - and says explicitly that a
person has not verified them. Automatically declaring a finding wrong would be
the same category of error the whole project is built to avoid: asserting a
conclusion the evidence does not support.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings

VERIFICATION_NOTICE = (
    "Items below are flagged for human review. None has been verified by a "
    "person, so each is a POTENTIAL false positive or negative, not a "
    "confirmed one."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.analyze_runs",
        description="Analyze stored experiment artifacts for failure modes and disagreement.",
    )
    parser.add_argument("--dir", type=Path, default=None, help="Experiment artifact directory")
    parser.add_argument("--experiment", type=Path, default=None, help="A single artifact file")
    parser.add_argument("--out", type=Path, default=None, help="Write the analysis as JSON")
    parser.add_argument("--top", type=int, default=10, help="How many items to list per section")
    return parser.parse_args(argv)


def load_artifacts(args: argparse.Namespace) -> list[tuple[Path, dict]]:
    if args.experiment:
        paths = [args.experiment]
    else:
        directory = args.dir or (settings.runs_dir / "experiments")
        if not directory.exists():
            return []
        paths = sorted(
            (p for p in directory.glob("exp-*.json") if not p.name.endswith("-review.json")),
            key=lambda p: p.stat().st_mtime,
        )
    artifacts = []
    for path in paths:
        try:
            artifacts.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as error:
            print(f"Skipping unreadable artifact {path}: {error}", file=sys.stderr)
    return artifacts


def analyze(artifacts: list[tuple[Path, dict]]) -> dict:
    """Aggregate failure modes and disagreement across artifacts."""
    failure_modes: Counter[str] = Counter()
    per_model: dict[str, Counter[str]] = {}
    severity: Counter[str] = Counter()
    adjudication_disagreements: list[dict] = []
    anchoring: list[dict] = []
    generation_errors: list[str] = []
    models_seen: dict[str, dict] = {}
    any_real_output = False

    for path, artifact in artifacts:
        for model in artifact.get("models", []):
            spec = model["model"]
            label = spec["label"]
            models_seen[label] = spec
            if spec.get("is_real_model_output"):
                any_real_output = True

            counts = per_model.setdefault(label, Counter())
            for group, count in model.get("failure_taxonomy", {}).items():
                if count:
                    failure_modes[group] += count
                    counts[group] += count
            for level, count in model.get("severity_counts", {}).items():
                if count:
                    severity[level] += count
            generation_errors.extend(model.get("generation_errors", []))

            comparison = model.get("adjudication_comparison") or {}
            for entry in comparison.get("disagreement_detail", []):
                anchoring.append({"artifact": path.name, "model": label, **entry})
            if comparison:
                anchoring_signal = (
                    comparison.get("judgment_agreement_rate") is not None
                    and comparison["judgment_agreement_rate"] < 1.0
                ) or comparison.get("withdrawal_delta", 0) != 0
                adjudication_disagreements.append(
                    {
                        "artifact": path.name,
                        "model": label,
                        "judged_in_both": comparison.get("findings_judged_in_both"),
                        "agreement_rate": comparison.get("judgment_agreement_rate"),
                        "withdrawal_delta": comparison.get("withdrawal_delta"),
                        "anchoring_signal": anchoring_signal,
                    }
                )

    return {
        "artifacts_analyzed": len(artifacts),
        "models": models_seen,
        "any_real_model_output": any_real_output,
        "recurring_failure_modes": dict(failure_modes.most_common()),
        "per_model_failure_modes": {
            label: dict(counts.most_common()) for label, counts in sorted(per_model.items())
        },
        "severity_totals": dict(severity.most_common()),
        "adjudication_conditions": adjudication_disagreements,
        "anchoring_disagreements": anchoring,
        "generation_errors": generation_errors,
    }


def analyze_review_sets(args: argparse.Namespace, top: int) -> dict:
    """Pull potential false positives and negatives from the review artifacts."""
    directory = args.dir or (settings.runs_dir / "experiments")
    if args.experiment:
        review_paths = [args.experiment.with_name(args.experiment.stem + "-review.json")]
    elif directory.exists():
        review_paths = sorted(directory.glob("exp-*-review.json"))
    else:
        review_paths = []

    potential_false_positives: list[dict] = []
    potential_false_negatives: list[dict] = []
    needs_review: list[dict] = []

    for path in review_paths:
        if not path.exists():
            continue
        try:
            review = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        for entry in review.get("entries", []):
            finding = entry["deterministic_finding"]
            judgments = entry.get("llm_judgment") or {}

            # A weak match is the classic shape of a false positive: the
            # evaluator concluded something on thin evidence.
            weak = finding.get("match_level") in (None, "none", "partial")
            llm_says_supported = any(
                j.get("judgment") == "SUPPORTED" for j in judgments.values()
            )

            if llm_says_supported:
                potential_false_positives.append(
                    {
                        "case_id": entry["case_id"],
                        "model": entry["model"],
                        "finding_type": finding["type"],
                        "why": "LLM judged the content SUPPORTED against a deterministic finding",
                        "deterministic_confidence": finding["confidence"],
                        "source_evidence": finding["source_evidence"],
                    }
                )
            elif weak and finding["type"].endswith("omission"):
                potential_false_positives.append(
                    {
                        "case_id": entry["case_id"],
                        "model": entry["model"],
                        "finding_type": finding["type"],
                        "why": "omission concluded from a weak or absent match",
                        "deterministic_confidence": finding["confidence"],
                        "source_evidence": finding["source_evidence"],
                    }
                )

            if any(j.get("judgment") == "CONTRADICTED" for j in judgments.values()) and (
                finding["type"].endswith("omission")
            ):
                potential_false_negatives.append(
                    {
                        "case_id": entry["case_id"],
                        "model": entry["model"],
                        "finding_type": finding["type"],
                        "why": "LLM judged CONTRADICTED where the evaluator saw only omission",
                        "source_evidence": finding["source_evidence"],
                    }
                )

            if entry["selection_reason"] == "deterministic/LLM disagreement":
                needs_review.append(
                    {
                        "case_id": entry["case_id"],
                        "model": entry["model"],
                        "finding_type": finding["type"],
                        "severity": finding["severity"],
                        "disposition": entry["final_disposition"],
                    }
                )

    return {
        "review_sets_analyzed": len(review_paths),
        "potential_false_positives": potential_false_positives[:top],
        "potential_false_positive_count": len(potential_false_positives),
        "potential_false_negatives": potential_false_negatives[:top],
        "potential_false_negative_count": len(potential_false_negatives),
        "cases_requiring_human_review": needs_review[:top],
        "cases_requiring_human_review_count": len(needs_review),
        "verification_notice": VERIFICATION_NOTICE,
    }


def print_report(analysis: dict, review: dict, top: int) -> None:
    print("=" * 78)
    print("ClinEval AI - experiment error analysis")
    print("=" * 78)
    print(f"Artifacts analyzed : {analysis['artifacts_analyzed']}")
    print(f"Models seen        : {', '.join(sorted(analysis['models'])) or 'none'}")
    print(
        f"Real model output  : "
        f"{'yes' if analysis['any_real_model_output'] else 'NO - all results are from the mock provider'}"
    )

    print("\n-- Recurring failure modes (all models) ------------------------------")
    if not analysis["recurring_failure_modes"]:
        print("  none")
    for group, count in list(analysis["recurring_failure_modes"].items())[:top]:
        print(f"  {group:<34} {count:>4}")

    print("\n-- Per-model failure profile -----------------------------------------")
    for label, counts in analysis["per_model_failure_modes"].items():
        top_modes = ", ".join(f"{k}={v}" for k, v in list(counts.items())[:5]) or "none"
        print(f"  {label:<28} {top_modes}")

    print("\n-- Severity totals ----------------------------------------------------")
    print(f"  {analysis['severity_totals'] or 'none'}")

    print("\n-- Blind vs informed adjudication -------------------------------------")
    if not analysis["adjudication_conditions"]:
        print("  no adjudication was run")
    for entry in analysis["adjudication_conditions"]:
        signal = "ANCHORING SIGNAL" if entry["anchoring_signal"] else "no difference detected"
        print(
            f"  {entry['model']:<24} judged={entry['judged_in_both']:<4} "
            f"agreement={entry['agreement_rate']}  delta={entry['withdrawal_delta']}  {signal}"
        )
    if analysis["anchoring_disagreements"]:
        print(f"\n  Judgment differences ({len(analysis['anchoring_disagreements'])}):")
        for entry in analysis["anchoring_disagreements"][:top]:
            print(
                f"    {entry['finding_id']}: informed={entry['informed']} "
                f"blind={entry['blind']}"
            )

    print("\n-- Potential false positives ------------------------------------------")
    print(f"  {review['potential_false_positive_count']} flagged")
    for entry in review["potential_false_positives"]:
        print(f"    {entry['case_id']} [{entry['finding_type']}] - {entry['why']}")

    print("\n-- Potential false negatives ------------------------------------------")
    print(f"  {review['potential_false_negative_count']} flagged")
    for entry in review["potential_false_negatives"]:
        print(f"    {entry['case_id']} [{entry['finding_type']}] - {entry['why']}")

    print("\n-- Cases requiring human review ---------------------------------------")
    print(f"  {review['cases_requiring_human_review_count']} flagged")
    for entry in review["cases_requiring_human_review"]:
        print(f"    {entry['case_id']} [{entry['finding_type']}] severity={entry['severity']}")

    if analysis["generation_errors"]:
        print("\n-- Generation errors ---------------------------------------------------")
        for error in analysis["generation_errors"][:top]:
            print(f"    {error}")

    print()
    print(VERIFICATION_NOTICE)
    if not analysis["any_real_model_output"]:
        print(
            "\nNOTE: no real model output is present in these artifacts. Nothing here "
            "is evidence about how any real model behaves."
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = load_artifacts(args)

    if not artifacts:
        print(
            "No experiment artifacts found. Run:\n"
            "  python -m scripts.run_experiment --models mock:demo --adjudicate",
            file=sys.stderr,
        )
        return 1

    analysis = analyze(artifacts)
    review = analyze_review_sets(args, args.top)
    print_report(analysis, review, args.top)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"analysis": analysis, "review": review}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nAnalysis written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
