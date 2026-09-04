"""End-to-end evaluation against the Phase 2 benchmark.

The evaluator is tested here the way it will be used: over real generated
cases, in both directions.

**Specificity.** The reference summary is correct by construction, so any
finding against it is a false positive. This is the half most easily
overlooked when building a detector, and it is asserted at zero.

**Sensitivity.** Each controlled injection corrupts a correct summary in one
known way; the evaluator must detect it. Injections that cannot apply to a
case are skipped, never counted as passes.

Both halves are needed. An evaluator that flags everything scores perfectly on
sensitivity, and one that flags nothing scores perfectly on specificity.
"""

from __future__ import annotations

import pytest

from app.benchmark import generate_benchmark
from app.core.enums import FindingType
from app.evaluation.injection import INJECTIONS, inject, inject_all
from app.evaluation.providers import (
    EmptySummaryProvider,
    ReferenceSummaryProvider,
    SourceNoteProvider,
)
from app.evaluation.runner import EvaluationRunner
from app.models.schemas import AIOutput

BENCHMARK_SIZE = 40


@pytest.fixture(scope="module")
def cases():
    return generate_benchmark(case_count=BENCHMARK_SIZE, seed=42)


@pytest.fixture(scope="module")
def runner():
    return EvaluationRunner()


@pytest.fixture(scope="module")
def oracle_result(cases, runner):
    outputs = ReferenceSummaryProvider().generate_all(cases)
    return runner.evaluate_benchmark(cases, outputs, model_id="reference-oracle")


class TestSpecificity:
    """The correct summary must produce no findings."""

    def test_no_false_positives_against_the_reference_summary(self, oracle_result):
        assert oracle_result.findings == [], "\n".join(
            f"{f.case_id} {f.finding_type.value}: {f.explanation}"
            for f in oracle_result.findings[:10]
        )

    def test_perfect_critical_retention_on_correct_summaries(self, oracle_result):
        assert oracle_result.metrics["critical_information_retention"].value == 1.0

    def test_perfect_medication_consistency_on_correct_summaries(self, oracle_result):
        assert oracle_result.metrics["medication_consistency"].value == 1.0

    def test_perfect_negation_consistency_on_correct_summaries(self, oracle_result):
        assert oracle_result.metrics["negation_consistency"].value == 1.0

    def test_perfect_temporal_consistency_on_correct_summaries(self, oracle_result):
        assert oracle_result.metrics["temporal_consistency"].value == 1.0

    def test_no_hallucinations_detected_in_correct_summaries(self, oracle_result):
        assert oracle_result.metrics["unsupported_fact_rate"].value == 0.0

    def test_every_case_was_evaluated(self, oracle_result):
        assert oracle_result.cases_evaluated == BENCHMARK_SIZE
        assert oracle_result.configuration["cases_skipped_no_output"] == []

    def test_verbatim_note_retains_content_without_being_a_summary(self, cases, runner):
        """A degenerate 'summarizer' that copies its input scores well on
        retention while failing to summarize - which is why retention alone is
        not a quality measure."""
        outputs = SourceNoteProvider().generate_all(cases)
        result = runner.evaluate_benchmark(cases, outputs, model_id="verbatim-note")
        assert result.metrics["critical_information_retention"].value > 0.9


class TestSensitivity:
    """Each injected defect must be detected."""

    @pytest.mark.parametrize("injection_name", sorted(INJECTIONS))
    def test_injection_is_detected_wherever_it_applies(self, cases, runner, injection_name):
        detected = missed = skipped = 0
        misses: list[str] = []

        for case in cases:
            injection = inject(injection_name, case)
            if injection is None:
                skipped += 1
                continue
            output = AIOutput(
                case_id=case.case_id, model_id=f"injected-{injection_name}",
                summary_text=injection.corrupted_summary,
            )
            found = {f.finding_type for f in runner.evaluate_case(case, output).findings}
            if found & injection.acceptable_findings:
                detected += 1
            else:
                missed += 1
                misses.append(f"{case.case_id}: {injection.description}")

        applicable = detected + missed
        assert applicable > 0, (
            f"'{injection_name}' applied to no case, so it validated nothing. "
            f"Skipped {skipped}."
        )
        rate = detected / applicable
        assert rate >= 0.9, (
            f"'{injection_name}' detected in only {rate:.0%} of {applicable} cases.\n"
            + "\n".join(misses[:5])
        )

    def test_every_injection_applies_somewhere_in_the_benchmark(self, cases):
        """An injection that never applies is dead validation code."""
        applied: set[str] = set()
        for case in cases:
            for name, injection in inject_all(case).items():
                if injection is not None:
                    applied.add(name)
        assert applied == set(INJECTIONS)

    def test_injections_do_not_alter_the_case(self, cases):
        """Injection corrupts a copy of the summary, never the ground truth."""
        case = cases[0]
        before = case.model_dump(mode="json")
        inject_all(case)
        assert case.model_dump(mode="json") == before

    def test_injected_summaries_differ_from_the_reference(self, cases):
        for case in cases[:10]:
            for injection in inject_all(case).values():
                if injection is not None:
                    assert injection.corrupted_summary != case.reference_summary


class TestDegenerateOutputs:
    def test_empty_output_is_flagged_on_every_case(self, cases, runner):
        outputs = EmptySummaryProvider().generate_all(cases)
        result = runner.evaluate_benchmark(cases, outputs, model_id="empty-output")
        assert result.cases_evaluated == len(cases)
        for case_result in result.case_results:
            assert any(
                f.finding_type is FindingType.EMPTY_OUTPUT for f in case_result.findings
            )

    def test_empty_output_scores_zero_retention_not_undefined(self, cases, runner):
        outputs = EmptySummaryProvider().generate_all(cases)
        result = runner.evaluate_benchmark(cases, outputs, model_id="empty-output")
        assert result.metrics["critical_information_retention"].value == 0.0

    def test_empty_output_asserts_no_hallucinations(self, cases, runner):
        """Nothing asserted means nothing unsupported - undefined, not zero."""
        outputs = EmptySummaryProvider().generate_all(cases)
        result = runner.evaluate_benchmark(cases, outputs, model_id="empty-output")
        assert result.metrics.get("unsupported_fact_rate") is None


class TestProviderLabelling:
    def test_synthetic_outputs_are_labelled_as_such(self, cases):
        """No fabricated model result may be mistaken for a real generation."""
        for provider in (ReferenceSummaryProvider(), EmptySummaryProvider(), SourceNoteProvider()):
            output = provider.generate(cases[0])
            assert output.raw_metadata["is_real_model_output"] is False
            assert output.raw_metadata["provider_kind"] == "synthetic_test_provider"

    def test_provider_name_identifies_the_source(self, cases):
        assert ReferenceSummaryProvider().generate(cases[0]).model_id == "reference-oracle"


class TestFindingQuality:
    """Every finding must be explainable from evidence."""

    @pytest.fixture(scope="class")
    @classmethod
    def corrupted_findings(cls, cases):
        runner = EvaluationRunner()
        findings = []
        for case in cases[:15]:
            for injection in inject_all(case).values():
                if injection is None:
                    continue
                output = AIOutput(
                    case_id=case.case_id, model_id="injected",
                    summary_text=injection.corrupted_summary,
                )
                findings.extend(runner.evaluate_case(case, output).findings)
        return findings

    def test_findings_were_produced(self, corrupted_findings):
        assert corrupted_findings

    def test_every_finding_identifies_its_evaluator_and_version(self, corrupted_findings):
        for finding in corrupted_findings:
            assert finding.evaluator_name
            assert finding.evaluator_version

    def test_every_finding_has_a_severity_and_an_explanation(self, corrupted_findings):
        for finding in corrupted_findings:
            assert finding.severity is not None
            assert finding.explanation.strip()

    def test_every_finding_is_traceable_to_the_case(self, corrupted_findings):
        for finding in corrupted_findings:
            assert finding.case_id
            assert finding.finding_id.startswith(finding.case_id)

    def test_content_findings_carry_source_evidence(self, corrupted_findings):
        """Operational findings have no source fact; clinical ones must."""
        operational = {
            FindingType.EMPTY_OUTPUT, FindingType.MALFORMED_OUTPUT,
            FindingType.MISSING_ANNOTATION, FindingType.UNSUPPORTED_ADDITION,
            FindingType.MEDICATION_HALLUCINATION,
        }
        for finding in corrupted_findings:
            if finding.finding_type in operational:
                continue
            assert finding.source_evidence, f"{finding.finding_type}: {finding.explanation}"

    def test_confidence_is_within_bounds(self, corrupted_findings):
        for finding in corrupted_findings:
            assert 0.0 <= finding.confidence <= 1.0


class TestReproducibility:
    def test_evaluating_twice_gives_identical_results(self, cases, runner):
        outputs = ReferenceSummaryProvider().generate_all(cases[:10])
        subset = cases[:10]
        first = runner.evaluate_benchmark(subset, outputs, model_id="m", evaluation_run_id="R")
        second = runner.evaluate_benchmark(subset, outputs, model_id="m", evaluation_run_id="R")
        assert [f.model_dump() for f in first.findings] == [
            f.model_dump() for f in second.findings
        ]
        assert {k: v.value for k, v in first.metrics.items()} == {
            k: v.value for k, v in second.metrics.items()
        }
