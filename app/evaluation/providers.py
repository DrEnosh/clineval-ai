"""Summary providers - where evaluated text comes from.

    TEST AND VALIDATION USE ONLY. Nothing here is a model.

No real model has been evaluated in this project, and these providers exist so
the evaluation engine can be exercised and validated without one. Every output
they produce is derived from the benchmark itself and is labeled as such in
`AIOutput.raw_metadata["provider"]`, so a synthetic output can never be
mistaken for a model generation in stored results.

Two providers:

- `ReferenceSummaryProvider` - returns the benchmark's own reference summary.
  This is the *oracle*: a correct summary, against which the evaluator should
  report essentially no findings. It measures the evaluator's false-positive
  rate, which is the thing most easily overlooked when building a detector.

- `CorruptedSummaryProvider` - returns a reference summary with one controlled
  defect injected (see app/evaluation/injection.py). It measures the
  evaluator's sensitivity.

Together they are how the evaluator gets evaluated. Real model outputs arrive
in Phase 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.schemas import AIOutput, BenchmarkCase

# Stamped into raw_metadata on every generated output so synthetic text is
# always identifiable downstream.
SYNTHETIC_MARKER = "synthetic_test_provider"


class SummaryProvider(ABC):
    """Supplies the summary text to be evaluated for a case."""

    name: str

    @abstractmethod
    def generate(self, case: BenchmarkCase, run_index: int = 0) -> AIOutput:
        raise NotImplementedError

    def generate_all(self, cases: list[BenchmarkCase]) -> dict[str, AIOutput]:
        return {case.case_id: self.generate(case) for case in cases}

    def _output(
        self, case: BenchmarkCase, text: str, run_index: int, **metadata
    ) -> AIOutput:
        return AIOutput(
            case_id=case.case_id,
            model_id=self.name,
            summary_text=text,
            run_index=run_index,
            raw_metadata={
                "provider": self.name,
                "provider_kind": SYNTHETIC_MARKER,
                "is_real_model_output": False,
                **metadata,
            },
        )


class ReferenceSummaryProvider(SummaryProvider):
    """Returns the benchmark's own reference summary verbatim.

    The upper bound: a summary that is correct by construction. Findings
    against it are, by definition, evaluator false positives.
    """

    name = "reference-oracle"

    def generate(self, case: BenchmarkCase, run_index: int = 0) -> AIOutput:
        return self._output(case, case.reference_summary, run_index, corruption=None)


class EmptySummaryProvider(SummaryProvider):
    """Returns nothing, for exercising empty-output handling."""

    name = "empty-output"

    def generate(self, case: BenchmarkCase, run_index: int = 0) -> AIOutput:
        return self._output(case, "", run_index, corruption="empty")


class SourceNoteProvider(SummaryProvider):
    """Returns the full source note instead of a summary.

    A degenerate 'summarizer' that copies its input. Useful as a control: it
    should score near-perfectly on retention while failing to be a summary at
    all, which illustrates why retention alone is not a quality measure.
    """

    name = "verbatim-note"

    def generate(self, case: BenchmarkCase, run_index: int = 0) -> AIOutput:
        return self._output(case, case.clinical_note, run_index, corruption=None)
