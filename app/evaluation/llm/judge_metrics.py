"""Evaluating the LLM judge itself.

The judge is a component under test like any other, so it gets the same
treatment as the deterministic evaluators: labelled cases with ground truth
from the benchmark, and per-class accuracy, precision, recall, F1, and a
confusion matrix.

Two properties this implementation is careful about:

**Undefined is undefined.** Precision with no predictions of a class is not
0.0 and not 1.0 — it has no value. Reporting 0.0 would make a judge that never
predicts CONTRADICTED look actively bad at it rather than silent about it, and
those are different failures.

**Labels come from the benchmark, never from the judge.** Ground truth for
these cases is derived from structured annotations, which is the only way
"the judge was right" means anything.

Results describe this synthetic benchmark. They do not generalize.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.enums import AdjudicationJudgment


@dataclass
class ClassMetrics:
    """Per-class counts and derived rates."""

    label: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def support(self) -> int:
        """How many cases genuinely have this label."""
        return self.true_positives + self.false_negatives

    @property
    def predicted(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def precision(self) -> float | None:
        if not self.predicted:
            return None
        return round(self.true_positives / self.predicted, 4)

    @property
    def recall(self) -> float | None:
        if not self.support:
            return None
        return round(self.true_positives / self.support, 4)

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or (precision + recall) == 0:
            return None
        return round(2 * precision * recall / (precision + recall), 4)

    def report(self) -> dict:
        return {
            "label": self.label,
            "support": self.support,
            "predicted": self.predicted,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class JudgeEvaluation:
    """Aggregate judge performance against benchmark-derived labels."""

    total: int = 0
    correct: int = 0
    unparseable: int = 0
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)

    @property
    def accuracy(self) -> float | None:
        if not self.total:
            return None
        return round(self.correct / self.total, 4)

    def record(self, expected: AdjudicationJudgment, predicted: AdjudicationJudgment | None) -> None:
        """Record one judged case. `predicted=None` means the response was
        unusable, which is counted as an error rather than dropped."""
        self.total += 1
        expected_label = expected.value

        if predicted is None:
            self.unparseable += 1
            predicted_label = "UNPARSEABLE"
        else:
            predicted_label = predicted.value
            if predicted is expected:
                self.correct += 1

        key = (expected_label, predicted_label)
        self.confusion[key] = self.confusion.get(key, 0) + 1

        for label in {expected_label, predicted_label} | set(self.per_class):
            if label == "UNPARSEABLE":
                continue
            metrics = self.per_class.setdefault(label, ClassMetrics(label=label))
            if expected_label == label and predicted_label == label:
                metrics.true_positives += 1
            elif predicted_label == label:
                metrics.false_positives += 1
            elif expected_label == label:
                metrics.false_negatives += 1
            else:
                metrics.true_negatives += 1

    def confusion_matrix(self) -> dict[str, dict[str, int]]:
        """Nested expected -> predicted -> count."""
        matrix: dict[str, dict[str, int]] = {}
        for (expected, predicted), count in sorted(self.confusion.items()):
            matrix.setdefault(expected, {})[predicted] = count
        return matrix

    def macro_f1(self) -> float | None:
        """Unweighted mean F1 across classes that have a defined F1."""
        scores = [
            metrics.f1 for metrics in self.per_class.values() if metrics.f1 is not None
        ]
        return round(sum(scores) / len(scores), 4) if scores else None

    def report(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1(),
            "unparseable": self.unparseable,
            "per_class": {
                label: metrics.report()
                for label, metrics in sorted(self.per_class.items())
            },
            "confusion_matrix": self.confusion_matrix(),
            "scope_note": (
                "Measured on the synthetic ClinEval benchmark only. These figures "
                "do not generalize to real clinical documentation."
            ),
        }
