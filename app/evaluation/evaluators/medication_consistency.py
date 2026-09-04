"""Medication consistency.

Compares the case's structured medication list against the output for:
omission, hallucination, wrong status, and wrong dose.

Status semantics are the point. This evaluator distinguishes current,
historical, discontinued, newly started, and dose-changed, because a summary
that lists a discontinued drug as current is not a wording problem - it risks
duplicate or contraindicated therapy. Comparison runs against
`MedicationStatus`, not against phrasing, with cue detection only to infer how
the *output* framed the drug.
"""

from __future__ import annotations

from app.benchmark import content
from app.core.enums import FactImportance, FindingType, SupportStatus
from app.evaluation import metrics
from app.evaluation.base import DeterministicEvaluator, EvaluatorResult
from app.evaluation.matching import FactMatcher
from app.evaluation.normalization import normalize_text
from app.evaluation.support import analyze_medication
from app.models.schemas import AIOutput, BenchmarkCase


class MedicationConsistencyEvaluator(DeterministicEvaluator):
    """Checks drug presence, status, and dose against the structured list."""

    name = "medication_consistency"
    version = "0.5.0"

    def evaluate(self, case: BenchmarkCase, ai_output: AIOutput) -> EvaluatorResult:
        structured = case.structured_case
        medications = structured.medications if structured else []

        empty = self.guard_output(case, ai_output)
        if empty is not None:
            empty.metrics = {
                "medication_consistency": metrics.compute("medication_consistency", 0, len(medications)),
                "medication_omission_rate": metrics.compute(
                    "medication_omission_rate", len(medications), len(medications)
                ),
                "medication_status_error_rate": metrics.compute("medication_status_error_rate", 0, 0),
            }
            return empty

        result = self._empty_result(case)
        output = ai_output.summary_text
        counter = 0

        consistent = omitted = status_errors = dose_errors = 0
        mentioned = 0

        for medication in medications:
            support = analyze_medication(medication, output, self.matcher, structured)
            importance = (
                FactImportance.CRITICAL if medication.is_change else FactImportance.NON_CRITICAL
            )
            shared = {
                "source_ref": medication.entity_id,
                "source_evidence": support.source_evidence,
                "output_evidence": support.output_evidence,
                "match_level": support.match_level,
                "confidence": support.confidence,
            }

            if support.status is SupportStatus.OMITTED:
                omitted += 1
                result.findings.append(
                    self.finding(
                        case, FindingType.MEDICATION_OMISSION,
                        f"{medication.drug_name} ({medication.status.value}) is not mentioned "
                        f"in the summary.",
                        index=counter,
                        details={
                            "drug": medication.drug_name,
                            "status": medication.status.value,
                            "fact_importance": importance.value,
                        },
                        **shared,
                    )
                )
                counter += 1
                continue

            mentioned += 1
            error = support.details.get("error")

            if support.status is SupportStatus.SUPPORTED:
                consistent += 1
                continue

            if error == "dose":
                dose_errors += 1
                finding_type = FindingType.MEDICATION_DOSE_ERROR
            else:
                status_errors += 1
                finding_type = FindingType.MEDICATION_STATUS_ERROR

            result.findings.append(
                self.finding(
                    case, finding_type, support.reason.capitalize() + ".",
                    index=counter,
                    details={
                        "drug": medication.drug_name,
                        "expected_status": medication.status.value,
                        "reported_status": support.details.get("reported"),
                        "fact_importance": importance.value,
                    },
                    **shared,
                )
            )
            counter += 1

        hallucinated = self._hallucinated_drugs(structured, output)
        for drug in hallucinated:
            result.findings.append(
                self.finding(
                    case, FindingType.MEDICATION_HALLUCINATION,
                    f"The summary names {drug}, which is not among this case's medications.",
                    index=counter,
                    output_evidence=self._quote(drug, output),
                    details={"drug": drug, "fact_importance": FactImportance.CRITICAL.value},
                )
            )
            counter += 1

        total = len(medications)
        result.metrics = {
            "medication_consistency": metrics.compute("medication_consistency", consistent, total),
            "medication_omission_rate": metrics.compute("medication_omission_rate", omitted, total),
            "medication_status_error_rate": metrics.compute(
                "medication_status_error_rate", status_errors, mentioned
            ),
        }
        result.details = {
            "medications": total,
            "consistent": consistent,
            "omitted": omitted,
            "status_errors": status_errors,
            "dose_errors": dose_errors,
            "hallucinated": len(hallucinated),
        }
        return result

    def _hallucinated_drugs(self, structured, output: str) -> list[str]:
        """Drugs named in the output that this case does not contain.

        Bounded by the content library's vocabulary, so this detects invented
        drugs the benchmark could plausibly have produced. A drug outside that
        vocabulary is not detected - a documented limit, not a silent one.
        """
        if structured is None:
            return []
        case_drugs = {normalize_text(m.drug_name) for m in structured.medications}
        normalized_output = normalize_text(output)
        matcher = FactMatcher()

        hallucinated = []
        for drug in sorted(content.all_drug_names()):
            normalized_drug = normalize_text(drug)
            if normalized_drug in case_drugs:
                continue
            if matcher._contains_form(normalized_output, normalized_drug):
                hallucinated.append(drug)
        return hallucinated

    def _quote(self, term: str, output: str) -> str | None:
        from app.evaluation.negation import containing_sentence

        return containing_sentence(term, output)
