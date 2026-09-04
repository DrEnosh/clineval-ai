"""Structured fact support.

Two directions, both needed:

**Source -> output.** Every analyzable entity in the case is classified as
supported, partially supported, omitted, or contradicted. This is the broad
coverage measure across conditions, medications, labs, vitals, procedures,
findings, allergies, and trends.

**Output -> source.** Clinical terms and values the output asserts are checked
back against the structured case. Anything absent is an unsupported addition -
the hallucination direction.

The second direction is vocabulary-bounded: it detects invented content drawn
from the benchmark's own content library, which is what the benchmark can
plausibly produce. Content invented in terms outside that vocabulary is not
detected. That makes `unsupported_fact_rate` a floor on hallucination rather
than a ceiling - stated here and in the metric definition, because a
hallucination metric that quietly under-reports is worse than none.
"""

from __future__ import annotations

import re

from app.benchmark import content
from app.core.enums import FindingType, SupportStatus
from app.evaluation import metrics
from app.evaluation.base import DeterministicEvaluator, EvaluatorResult
from app.evaluation.negation import containing_sentence, is_negated
from app.evaluation.normalization import normalize_text
from app.evaluation.support import analyze_entity
from app.models.schemas import AIOutput, BenchmarkCase

_NUMBER = re.compile(r"\d+(?:\.\d+)?")

# Numbers that appear in ordinary prose rather than as clinical values.
_INNOCUOUS_NUMBERS = frozenset({"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "24"})


class FactSupportEvaluator(DeterministicEvaluator):
    """Generic support analysis across every structured entity type."""

    name = "fact_support"
    version = "0.5.0"

    def evaluate(self, case: BenchmarkCase, ai_output: AIOutput) -> EvaluatorResult:
        structured = case.structured_case
        entities = structured.all_entities() if structured else []

        empty = self.guard_output(case, ai_output)
        if empty is not None:
            analyzable = sum(
                1 for entity in entities if analyze_entity(entity, "x", self.matcher, structured) is not None
            )
            empty.metrics = {
                "supported_fact_rate": metrics.compute("supported_fact_rate", 0, analyzable),
                # An empty output asserts nothing, so there is nothing to be
                # unsupported. Undefined, not zero.
                "unsupported_fact_rate": metrics.compute("unsupported_fact_rate", 0, 0),
            }
            return empty

        result = self._empty_result(case)
        output = ai_output.summary_text
        counter = 0

        supported = analyzable = 0
        by_status: dict[str, int] = {}

        for entity in entities:
            support = analyze_entity(entity, output, self.matcher, structured)
            if support is None:
                continue
            analyzable += 1
            by_status[support.status.value] = by_status.get(support.status.value, 0) + 1

            if support.status in {SupportStatus.SUPPORTED, SupportStatus.PARTIAL}:
                supported += 1
                continue

            if support.status is SupportStatus.CONTRADICTED:
                result.findings.append(
                    self.finding(
                        case, FindingType.CONTRADICTION,
                        f"Output contradicts the source: {support.reason}.",
                        index=counter, source_ref=support.entity_id,
                        source_evidence=support.source_evidence,
                        output_evidence=support.output_evidence,
                        match_level=support.match_level, confidence=support.confidence,
                        details={"entity_kind": support.entity_kind},
                    )
                )
                counter += 1

        unsupported, asserted = self._unsupported_assertions(case, structured, output, result, counter)

        result.metrics = {
            "supported_fact_rate": metrics.compute("supported_fact_rate", supported, analyzable),
            "unsupported_fact_rate": metrics.compute("unsupported_fact_rate", unsupported, asserted),
        }
        result.details = {
            "entities_analyzed": analyzable,
            "supported": supported,
            "by_status": by_status,
            "unsupported_assertions": unsupported,
            "assertions_detected": asserted,
        }
        return result

    def _unsupported_assertions(
        self, case: BenchmarkCase, structured, output: str, result: EvaluatorResult, counter: int
    ) -> tuple[int, int]:
        """Find clinical terms and values asserted but absent from the source.

        Returns (unsupported, total_assertions). Negated mentions are skipped:
        a summary saying "no evidence of pneumonia" for a case without
        pneumonia is correct, not a hallucination.
        """
        if structured is None:
            return 0, 0

        # Normalize the corpus, not just the terms. Comparing a normalized
        # term against a raw corpus reported "Community-acquired pneumonia" as
        # unsupported in a case that contains it, because normalization turns
        # the hyphen into a space and the raw corpus keeps it.
        corpus = normalize_text(structured.text_corpus())
        normalized_output = normalize_text(output)
        unsupported = 0
        asserted = 0

        vocabularies = (
            ("condition", content.all_condition_names()),
            ("medication", content.all_drug_names()),
            ("lab", content.all_lab_names()),
        )

        for kind, vocabulary in vocabularies:
            for term in sorted(vocabulary):
                normalized_term = normalize_text(term)
                if not self.matcher._contains_form(normalized_output, normalized_term):
                    continue
                asserted += 1
                if normalized_term in corpus:
                    continue
                if is_negated(term, output).negated:
                    continue

                unsupported += 1
                result.findings.append(
                    self.finding(
                        case, FindingType.UNSUPPORTED_ADDITION,
                        f"The summary asserts {kind} '{term}', which does not appear anywhere "
                        f"in the structured case.",
                        index=counter,
                        output_evidence=containing_sentence(term, output),
                        details={"entity_kind": kind, "term": term},
                    )
                )
                counter += 1

        counter, value_unsupported, value_asserted = self._unsupported_values(
            case, corpus, output, result, counter
        )
        return unsupported + value_unsupported, asserted + value_asserted

    def _unsupported_values(
        self, case: BenchmarkCase, corpus: str, output: str, result: EvaluatorResult, counter: int
    ) -> tuple[int, int, int]:
        """Flag numeric values in the output that the source never states.

        Compares **numerically**, not by string: "12.50" is the value 12.5, and
        a model that pads a decimal has not fabricated anything. String
        containment here reported correctly-written values as hallucinations
        whenever notation differed - a false positive found by the Phase 5
        numeric-forms challenge.

        Small integers common in prose ("2 weeks", "3 months") are excluded -
        flagging them would bury real value fabrications in noise.
        """
        supported_values = {
            float(token) for token in _NUMBER.findall(corpus)
        }
        unsupported = 0
        asserted = 0
        for token in sorted(set(_NUMBER.findall(output))):
            if token in _INNOCUOUS_NUMBERS:
                continue
            asserted += 1
            if float(token) in supported_values:
                continue
            unsupported += 1
            result.findings.append(
                self.finding(
                    case, FindingType.UNSUPPORTED_ADDITION,
                    f"The summary states the value '{token}', which does not appear in the "
                    f"structured case.",
                    index=counter,
                    output_evidence=containing_sentence(token, output),
                    details={"entity_kind": "value", "value": token},
                )
            )
            counter += 1
        return counter, unsupported, asserted
