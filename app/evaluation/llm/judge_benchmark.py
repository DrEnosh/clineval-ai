"""A labelled benchmark for the LLM judge.

Builds adjudication questions whose correct answers are derived from the
benchmark's structured ground truth, not from any model. Each case pairs one
source fact with a summary that is either faithful, corrupted in a known way,
or reworded — and states what the judge should say.

    Scenario              Summary                          Expected
    -------------------   ------------------------------   ------------
    clearly_correct       reference summary                SUPPORTED
    valid_paraphrase      reworded, meaning preserved      SUPPORTED
    clearly_omitted       fact's sentence removed          UNSUPPORTED
    clearly_hallucinated  invented diagnosis appended      UNSUPPORTED
    explicit_contradiction  lab value altered              CONTRADICTED
    negation_reversal     negative flipped to positive     CONTRADICTED
    temporal_reversal     trend direction reversed         CONTRADICTED
    ambiguous             vague gesture at the fact        UNCERTAIN

`ambiguous` deliberately admits more than one defensible answer. It is
included so the harness can measure whether a judge uses UNCERTAIN at all — a
judge that never says "I cannot tell" is guessing somewhere, and this is where
that shows up. It is reported separately from the decidable scenarios for
exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from app.core.enums import AdjudicationJudgment
from app.evaluation.injection import inject


@dataclass(frozen=True)
class JudgeCase:
    """One labelled adjudication question."""

    case_id: str
    scenario: str
    fact_type: str
    source_fact: str
    summary: str
    expected: AdjudicationJudgment
    #: True where more than one answer is defensible, so accuracy on this case
    #: says less than accuracy on a decidable one.
    is_decidable: bool = True
    notes: str = ""


@dataclass
class JudgeBenchmark:
    cases: list[JudgeCase] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    @property
    def decidable(self) -> list[JudgeCase]:
        return [case for case in self.cases if case.is_decidable]

    def by_scenario(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            counts[case.scenario] = counts.get(case.scenario, 0) + 1
        return dict(sorted(counts.items()))


def _first_critical_lab_fact(case) -> tuple[str, str] | None:
    """A critical laboratory fact and its content, if the case has one."""
    structured = case.structured_case
    if structured is None:
        return None
    critical_refs = {fact.source_ref: fact for fact in case.critical_facts}
    for lab in structured.labs:
        fact = critical_refs.get(lab.entity_id)
        if fact is not None and lab.format_value() in case.reference_summary:
            return "lab", fact.content
    return None


def _first_negation(case) -> tuple[str, str] | None:
    for fact in case.negations:
        if fact.content.lower().startswith("no "):
            return "negation", fact.content
    return None


def _first_trend(case) -> tuple[str, str] | None:
    for fact in case.temporal_facts:
        if fact.importance.value == "critical":
            return "temporal", fact.content
    return None


def _paraphrase(summary: str) -> str:
    from app.providers.mock import _paraphrase as reword

    return reword(summary)


def build_judge_benchmark(cases, seed: int = 0) -> JudgeBenchmark:
    """Construct labelled judge cases from benchmark ground truth."""
    rng = Random(f"judge:{seed}")
    built: list[JudgeCase] = []

    for case in cases:
        lab_fact = _first_critical_lab_fact(case)
        negation_fact = _first_negation(case)
        trend_fact = _first_trend(case)

        # SUPPORTED: the reference summary conveys its own facts.
        if lab_fact:
            fact_type, content = lab_fact
            built.append(
                JudgeCase(
                    case.case_id, "clearly_correct", fact_type, content,
                    case.reference_summary, AdjudicationJudgment.SUPPORTED,
                    notes="reference summary states this fact directly",
                )
            )
            built.append(
                JudgeCase(
                    case.case_id, "valid_paraphrase", fact_type, content,
                    _paraphrase(case.reference_summary), AdjudicationJudgment.SUPPORTED,
                    notes="reworded without changing clinical meaning",
                )
            )

            # UNSUPPORTED: the sentence carrying the fact is removed.
            omission = inject("remove_critical_fact", case, seed=seed)
            if omission is not None:
                built.append(
                    JudgeCase(
                        case.case_id, "clearly_omitted", fact_type, content,
                        omission.corrupted_summary, AdjudicationJudgment.UNSUPPORTED,
                        notes=omission.description,
                    )
                )

            # CONTRADICTED: the value is altered.
            altered = inject("change_lab_value", case, seed=seed)
            if altered is not None and altered.original_fragment:
                built.append(
                    JudgeCase(
                        case.case_id, "explicit_contradiction", fact_type, content,
                        altered.corrupted_summary, AdjudicationJudgment.CONTRADICTED,
                        notes=altered.description,
                    )
                )

            # UNCERTAIN: a vague gesture that neither states nor denies it.
            built.append(
                JudgeCase(
                    case.case_id, "ambiguous", fact_type, content,
                    "Laboratory results were reviewed and discussed with the patient.",
                    AdjudicationJudgment.UNCERTAIN,
                    is_decidable=False,
                    notes="vague reference; more than one answer is defensible",
                )
            )

        # CONTRADICTED: a negative asserted as positive.
        if negation_fact:
            reversal = inject("reverse_negation", case, seed=seed)
            if reversal is not None:
                fact_type, content = negation_fact
                built.append(
                    JudgeCase(
                        case.case_id, "negation_reversal", fact_type, content,
                        reversal.corrupted_summary, AdjudicationJudgment.CONTRADICTED,
                        notes=reversal.description,
                    )
                )

        # CONTRADICTED: a trend stated backwards.
        if trend_fact:
            reversal = inject("reverse_temporal_trend", case, seed=seed)
            if reversal is not None:
                fact_type, content = trend_fact
                built.append(
                    JudgeCase(
                        case.case_id, "temporal_reversal", fact_type, content,
                        reversal.corrupted_summary, AdjudicationJudgment.CONTRADICTED,
                        notes=reversal.description,
                    )
                )

        # UNSUPPORTED: a fact the summary never contained, asked about a
        # summary that invented something else entirely.
        hallucination = inject("add_unsupported_diagnosis", case, seed=seed)
        if hallucination is not None and hallucination.replacement_fragment:
            built.append(
                JudgeCase(
                    case.case_id, "clearly_hallucinated", "condition",
                    f"The case does not record {hallucination.replacement_fragment}.",
                    hallucination.corrupted_summary, AdjudicationJudgment.UNSUPPORTED,
                    notes=hallucination.description,
                )
            )

    rng.shuffle(built)
    return JudgeBenchmark(cases=built)


def evaluate_judge(
    benchmark: JudgeBenchmark, adjudicator, decidable_only: bool = True
):
    """Run a judge over the labelled benchmark and score it.

    `decidable_only` excludes the deliberately ambiguous cases from scoring,
    since an accuracy figure that includes questions with no single right
    answer is not a measure of correctness.
    """
    from app.evaluation.llm.judge_metrics import JudgeEvaluation
    from app.evaluation.llm.parser import parse_adjudication
    from app.providers.base import ModelRequest, ProviderError

    evaluation = JudgeEvaluation()
    cases = benchmark.decidable if decidable_only else benchmark.cases

    for case in cases:
        system, user = adjudicator.prompt.render(
            fact_type=case.fact_type,
            source_fact=case.source_fact,
            summary=case.summary,
            deterministic_note="(judge benchmark: no deterministic result supplied)",
        )
        request = ModelRequest(
            system=system, user=user, model=adjudicator.model,
            temperature=adjudicator.temperature, max_tokens=adjudicator.max_tokens,
            metadata={"case_id": case.case_id, "scenario": case.scenario},
        )
        try:
            response = adjudicator.provider.generate(request)
        except ProviderError:
            evaluation.record(case.expected, None)
            continue

        parsed = parse_adjudication(response.text)
        evaluation.record(
            case.expected, parsed.response.judgment if parsed.succeeded else None
        )

    return evaluation
