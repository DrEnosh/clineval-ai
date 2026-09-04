"""Reconciling deterministic findings with LLM judgments.

The rule that governs this module: **an LLM judgment never silently overrides
a deterministic finding.** When the two disagree, the disagreement is recorded
explicitly, and only a small set of documented rules produces a resolution.
Everything else stays `ADJUDICATION_REQUIRED` — genuinely unresolved, and
visible as such.

The resolution rules, in full:

| Rule | Condition | Disposition |
|---|---|---|
| **R1** | LLM agrees the finding stands (UNSUPPORTED / CONTRADICTED) | `AGREED` — finding kept |
| **R2** | Deterministic *omission* + LLM SUPPORTED at high confidence | `RESOLVED_BY_LLM` — finding withdrawn |
| **R3** | Deterministic *contradiction* + LLM SUPPORTED | `ADJUDICATION_REQUIRED` — finding kept |
| **R4** | LLM UNCERTAIN | `ADJUDICATION_REQUIRED` — finding kept |
| **R5** | LLM SUPPORTED below the confidence floor | `ADJUDICATION_REQUIRED` — finding kept |
| **R6** | The call or parse failed | `FAILED` — finding kept |

Why R2 and R3 differ. R2 is the case the LLM is genuinely useful for: the
deterministic matcher could not find a fact that was there, expressed
differently. Withdrawing that finding removes a false positive. R3 is the
opposite risk: the deterministic evaluator found the summary asserting
something incompatible with the source, and a model saying "looks fine to me"
is not sufficient grounds to erase a detected contradiction. Contradictions are
the findings with the highest clinical cost if wrongly dismissed, so they are
never dismissed by a model — only escalated to human review.

Every rule keeps the finding except R2. The system's bias is toward reporting
too much rather than silently dropping a clinical error.
"""

from __future__ import annotations

from app.core.enums import (
    AdjudicationJudgment,
    AdjudicationStatus,
    DisagreementType,
    FindingType,
)
from app.evaluation.llm.router import AdjudicationPolicy
from app.evaluation.llm.schemas import AdjudicationRecord, AdjudicationResponse

#: Finding types asserting the output states something incompatible with the
#: source. Never withdrawn on a model's say-so.
CONTRADICTION_TYPES: frozenset[FindingType] = frozenset(
    {
        FindingType.CONTRADICTION,
        FindingType.NEGATION_REVERSAL,
        FindingType.TEMPORAL_TREND_REVERSAL,
        FindingType.TEMPORAL_VALUE_CONFUSION,
        FindingType.MEDICATION_STATUS_ERROR,
        FindingType.MEDICATION_DOSE_ERROR,
        FindingType.HISTORICAL_CURRENT_CONFUSION,
        FindingType.CONFLICT_RESOLUTION_ERROR,
        FindingType.VALUE_ERROR,
    }
)

#: Finding types asserting content is absent. These are withdrawable under R2:
#: "absent" is exactly the claim a paraphrase can falsify.
OMISSION_TYPES: frozenset[FindingType] = frozenset(
    {
        FindingType.CRITICAL_OMISSION,
        FindingType.NON_CRITICAL_OMISSION,
        FindingType.MEDICATION_OMISSION,
        FindingType.NEGATION_OMISSION,
        FindingType.TEMPORAL_INFORMATION_LOSS,
    }
)

#: Finding types asserting the output added something unsupported. A judge
#: saying SUPPORTED here means it found source backing the matcher missed.
ADDITION_TYPES: frozenset[FindingType] = frozenset({FindingType.UNSUPPORTED_ADDITION})


def classify_disagreement(
    finding_type: FindingType, judgment: AdjudicationJudgment
) -> DisagreementType:
    """How a judgment differs from the deterministic finding it was asked about.

    The deterministic finding always asserts a problem, so agreement means the
    judge also found a problem.
    """
    if judgment is AdjudicationJudgment.UNCERTAIN:
        return DisagreementType.LLM_UNCERTAIN

    if judgment is AdjudicationJudgment.SUPPORTED:
        # The judge says the summary is fine; the evaluator said it is not.
        return DisagreementType.LLM_MORE_LENIENT

    if finding_type in OMISSION_TYPES and judgment is AdjudicationJudgment.CONTRADICTED:
        # Both see a problem, but the judge sees a worse one.
        return DisagreementType.LLM_MORE_STRICT

    if finding_type in ADDITION_TYPES and judgment is AdjudicationJudgment.UNSUPPORTED:
        # The judge confirms the content is unsupported.
        return DisagreementType.NONE

    return DisagreementType.NONE


def reconcile(
    record: AdjudicationRecord,
    response: AdjudicationResponse | None,
    policy: AdjudicationPolicy,
) -> AdjudicationRecord:
    """Apply the documented rules to one adjudication.

    Mutates and returns `record`. Both views are always retained; only the
    disposition is decided here.
    """
    finding_type = FindingType(record.deterministic_finding_type)

    if response is None:
        record.status = AdjudicationStatus.FAILED
        record.resolution_rule = "R6"
        record.finding_withdrawn = False
        return record

    record.judgment = response.judgment
    record.model_reported_confidence = response.confidence
    record.model_reason = response.reason
    record.model_evidence = list(response.evidence)
    record.disagreement = classify_disagreement(finding_type, response.judgment)

    # R4 - the judge declined to decide.
    if response.judgment is AdjudicationJudgment.UNCERTAIN:
        record.status = AdjudicationStatus.ADJUDICATION_REQUIRED
        record.resolution_rule = "R4"
        record.finding_withdrawn = False
        return record

    # R1 - the judge agrees a problem exists.
    if response.judgment in (
        AdjudicationJudgment.UNSUPPORTED,
        AdjudicationJudgment.CONTRADICTED,
    ):
        record.status = AdjudicationStatus.AGREED
        record.resolution_rule = "R1"
        record.finding_withdrawn = False
        return record

    # From here the judgment is SUPPORTED: the judge disagrees with the finding.

    # R5 - not confident enough to overturn anything.
    if response.confidence < policy.min_judgment_confidence:
        record.status = AdjudicationStatus.ADJUDICATION_REQUIRED
        record.resolution_rule = "R5"
        record.finding_withdrawn = False
        return record

    # R3 - never let a model erase a detected contradiction.
    if finding_type in CONTRADICTION_TYPES:
        record.status = AdjudicationStatus.ADJUDICATION_REQUIRED
        record.resolution_rule = "R3"
        record.finding_withdrawn = False
        return record

    # R2 - the paraphrase case: withdraw a false omission.
    if finding_type in OMISSION_TYPES or finding_type in ADDITION_TYPES:
        record.status = AdjudicationStatus.RESOLVED_BY_LLM
        record.resolution_rule = "R2"
        record.finding_withdrawn = True
        return record

    # Anything not covered by a documented rule stays unresolved.
    record.status = AdjudicationStatus.ADJUDICATION_REQUIRED
    record.disagreement = DisagreementType.INCOMPATIBLE
    record.resolution_rule = None
    record.finding_withdrawn = False
    return record


def apply_records(findings, records: list[AdjudicationRecord]):
    """Split findings into those that stand and those withdrawn by rule R2.

    Returns (kept, withdrawn). Withdrawn findings are returned rather than
    discarded so a report can show what adjudication removed and why - an
    evaluation that silently drops findings is not auditable.
    """
    withdrawn_ids = {
        record.finding_id for record in records if record.finding_withdrawn
    }
    kept = [finding for finding in findings if finding.finding_id not in withdrawn_ids]
    withdrawn = [finding for finding in findings if finding.finding_id in withdrawn_ids]
    return kept, withdrawn
