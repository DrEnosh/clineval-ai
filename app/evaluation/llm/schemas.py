"""Structured LLM judgment schemas.

Every LLM response is parsed into a validated Pydantic model. Free-text
judgments are never interpreted: a judge that cannot produce the schema has
failed, and a failure is recorded as such rather than guessed at.

`AdjudicationRecord` is the audit trail. It holds the deterministic finding,
the model judgment, the disagreement type, and the final disposition — so a
reader can see not just what was decided but that a decision was made, by
what, and over what disagreement.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.enums import (
    AdjudicationJudgment,
    AdjudicationStatus,
    DisagreementType,
    ProviderKind,
)


class EvidenceSpan(BaseModel):
    """A quoted pair the judge used to justify its verdict."""

    source_span: str = ""
    output_span: str = ""


class AdjudicationResponse(BaseModel):
    """The judgment payload an LLM must return.

    Strict: unknown fields are rejected rather than ignored, because a judge
    inventing extra keys is a signal that it has not followed the contract and
    its other fields deserve suspicion too.
    """

    model_config = {"extra": "forbid"}

    judgment: AdjudicationJudgment
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    evidence: list[EvidenceSpan] = Field(default_factory=list)

    @field_validator("reason")
    @classmethod
    def _trim(cls, value: str) -> str:
        return value.strip()


class LLMCallRecord(BaseModel):
    """Everything about one judge call, for reproducibility and cost tracking.

    Carries no credential. `raw_response` is populated only when the caller
    opts in, since raw payloads echo clinical content.
    """

    provider: ProviderKind
    model: str
    model_version: str | None = None
    prompt_name: str
    prompt_version: str
    prompt_hash: str
    temperature: float
    called_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    attempts: int = 1
    parse_failures: int = 0
    raw_response: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class AdjudicationRecord(BaseModel):
    """One adjudication: deterministic view, model view, and disposition.

    Both views are always retained. When they disagree in a way no documented
    rule resolves, `status` is ADJUDICATION_REQUIRED and neither is discarded —
    the uncertainty is the result.
    """

    finding_id: str
    case_id: str
    source_ref: str | None = None
    fact_id: str | None = None

    # What the deterministic evaluator concluded.
    deterministic_finding_type: str
    deterministic_confidence: float
    deterministic_match_level: str | None = None
    source_evidence: str | None = None
    output_evidence: str | None = None

    # What the model concluded. None when no call was made or it failed.
    judgment: AdjudicationJudgment | None = None
    #: The model's own stated confidence. NOT a calibrated probability - it is
    #: a number the model produced about itself, and this project has not
    #: measured whether it corresponds to accuracy. Named to prevent it being
    #: read as one. See docs/LLM_EVALUATION.md.
    model_reported_confidence: float | None = None
    model_reason: str | None = None
    model_evidence: list[EvidenceSpan] = Field(default_factory=list)

    disagreement: DisagreementType = DisagreementType.NONE
    status: AdjudicationStatus = AdjudicationStatus.NOT_REQUIRED
    #: Why the router routed (or did not route) this finding.
    routing_reason: str = ""
    #: Which documented rule produced the disposition, if any.
    resolution_rule: str | None = None
    #: True when the finding should be withdrawn from the results.
    finding_withdrawn: bool = False

    call: LLMCallRecord | None = None

    def audit(self) -> dict[str, Any]:
        """A flat, human-readable view for reports and the CLI."""
        return {
            "finding_id": self.finding_id,
            "case_id": self.case_id,
            "deterministic": self.deterministic_finding_type,
            "llm_judgment": self.judgment.value if self.judgment else None,
            "model_reported_confidence": self.model_reported_confidence,
            "disagreement": self.disagreement.value,
            "status": self.status.value,
            "resolution_rule": self.resolution_rule,
            "finding_withdrawn": self.finding_withdrawn,
        }


class AdjudicationSummary(BaseModel):
    """Aggregate adjudication and cost reporting for a run."""

    findings_considered: int = 0
    routed_to_llm: int = 0
    skipped_high_confidence: int = 0
    agreed: int = 0
    resolved_by_llm: int = 0
    adjudication_required: int = 0
    failed: int = 0

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tokens_known_for: int = 0
    estimated_cost_usd: float = 0.0
    cost_known_for: int = 0
    total_latency_ms: float = 0.0
    latency_known_for: int = 0
    parse_failures: int = 0

    @property
    def adjudication_rate(self) -> float | None:
        if not self.findings_considered:
            return None
        return round(self.routed_to_llm / self.findings_considered, 4)

    @property
    def disagreement_rate(self) -> float | None:
        """Share of adjudicated findings where the two methods disagreed."""
        adjudicated = self.agreed + self.resolved_by_llm + self.adjudication_required
        if not adjudicated:
            return None
        return round((self.resolved_by_llm + self.adjudication_required) / adjudicated, 4)

    @property
    def average_latency_ms(self) -> float | None:
        if not self.latency_known_for:
            return None
        return round(self.total_latency_ms / self.latency_known_for, 2)

    def record(self, record: AdjudicationRecord) -> None:
        self.findings_considered += 1

        if record.status is AdjudicationStatus.NOT_REQUIRED:
            self.skipped_high_confidence += 1
        elif record.status is AdjudicationStatus.AGREED:
            self.agreed += 1
        elif record.status is AdjudicationStatus.RESOLVED_BY_LLM:
            self.resolved_by_llm += 1
        elif record.status is AdjudicationStatus.ADJUDICATION_REQUIRED:
            self.adjudication_required += 1
        elif record.status is AdjudicationStatus.FAILED:
            self.failed += 1

        call = record.call
        if call is None:
            return

        self.routed_to_llm += 1
        self.llm_calls += 1
        self.parse_failures += call.parse_failures

        if call.total_tokens is not None:
            self.tokens_known_for += 1
            self.total_tokens += call.total_tokens
            self.input_tokens += call.input_tokens or 0
            self.output_tokens += call.output_tokens or 0
        if call.estimated_cost_usd is not None:
            self.cost_known_for += 1
            self.estimated_cost_usd = round(
                self.estimated_cost_usd + call.estimated_cost_usd, 6
            )
        if call.latency_ms is not None:
            self.latency_known_for += 1
            self.total_latency_ms += call.latency_ms

    def report(self) -> dict[str, Any]:
        return {
            "findings_considered": self.findings_considered,
            "routed_to_llm": self.routed_to_llm,
            "skipped_high_confidence": self.skipped_high_confidence,
            "adjudication_rate": self.adjudication_rate,
            "agreed": self.agreed,
            "resolved_by_llm": self.resolved_by_llm,
            "adjudication_required": self.adjudication_required,
            "failed": self.failed,
            "disagreement_rate": self.disagreement_rate,
            "llm_calls": self.llm_calls,
            "parse_failures": self.parse_failures,
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "total": self.total_tokens,
                "known_for_calls": self.tokens_known_for,
            },
            "estimated_cost_usd": self.estimated_cost_usd if self.cost_known_for else None,
            "cost_known_for_calls": self.cost_known_for,
            "average_latency_ms": self.average_latency_ms,
        }
