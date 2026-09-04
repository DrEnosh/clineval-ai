"""LLM adjudication of ambiguous deterministic findings.

Takes findings the router selected, asks a model one narrow question about
each, parses the answer strictly, and reconciles it against the deterministic
result under the documented rules.

The question asked is deliberately narrow — "is this one source fact conveyed
by this summary?" — rather than "evaluate this summary". A narrow question has
a checkable answer; a broad one invites the model to produce an impressive
paragraph that cannot be verified. It also means the model is never asked to
decide what is clinically critical: criticality comes from the benchmark's own
rules (CR-01..CR-10), and letting a model reassign it would put ground truth
back in the model's hands.
"""

from __future__ import annotations

import time

from app.core.enums import AdjudicationStatus, ProviderKind
from app.evaluation.llm.parser import parse_adjudication
from app.evaluation.llm.reconciler import reconcile
from app.evaluation.llm.router import AdjudicationPolicy, AdjudicationRouter
from app.evaluation.llm.schemas import (
    AdjudicationRecord,
    AdjudicationSummary,
    LLMCallRecord,
)
from app.models.schemas import EvaluationFinding
from app.prompts.library import Prompt, load_prompt
from app.providers.base import ModelProvider, ModelRequest, ProviderError

DEFAULT_MAX_ATTEMPTS = 2


class LLMAdjudicator:
    """Asks a model to adjudicate individual deterministic findings."""

    name = "llm_adjudicator"
    version = "0.4.0"

    def __init__(
        self,
        provider: ModelProvider,
        model: str,
        policy: AdjudicationPolicy | None = None,
        prompt_name: str = "adjudication",
        prompt_version: str = "v1",
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        capture_raw: bool = False,
        prompt: Prompt | None = None,
    ):
        self.provider = provider
        self.model = model
        self.policy = policy or AdjudicationPolicy()
        self.router = AdjudicationRouter(self.policy)
        self.prompt = prompt or load_prompt(prompt_name, prompt_version)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max(1, max_attempts)
        self.capture_raw = capture_raw
        self.summary = AdjudicationSummary()

    # -- single finding -----------------------------------------------------

    def adjudicate(self, finding: EvaluationFinding, summary_text: str) -> AdjudicationRecord:
        """Route, call, parse, and reconcile one finding."""
        record = AdjudicationRecord(
            finding_id=finding.finding_id,
            case_id=finding.case_id,
            source_ref=finding.source_ref,
            fact_id=finding.fact_id,
            deterministic_finding_type=finding.finding_type.value,
            deterministic_confidence=finding.confidence,
            deterministic_match_level=(
                finding.match_level.value if finding.match_level else None
            ),
            source_evidence=finding.source_evidence,
            output_evidence=finding.output_evidence,
        )

        decision = self.router.decide(finding)
        record.routing_reason = decision.reason
        if not decision.route:
            record.status = AdjudicationStatus.NOT_REQUIRED
            self.summary.record(record)
            return record

        call, response = self._call(finding, summary_text)
        record.call = call
        reconcile(record, response, self.policy)
        self.summary.record(record)
        return record

    def _call(self, finding: EvaluationFinding, summary_text: str):
        """Call the judge with bounded retry on transport and parse failures.

        A parse failure is retried because it is often transient formatting;
        it is retried a *bounded* number of times because a model that cannot
        produce the schema twice will not produce it on the tenth attempt, and
        the run should not stall discovering that.
        """
        system, user = self.prompt.render(
            fact_type=finding.finding_type.value,
            source_fact=finding.source_evidence or "(no source evidence recorded)",
            summary=summary_text,
            deterministic_note=(
                f"{finding.explanation} "
                f"(confidence {finding.confidence:.2f}, "
                f"match {finding.match_level.value if finding.match_level else 'n/a'})"
            ),
        )
        request = ModelRequest(
            system=system,
            user=user,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            metadata={"case_id": finding.case_id, "finding_id": finding.finding_id},
        )

        call = LLMCallRecord(
            provider=self.provider.kind,
            model=self.model,
            prompt_name=self.prompt.name,
            prompt_version=self.prompt.version,
            prompt_hash=self.prompt.content_hash,
            temperature=self.temperature,
        )

        last_error = "no attempt made"
        started = time.perf_counter()

        for attempt in range(1, self.max_attempts + 1):
            call.attempts = attempt
            try:
                model_response = self.provider.generate(request)
            except ProviderError as error:
                last_error = f"{type(error).__name__}: {error}"
                continue

            call.model_version = model_response.model_version
            call.latency_ms = model_response.latency_ms
            call.input_tokens = model_response.usage.input_tokens
            call.output_tokens = model_response.usage.output_tokens
            call.total_tokens = model_response.usage.merged_total()
            call.estimated_cost_usd = model_response.usage.estimated_cost_usd
            if self.capture_raw:
                call.raw_response = model_response.text

            parsed = parse_adjudication(model_response.text)
            if parsed.succeeded:
                return call, parsed.response

            call.parse_failures += 1
            last_error = f"unparseable judge response: {parsed.error}"

        call.error = last_error
        if call.latency_ms is None:
            call.latency_ms = round((time.perf_counter() - started) * 1000, 3)
        return call, None

    # -- batch --------------------------------------------------------------

    def adjudicate_findings(
        self, findings: list[EvaluationFinding], summaries: dict[str, str]
    ) -> list[AdjudicationRecord]:
        """Adjudicate every finding for which the summary text is available.

        A finding whose case has no summary is skipped explicitly rather than
        adjudicated against an empty string, which would invite the judge to
        confirm an omission that was never actually generated.
        """
        records: list[AdjudicationRecord] = []
        for finding in findings:
            summary_text = summaries.get(finding.case_id)
            if summary_text is None:
                record = AdjudicationRecord(
                    finding_id=finding.finding_id,
                    case_id=finding.case_id,
                    deterministic_finding_type=finding.finding_type.value,
                    deterministic_confidence=finding.confidence,
                    status=AdjudicationStatus.SKIPPED,
                    routing_reason="no generated summary available for this case",
                )
                records.append(record)
                self.summary.record(record)
                continue
            records.append(self.adjudicate(finding, summary_text))
        return records

    def metadata(self) -> dict:
        """Non-secret description of this adjudicator, for run metadata."""
        return {
            "adjudicator": self.name,
            "adjudicator_version": self.version,
            **self.provider.describe(),
            "model": self.model,
            **self.prompt.metadata(),
            "temperature": self.temperature,
            "max_attempts": self.max_attempts,
            "is_model_based_judgment": True,
            "policy": self.policy.model_dump(mode="json"),
        }

    @property
    def is_mock(self) -> bool:
        return self.provider.kind is ProviderKind.MOCK
