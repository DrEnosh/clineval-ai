"""Model output generation schemas.

The request/response pair for turning a benchmark case into a summary. Kept
separate from `ModelRequest`/`ModelResponse` (which are provider-level) because
this layer knows about clinical cases and prompts, and the provider layer must
not.

No secret ever reaches these shapes. `ClinicalSummaryResponse` is written
verbatim into run files.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.core.enums import ProviderKind
from app.providers.base import TokenUsage


class ClinicalSummaryRequest(BaseModel):
    """A request to summarize one clinical note."""

    case_id: str
    clinical_note: str
    model: str
    provider: str
    prompt_name: str = "summarization"
    prompt_version: str = "v1"
    temperature: float = 0.0
    max_tokens: int = 1024
    #: Distinguishes repeated generations of the same case, for repeatability
    #: testing in a later phase.
    run_index: int = 0

    def configuration(self) -> dict[str, Any]:
        """The settings that determine the output, for cache keys and metadata."""
        return {
            "model": self.model,
            "provider": self.provider,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "run_index": self.run_index,
        }


class ClinicalSummaryResponse(BaseModel):
    """A generated summary plus everything needed to reproduce and cost it."""

    case_id: str
    summary_text: str
    provider: ProviderKind
    model: str
    model_version: str | None = None
    prompt_name: str
    prompt_version: str
    prompt_hash: str
    temperature: float
    max_tokens: int
    run_index: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str | None = None
    from_cache: bool = False
    #: True only for genuine remote model output. Mock and cached-mock outputs
    #: are False, so a synthetic generation can never be read as a real one.
    is_real_model_output: bool = False
    #: Injection-like content detected in this case's note, recorded for
    #: visibility. Never used to alter or block the content.
    injection_scan: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_ai_output(self):
        """Convert to the `AIOutput` the Phase 3 evaluators consume."""
        from app.models.schemas import AIOutput

        return AIOutput(
            case_id=self.case_id,
            model_id=f"{self.provider.value}:{self.model}",
            summary_text=self.summary_text,
            run_index=self.run_index,
            generated_at=self.generated_at,
            raw_metadata={
                "provider": self.provider.value,
                "model": self.model,
                "model_version": self.model_version,
                "prompt_version": self.prompt_version,
                "prompt_hash": self.prompt_hash,
                "is_real_model_output": self.is_real_model_output,
                "from_cache": self.from_cache,
                "latency_ms": self.latency_ms,
            },
        )


class GenerationSummary(BaseModel):
    """Aggregate cost, token, and latency reporting for a generation batch.

    Counts of *known* values are tracked separately from totals, because
    "$0.00 across 10 calls" and "$0.00 across 0 priced calls" mean entirely
    different things and must not be reported the same way.
    """

    requested: int = 0
    generated: int = 0
    from_cache: int = 0
    failed: int = 0
    calls_made: int = 0

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tokens_known_for: int = 0

    estimated_cost_usd: float = 0.0
    cost_known_for: int = 0

    total_latency_ms: float = 0.0
    latency_known_for: int = 0

    @property
    def average_latency_ms(self) -> float | None:
        if not self.latency_known_for:
            return None
        return round(self.total_latency_ms / self.latency_known_for, 2)

    @property
    def cost_is_complete(self) -> bool:
        """Whether every call that was made had a known cost."""
        return self.calls_made > 0 and self.cost_known_for == self.calls_made

    def record(self, response: ClinicalSummaryResponse) -> None:
        self.requested += 1
        if response.error is not None:
            self.failed += 1
            return
        if response.from_cache:
            self.from_cache += 1
        else:
            self.generated += 1
            self.calls_made += 1

        usage = response.usage
        total = usage.merged_total()
        if total is not None:
            self.tokens_known_for += 1
            self.total_tokens += total
            self.input_tokens += usage.input_tokens or 0
            self.output_tokens += usage.output_tokens or 0
        if usage.estimated_cost_usd is not None:
            self.cost_known_for += 1
            self.estimated_cost_usd = round(
                self.estimated_cost_usd + usage.estimated_cost_usd, 6
            )
        if response.latency_ms is not None and not response.from_cache:
            self.latency_known_for += 1
            self.total_latency_ms += response.latency_ms

    def report(self) -> dict[str, Any]:
        """Human-readable aggregate. Unknown values are stated as unknown."""
        return {
            "cases_requested": self.requested,
            "generated": self.generated,
            "served_from_cache": self.from_cache,
            "failed": self.failed,
            "model_calls": self.calls_made,
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "total": self.total_tokens,
                "known_for_calls": self.tokens_known_for,
            },
            "estimated_cost_usd": (
                self.estimated_cost_usd if self.cost_known_for else None
            ),
            "cost_known_for_calls": self.cost_known_for,
            "cost_is_complete": self.cost_is_complete,
            "average_latency_ms": self.average_latency_ms,
        }
