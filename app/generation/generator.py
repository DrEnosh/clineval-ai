"""Clinical summary generation.

Turns benchmark cases into model outputs, through a provider, with caching and
bounded retry.

Failure policy: a case whose generation fails is recorded as a failed response
with the error text, and the batch continues. One unreachable call must not
lose the other ninety-nine, and a failure recorded explicitly is better than a
silently missing case — the evaluation runner already skips cases with no
output rather than scoring them as model failures.
"""

from __future__ import annotations

import time

from app.core.enums import ProviderKind
from app.generation.cache import OutputCache
from app.generation.schemas import (
    ClinicalSummaryRequest,
    ClinicalSummaryResponse,
    GenerationSummary,
)
from app.prompts.library import Prompt, load_prompt
from app.prompts.sanitize import scan_for_injection
from app.providers.base import ModelProvider, ModelRequest, ProviderError

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 0.0


class SummaryGenerator:
    """Generates clinical summaries for benchmark cases."""

    def __init__(
        self,
        provider: ModelProvider,
        model: str,
        cache: OutputCache | None = None,
        prompt_name: str = "summarization",
        prompt_version: str = "v1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY_SECONDS,
        prompt: Prompt | None = None,
    ):
        self.provider = provider
        self.model = model
        self.cache = cache if cache is not None else OutputCache()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max(1, max_attempts)
        self.retry_delay = retry_delay
        self.prompt = prompt or load_prompt(prompt_name, prompt_version)
        self.summary = GenerationSummary()

    # -- single case --------------------------------------------------------

    def build_request(self, case, run_index: int = 0) -> ClinicalSummaryRequest:
        return ClinicalSummaryRequest(
            case_id=case.case_id,
            clinical_note=case.clinical_note,
            model=self.model,
            provider=self.provider.kind.value,
            prompt_name=self.prompt.name,
            prompt_version=self.prompt.version,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            run_index=run_index,
        )

    def generate(self, case, run_index: int = 0, force: bool = False) -> ClinicalSummaryResponse:
        """Generate one summary, using the cache unless `force` is set."""
        request = self.build_request(case, run_index)

        if not force:
            cached = self.cache.get(request, self.prompt.content_hash)
            if cached is not None:
                self.summary.record(cached)
                return cached

        scan = scan_for_injection(case.clinical_note)
        system, user = self.prompt.render(clinical_note=case.clinical_note)
        model_request = ModelRequest(
            system=system,
            user=user,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            metadata={"case_id": case.case_id, "run_index": run_index},
        )

        response = self._call_with_retry(request, model_request, scan)
        self.cache.put(request, self.prompt.content_hash, response)
        self.summary.record(response)
        return response

    def _call_with_retry(
        self, request: ClinicalSummaryRequest, model_request: ModelRequest, scan
    ) -> ClinicalSummaryResponse:
        last_error: str | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                model_response = self.provider.generate(model_request)
            except ProviderError as error:
                last_error = f"{type(error).__name__}: {error}"
                if attempt < self.max_attempts and self.retry_delay:
                    time.sleep(self.retry_delay)
                continue

            return ClinicalSummaryResponse(
                case_id=request.case_id,
                summary_text=model_response.text,
                provider=model_response.provider,
                model=model_response.model,
                model_version=model_response.model_version,
                prompt_name=self.prompt.name,
                prompt_version=self.prompt.version,
                prompt_hash=self.prompt.content_hash,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                run_index=request.run_index,
                latency_ms=model_response.latency_ms,
                usage=model_response.usage,
                finish_reason=model_response.finish_reason,
                is_real_model_output=model_response.provider is not ProviderKind.MOCK,
                injection_scan={
                    "suspicious": scan.suspicious,
                    "patterns": scan.patterns,
                },
            )

        return ClinicalSummaryResponse(
            case_id=request.case_id,
            summary_text="",
            provider=self.provider.kind,
            model=self.model,
            prompt_name=self.prompt.name,
            prompt_version=self.prompt.version,
            prompt_hash=self.prompt.content_hash,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            run_index=request.run_index,
            injection_scan={"suspicious": scan.suspicious, "patterns": scan.patterns},
            error=f"generation failed after {self.max_attempts} attempt(s): {last_error}",
        )

    # -- batch --------------------------------------------------------------

    def generate_all(
        self, cases, run_index: int = 0, force: bool = False
    ) -> dict[str, ClinicalSummaryResponse]:
        """Generate summaries for every case, continuing past failures."""
        return {
            case.case_id: self.generate(case, run_index=run_index, force=force)
            for case in cases
        }

    def metadata(self) -> dict:
        """Non-secret description of how this generator is configured."""
        return {
            **self.provider.describe(),
            "model": self.model,
            **self.prompt.metadata(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_attempts": self.max_attempts,
            "cache": self.cache.stats(),
        }


def to_ai_outputs(responses: dict[str, ClinicalSummaryResponse]) -> dict:
    """Convert successful generations into the `AIOutput`s evaluators consume.

    Failed generations are excluded rather than passed through as empty
    strings: an empty summary is a model that produced nothing, while a failed
    call is a harness problem, and scoring them alike would misattribute
    infrastructure failures to the model.
    """
    return {
        case_id: response.to_ai_output()
        for case_id, response in responses.items()
        if response.succeeded
    }
