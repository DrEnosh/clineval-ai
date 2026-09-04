"""Deterministic mock provider.

Runs the entire pipeline — generation, deterministic evaluation, LLM
adjudication — with no network access and no API key. **CI and the test suite
use only this provider**, so no test ever depends on a paid external model or
on a remote service being reachable.

Two factories cover the two roles a model plays in this system:

- `mock_summarizer` produces clinical summaries with a chosen defect, built by
  running the Phase 3 injection harness over the benchmark's own reference
  summaries. Reusing that harness means a mock "hallucinating model" produces
  exactly the corruption the evaluator was validated against, rather than a
  separately-invented approximation of one.
- `mock_judge` returns scripted adjudication JSON, including deliberately
  malformed responses so the parser's failure handling is exercised.

Outputs are labelled `is_real_model_output: False` in provider config, and the
provider kind is `MOCK`, so a mock generation can never be mistaken for a real
one in a stored run.
"""

from __future__ import annotations

import time
from typing import Callable

from app.core.enums import ProviderKind
from app.providers.base import (
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderCallFailed,
    TokenUsage,
)

Responder = Callable[[ModelRequest], str]

# Rough token estimate for mock usage figures: ~4 characters per token. Marked
# as estimated in the response so it is never mistaken for a provider count.
_CHARS_PER_TOKEN = 4


class MockProvider(ModelProvider):
    """A provider that answers from a script rather than a model."""

    kind = ProviderKind.MOCK
    is_remote = False

    def __init__(
        self,
        responder: Responder | None = None,
        responses: dict[str, str] | None = None,
        default_response: str = "",
        model: str = "mock-model-v1",
        fail_times: int = 0,
        latency_ms: float = 0.0,
    ):
        """
        `responses` keys on `request.metadata["case_id"]`; `responder` is the
        general form. `fail_times` makes the first N calls raise, which is how
        retry behaviour is tested without patching internals.
        """
        self._responder = responder
        self._responses = responses or {}
        self._default_response = default_response
        self._model = model
        self._remaining_failures = fail_times
        self._latency_ms = latency_ms
        self.calls: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)

        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise ProviderCallFailed(
                f"Mock provider failing on purpose ({self._remaining_failures} remaining)"
            )

        started = time.perf_counter()
        text = self._resolve(request)
        latency = self._latency_ms or (time.perf_counter() - started) * 1000

        input_tokens = max(1, len(f"{request.system or ''}{request.user}") // _CHARS_PER_TOKEN)
        output_tokens = max(1, len(text) // _CHARS_PER_TOKEN)

        return ModelResponse(
            text=text,
            provider=self.kind,
            model=request.model or self._model,
            model_version="mock",
            latency_ms=round(latency, 3),
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                # Deliberately None: a mock has no cost, and reporting 0.0
                # would make cost aggregates look like real measurements.
                estimated_cost_usd=None,
            ),
            finish_reason="stop",
            provider_config={
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "is_real_model_output": False,
                "token_counts": "estimated_by_mock",
            },
        )

    def _resolve(self, request: ModelRequest) -> str:
        if self._responder is not None:
            return self._responder(request)
        case_id = str(request.metadata.get("case_id", ""))
        return self._responses.get(case_id, self._default_response)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def describe(self) -> dict:
        return {**super().describe(), "model": self._model, "is_real_model_output": False}


# ---------------------------------------------------------------------------
# Summarizer behaviours
# ---------------------------------------------------------------------------

#: Behaviour name -> the Phase 3 injection that produces it. `correct` and the
#: degenerate behaviours are handled separately.
BEHAVIOUR_INJECTIONS: dict[str, str] = {
    "omission": "remove_critical_fact",
    "hallucination": "add_unsupported_diagnosis",
    "contradiction": "change_lab_value",
    "medication_error": "change_medication_status",
    "dose_error": "change_dose",
    "negation_reversal": "reverse_negation",
    "temporal_reversal": "reverse_temporal_trend",
    "historical_confusion": "present_resolved_as_current",
}

BEHAVIOURS: tuple[str, ...] = (
    "correct", "empty", "verbatim_note", "paraphrase", "abstractive",
    *sorted(BEHAVIOUR_INJECTIONS),
)

#: Analyte -> clinically equivalent generalization, for the abstractive
#: behaviour. Each preserves the clinical meaning while dropping the numbers,
#: which is the case deterministic matching genuinely cannot settle.
_ABSTRACTIONS: dict[str, tuple[str, str]] = {
    "Creatinine": ("Renal function has deteriorated", "Renal function has improved"),
    "eGFR": ("Renal function has deteriorated", "Renal function has improved"),
    "Hemoglobin": ("Anemia has worsened", "Anemia has improved"),
    "BNP": ("Heart failure markers have worsened", "Heart failure markers have improved"),
    "White blood cell count": (
        "Inflammatory markers have risen", "Inflammatory markers have settled"
    ),
    "Potassium": ("Potassium has become deranged", "Potassium has normalized"),
    "Lactate": ("Perfusion markers have worsened", "Perfusion markers have improved"),
    "Troponin I": ("Cardiac markers have risen", "Cardiac markers have settled"),
}


def _paraphrase(summary: str) -> str:
    """A meaning-preserving reword, for exercising the adjudication path.

    Substitutions are lexical but semantically neutral: the deterministic
    matcher may score these as partial, which is exactly the ambiguity the LLM
    adjudicator exists to resolve.
    """
    replacements = (
        ("rose from", "increased from"),
        ("fell from", "decreased from"),
        ("was discontinued", "has been stopped"),
        ("was increased from", "was titrated up from"),
        ("Continued on", "Remains on"),
        ("seen for", "presenting for"),
        ("Documented negatives:", "Pertinent negatives:"),
        ("was newly started", "has been initiated"),
    )
    for source, target in replacements:
        summary = summary.replace(source, target)
    return summary


def _abstractive(case) -> str:
    """Replace numeric trend statements with clinical generalizations.

    "Creatinine rose from 1.4 to 1.9 mg/dL" becomes "Renal function has
    deteriorated". The meaning survives; the values do not. Deterministic
    matching correctly reports the values as absent, and whether that counts
    as a failure is a judgment about clinical equivalence - which is precisely
    what the adjudication path exists to settle.
    """
    import re

    structured = case.structured_case
    summary = case.reference_summary
    if structured is None:
        return summary

    for trend in structured.lab_trends:
        if not trend.clinically_significant:
            continue
        abstraction = _ABSTRACTIONS.get(trend.lab_name)
        if abstraction is None:
            continue
        worse, better = abstraction
        rising = trend.later_value > trend.earlier_value
        concerning = rising if trend.lab_name != "eGFR" else not rising
        replacement = f"{worse if concerning else better} since the previous measurement."

        pattern = re.compile(
            rf"{re.escape(trend.lab_name)}\s+(?:rose|fell|remained stable|changed).*?\.(?=\s|$)",
            re.IGNORECASE,
        )
        summary = pattern.sub(replacement, summary, count=1)

    return summary


def build_behaviour_responses(cases, behaviour: str, seed: int = 0) -> dict[str, str]:
    """Map case_id -> summary text exhibiting `behaviour`.

    A case that cannot exhibit the requested defect (no medications for a dose
    error, say) falls back to its correct summary. That is honest: the mock
    reports what it could actually produce, and the run's findings reflect the
    cases where the defect was really present.
    """
    from app.evaluation.injection import inject

    if behaviour not in BEHAVIOURS:
        raise ValueError(
            f"Unknown mock behaviour '{behaviour}'. Available: {', '.join(BEHAVIOURS)}"
        )

    responses: dict[str, str] = {}
    for case in cases:
        if behaviour == "correct":
            responses[case.case_id] = case.reference_summary
        elif behaviour == "empty":
            responses[case.case_id] = ""
        elif behaviour == "verbatim_note":
            responses[case.case_id] = case.clinical_note
        elif behaviour == "paraphrase":
            responses[case.case_id] = _paraphrase(case.reference_summary)
        elif behaviour == "abstractive":
            responses[case.case_id] = _abstractive(case)
        else:
            injection = inject(BEHAVIOUR_INJECTIONS[behaviour], case, seed=seed)
            responses[case.case_id] = (
                injection.corrupted_summary if injection else case.reference_summary
            )
    return responses


def mock_summarizer(cases, behaviour: str = "correct", model: str = "mock-summarizer-v1", **kwargs):
    """A `MockProvider` that summarizes the given cases with `behaviour`."""
    return MockProvider(
        responses=build_behaviour_responses(cases, behaviour),
        model=model,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Judge behaviours
# ---------------------------------------------------------------------------


def judgment_json(
    judgment: str,
    confidence: float = 0.9,
    reason: str = "Mock adjudication.",
    source_span: str = "source evidence",
    output_span: str = "output evidence",
) -> str:
    """A well-formed adjudication payload, for scripting a mock judge."""
    import json

    return json.dumps(
        {
            "judgment": judgment,
            "confidence": confidence,
            "reason": reason,
            "evidence": [{"source_span": source_span, "output_span": output_span}],
        }
    )


def mock_judge(
    judgment: str = "SUPPORTED",
    confidence: float = 0.9,
    responder: Responder | None = None,
    model: str = "mock-judge-v1",
    **kwargs,
) -> MockProvider:
    """A `MockProvider` that returns adjudication JSON."""
    return MockProvider(
        responder=responder,
        default_response=judgment_json(judgment, confidence),
        model=model,
        **kwargs,
    )
