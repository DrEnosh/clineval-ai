"""Strict parsing of LLM judgment responses.

Never parse arbitrary natural language. A judge either produces the schema or
it has failed, and a failure is recorded rather than salvaged by guessing at
intent.

The parser does allow for the *packaging* models put around otherwise valid
JSON — a ```json fence, a sentence of preamble — because that is a formatting
habit, not a content ambiguity. What it will not do is infer a judgment from
prose: text saying "this looks supported to me" yields a parse failure, not
SUPPORTED. Guessing there would fabricate a verdict the model never committed
to, in the one place where the system is trying to measure whether the model
can be trusted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from app.evaluation.llm.schemas import AdjudicationResponse

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class ParseFailure(ValueError):
    """The response could not be parsed into the required schema."""


@dataclass(frozen=True)
class ParseResult:
    response: AdjudicationResponse | None
    error: str | None = None
    strategy: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.response is not None


def _candidates(text: str) -> list[tuple[str, str]]:
    """Candidate JSON strings, most-trusted first, each with its strategy name."""
    stripped = text.strip()
    if not stripped:
        return []

    found: list[tuple[str, str]] = [("direct", stripped)]

    fence = _FENCE.search(stripped)
    if fence:
        found.append(("code_fence", fence.group(1).strip()))

    # The outermost balanced brace span. Handles a model that wrote a sentence
    # before or after an otherwise valid object.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        found.append(("brace_span", stripped[start : end + 1]))

    return found


def parse_adjudication(text: str) -> ParseResult:
    """Parse a judge response into `AdjudicationResponse`.

    Returns a `ParseResult` rather than raising, so the caller can retry and
    then record the failure as a normal (if unhelpful) outcome.
    """
    if not text or not text.strip():
        return ParseResult(None, error="empty response", strategy=None)

    last_error = "no JSON object found in response"

    for strategy, candidate in _candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as error:
            last_error = f"invalid JSON ({strategy}): {error.msg}"
            continue

        if not isinstance(payload, dict):
            last_error = f"expected a JSON object, got {type(payload).__name__}"
            continue

        try:
            return ParseResult(AdjudicationResponse.model_validate(payload), strategy=strategy)
        except ValidationError as error:
            first = error.errors()[0]
            location = ".".join(str(part) for part in first.get("loc", ())) or "response"
            last_error = f"schema validation failed at '{location}': {first.get('msg')}"

    return ParseResult(None, error=last_error)


def parse_or_raise(text: str) -> AdjudicationResponse:
    result = parse_adjudication(text)
    if result.response is None:
        raise ParseFailure(result.error or "unparseable judge response")
    return result.response
