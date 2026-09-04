"""Model provider abstraction.

One interface — `ModelProvider.generate(request) -> ModelResponse` — behind
which any backend sits: a hosted API, a local server, or the deterministic
mock. Evaluation code never imports a provider directly, so adding a backend
changes nothing above this layer.

Three rules hold for every implementation:

**No credential is ever hard-coded, stored, or logged.** A provider reads its
key from `Settings` at construction and keeps it out of every serialized
shape. `ModelResponse.provider_config` is the configuration *minus* secrets,
and is what gets written to run files.

**Nothing leaves the machine unless a provider is explicitly configured.** The
default path is the mock provider. A remote provider requires a credential the
user set deliberately.

**Failures are typed, not swallowed.** A provider raises `ProviderError` (or a
subclass) so callers can distinguish "not configured", "transport failed", and
"the model returned something unusable" — three problems with different fixes.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.core.enums import ProviderKind


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class ProviderNotConfigured(ProviderError):
    """No usable credential or dependency for this provider.

    Distinct from a transport failure because the fix is different: configure
    a key, or install an SDK, rather than retry.
    """


class ProviderCallFailed(ProviderError):
    """The provider was reached but the call did not succeed."""


class TokenUsage(BaseModel):
    """Token counts and cost for one call.

    Every field is optional and defaults to None. A provider that does not
    report usage leaves these unset — the system records "unknown" rather
    than inventing a number, because a fabricated cost figure is worse than
    an absent one.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None

    @property
    def is_known(self) -> bool:
        return self.total_tokens is not None or self.input_tokens is not None

    def merged_total(self) -> int | None:
        if self.total_tokens is not None:
            return self.total_tokens
        if self.input_tokens is not None and self.output_tokens is not None:
            return self.input_tokens + self.output_tokens
        return None


class ModelRequest(BaseModel):
    """One generation request.

    `system` and `user` are kept separate rather than concatenated so
    providers can use their native system-prompt channel, which is also the
    strongest available boundary between evaluator instructions and untrusted
    clinical content (see app/prompts/sanitize.py).
    """

    system: str | None = None
    user: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    stop: list[str] = Field(default_factory=list)
    # Free-form, never sent to the provider. Carries case_id and similar so a
    # mock can respond deterministically and so responses can be correlated.
    metadata: dict[str, Any] = Field(default_factory=dict)

    def cache_key(self, extra: dict[str, Any] | None = None) -> str:
        """A stable hash of everything that determines the response.

        Metadata is excluded: it is bookkeeping, not input to the model, and
        including it would defeat caching across runs.
        """
        payload = {
            "system": self.system,
            "user": self.user,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": sorted(self.stop),
            **(extra or {}),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ModelResponse(BaseModel):
    """One generation response, with everything needed to reproduce it.

    Carries no credential. `provider_config` holds the non-secret settings
    that shaped the call, so a stored run says exactly how it was produced.
    """

    text: str
    provider: ProviderKind
    model: str
    model_version: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str | None = None
    provider_config: dict[str, Any] = Field(default_factory=dict)
    from_cache: bool = False
    # Raw provider payload, retained only when the caller opts in. Off by
    # default: raw responses are large and may echo clinical content.
    raw_response: dict[str, Any] | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class ModelProvider(ABC):
    """Backend that turns a `ModelRequest` into a `ModelResponse`."""

    kind: ProviderKind
    #: Whether this provider sends data off the machine. Used to warn the user.
    is_remote: bool = True

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Run one generation. Raises `ProviderError` on failure."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """Non-secret description of this provider, for run metadata."""
        return {"provider": self.kind.value, "is_remote": self.is_remote}
