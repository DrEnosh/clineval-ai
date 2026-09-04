"""Remote model providers: Anthropic, OpenAI, Gemini, and OpenAI-compatible local.

Each SDK is imported **inside** the call that needs it, so none is a hard
dependency. The application installs and runs with no provider SDK present;
attempting to use one that is not installed raises `ProviderNotConfigured`
with an instruction to install it, rather than failing at import time and
taking the whole package down.

Credentials come from `Settings` (the `CLINEVAL_` namespace) and are held only
on the instance. They are never placed in `ModelResponse`, never written to a
run file, and never logged — `describe()` and `provider_config` carry only
non-secret settings.

The local provider is the OpenAI client pointed at a self-hosted
OpenAI-compatible server (Ollama, vLLM, LM Studio), which is the cheapest way
to support local models without a fourth SDK.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.config import Settings, settings as default_settings
from app.core.enums import ProviderKind
from app.providers.base import (
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderCallFailed,
    ProviderNotConfigured,
    TokenUsage,
)
from app.providers.pricing import estimate_cost


class _RemoteProvider(ModelProvider):
    """Shared credential handling and response assembly."""

    is_remote = True
    #: pip package that supplies this provider's SDK.
    sdk_package: str = ""

    def __init__(
        self,
        api_key: str | None = None,
        settings: Settings | None = None,
        base_url: str | None = None,
        capture_raw: bool = False,
    ):
        config = settings or default_settings
        self._api_key = api_key or config.api_key_for(self.kind.value)
        self._base_url = base_url
        self._capture_raw = capture_raw

        if not self._api_key:
            raise ProviderNotConfigured(
                f"No credential for provider '{self.kind.value}'. Set "
                f"CLINEVAL_{self._env_var()} in your environment or .env file. "
                f"Nothing is sent anywhere until you do."
            )

    def _env_var(self) -> str:
        return {
            ProviderKind.ANTHROPIC: "ANTHROPIC_API_KEY",
            ProviderKind.OPENAI: "OPENAI_API_KEY",
            ProviderKind.GEMINI: "GOOGLE_API_KEY",
            ProviderKind.LOCAL: "LOCAL_BASE_URL",
        }.get(self.kind, "LLM_API_KEY")

    def _require_sdk(self, module_name: str):
        try:
            import importlib

            return importlib.import_module(module_name)
        except ImportError as error:  # pragma: no cover - environment-specific
            raise ProviderNotConfigured(
                f"The '{self.sdk_package}' package is required for the "
                f"{self.kind.value} provider. Install it with: pip install {self.sdk_package}"
            ) from error

    def _response(
        self,
        request: ModelRequest,
        text: str,
        started: float,
        model_version: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        finish_reason: str | None,
        raw: Any = None,
    ) -> ModelResponse:
        total = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        return ModelResponse(
            text=text,
            provider=self.kind,
            model=request.model,
            model_version=model_version,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total,
                estimated_cost_usd=estimate_cost(request.model, input_tokens, output_tokens),
            ),
            finish_reason=finish_reason,
            # Secrets are deliberately absent from this dict; it is written to
            # run files verbatim.
            provider_config={
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "base_url": self._base_url,
            },
            raw_response=raw if self._capture_raw else None,
        )

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "base_url": self._base_url,
            "credential_configured": bool(self._api_key),
        }


class AnthropicProvider(_RemoteProvider):
    kind = ProviderKind.ANTHROPIC
    sdk_package = "anthropic"

    def generate(self, request: ModelRequest) -> ModelResponse:
        anthropic = self._require_sdk("anthropic")
        client = anthropic.Anthropic(api_key=self._api_key)
        started = time.perf_counter()

        try:
            message = client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=request.system or anthropic.NOT_GIVEN,
                messages=[{"role": "user", "content": request.user}],
                stop_sequences=request.stop or anthropic.NOT_GIVEN,
            )
        except Exception as error:  # noqa: BLE001 - SDK exceptions vary
            raise ProviderCallFailed(f"Anthropic call failed: {error}") from error

        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(message, "usage", None)
        return self._response(
            request, text, started,
            model_version=getattr(message, "model", None),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            finish_reason=getattr(message, "stop_reason", None),
            raw={"id": getattr(message, "id", None)},
        )


class OpenAIProvider(_RemoteProvider):
    kind = ProviderKind.OPENAI
    sdk_package = "openai"

    def generate(self, request: ModelRequest) -> ModelResponse:
        openai = self._require_sdk("openai")
        client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
        started = time.perf_counter()

        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.user})

        try:
            completion = client.chat.completions.create(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop or None,
            )
        except Exception as error:  # noqa: BLE001
            raise ProviderCallFailed(f"OpenAI call failed: {error}") from error

        choice = completion.choices[0]
        usage = getattr(completion, "usage", None)
        return self._response(
            request, choice.message.content or "", started,
            model_version=getattr(completion, "model", None),
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=getattr(choice, "finish_reason", None),
            raw={"id": getattr(completion, "id", None)},
        )


class GeminiProvider(_RemoteProvider):
    kind = ProviderKind.GEMINI
    sdk_package = "google-generativeai"

    def generate(self, request: ModelRequest) -> ModelResponse:
        genai = self._require_sdk("google.generativeai")
        genai.configure(api_key=self._api_key)
        started = time.perf_counter()

        try:
            model = genai.GenerativeModel(
                model_name=request.model, system_instruction=request.system or None
            )
            result = model.generate_content(
                request.user,
                generation_config={
                    "temperature": request.temperature,
                    "max_output_tokens": request.max_tokens,
                    "stop_sequences": request.stop or None,
                },
            )
        except Exception as error:  # noqa: BLE001
            raise ProviderCallFailed(f"Gemini call failed: {error}") from error

        usage = getattr(result, "usage_metadata", None)
        return self._response(
            request, getattr(result, "text", "") or "", started,
            model_version=request.model,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            finish_reason=None,
        )


class LocalOpenAICompatibleProvider(OpenAIProvider):
    """A locally-hosted model behind an OpenAI-compatible API.

    Ollama, vLLM, and LM Studio all expose this interface, so one adapter
    covers local models without a fourth SDK. Most such servers ignore the API
    key, hence the placeholder default.
    """

    kind = ProviderKind.LOCAL
    is_remote = False

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        settings: Settings | None = None,
        capture_raw: bool = False,
    ):
        config = settings or default_settings
        resolved = base_url or config.local_base_url
        if not resolved:
            raise ProviderNotConfigured(
                "No base URL for the local provider. Set CLINEVAL_LOCAL_BASE_URL "
                "to your OpenAI-compatible server (e.g. http://localhost:11434/v1)."
            )
        super().__init__(
            api_key=api_key or config.openai_api_key or "not-required",
            settings=config,
            base_url=resolved,
            capture_raw=capture_raw,
        )
