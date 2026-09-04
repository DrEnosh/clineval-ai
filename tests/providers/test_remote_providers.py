"""Tests for the remote provider adapters, with every SDK faked.

No network call is made and no API key is required. Each provider's SDK is
replaced with a stub injected into `sys.modules`, which lets the adapter's
request shaping, response mapping, usage extraction, and error handling be
tested exactly as they would run against the real client.

This matters because the remote adapters are the code most likely to be wrong
and least likely to be exercised: they only run when someone has configured a
paid provider.
"""

from __future__ import annotations

import sys
import types

import pytest

from app.core.config import Settings
from app.core.enums import ProviderKind
from app.providers.base import ModelRequest, ProviderCallFailed, ProviderNotConfigured
from app.providers.remote import (
    AnthropicProvider,
    GeminiProvider,
    LocalOpenAICompatibleProvider,
    OpenAIProvider,
)


@pytest.fixture
def configured() -> Settings:
    return Settings(
        anthropic_api_key="test-anthropic-key",
        openai_api_key="test-openai-key",
        google_api_key="test-google-key",
        local_base_url="http://localhost:11434/v1",
        _env_file=None,
    )


@pytest.fixture
def request_obj() -> ModelRequest:
    return ModelRequest(
        system="You are an adjudicator.", user="Judge this.",
        model="gpt-4o-mini", temperature=0.0, max_tokens=256,
    )


def _install(monkeypatch, name: str, module: types.ModuleType) -> None:
    """Install a fake SDK module so the adapter's lazy import resolves to it."""
    monkeypatch.setitem(sys.modules, name, module)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _fake_anthropic(*, text="judgment text", raise_error=None) -> types.ModuleType:
    module = types.ModuleType("anthropic")
    module.NOT_GIVEN = object()

    class Usage:
        input_tokens = 120
        output_tokens = 30

    class Message:
        id = "msg_123"
        model = "anthropic-sonnet-4-20260101"
        stop_reason = "end_turn"
        usage = Usage()
        content = [types.SimpleNamespace(type="text", text=text)]

    class Messages:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            module.last_kwargs = kwargs
            if raise_error:
                raise raise_error
            return Message()

    class Anthropic:
        def __init__(self, api_key=None):
            module.last_api_key = api_key
            self.messages = Messages()

    module.Anthropic = Anthropic
    return module


class TestAnthropicProvider:
    def test_generates_and_maps_the_response(self, monkeypatch, configured, request_obj):
        module = _fake_anthropic(text="SUPPORTED")
        _install(monkeypatch, "anthropic", module)

        response = AnthropicProvider(settings=configured).generate(request_obj)
        assert response.text == "SUPPORTED"
        assert response.provider is ProviderKind.ANTHROPIC
        assert response.model_version == "anthropic-sonnet-4-20260101"
        assert response.finish_reason == "end_turn"
        assert response.latency_ms is not None

    def test_usage_and_cost_are_extracted(self, monkeypatch, configured):
        _install(monkeypatch, "anthropic", _fake_anthropic())
        request = ModelRequest(user="x", model="anthropic-sonnet-4-20260101")
        response = AnthropicProvider(settings=configured).generate(request)
        assert response.usage.input_tokens == 120
        assert response.usage.output_tokens == 30
        assert response.usage.total_tokens == 150
        assert response.usage.estimated_cost_usd is not None

    def test_unpriced_model_reports_no_cost(self, monkeypatch, configured):
        _install(monkeypatch, "anthropic", _fake_anthropic())
        request = ModelRequest(user="x", model="anthropic-unreleased-99")
        response = AnthropicProvider(settings=configured).generate(request)
        assert response.usage.estimated_cost_usd is None

    def test_system_prompt_uses_the_native_channel(self, monkeypatch, configured, request_obj):
        """Channel separation is the strongest boundary against injection."""
        module = _fake_anthropic()
        _install(monkeypatch, "anthropic", module)
        AnthropicProvider(settings=configured).generate(request_obj)
        assert module.last_kwargs["system"] == "You are an adjudicator."
        assert module.last_kwargs["messages"] == [
            {"role": "user", "content": "Judge this."}
        ]

    def test_sdk_failure_becomes_a_provider_error(self, monkeypatch, configured, request_obj):
        _install(monkeypatch, "anthropic", _fake_anthropic(raise_error=RuntimeError("429")))
        with pytest.raises(ProviderCallFailed, match="Anthropic call failed"):
            AnthropicProvider(settings=configured).generate(request_obj)

    def test_missing_sdk_is_reported_as_not_configured(self, monkeypatch, configured, request_obj):
        """The fix is to install a package, not to retry."""
        monkeypatch.setitem(sys.modules, "anthropic", None)
        monkeypatch.setattr(
            "importlib.import_module",
            lambda name: (_ for _ in ()).throw(ImportError(name)),
        )
        with pytest.raises(ProviderNotConfigured, match="pip install anthropic"):
            AnthropicProvider(settings=configured).generate(request_obj)

    def test_raw_capture_is_opt_in(self, monkeypatch, configured, request_obj):
        _install(monkeypatch, "anthropic", _fake_anthropic())
        plain = AnthropicProvider(settings=configured).generate(request_obj)
        assert plain.raw_response is None

        capturing = AnthropicProvider(settings=configured, capture_raw=True)
        assert capturing.generate(request_obj).raw_response == {"id": "msg_123"}

    def test_credential_is_passed_but_never_stored_in_the_response(
        self, monkeypatch, configured, request_obj
    ):
        module = _fake_anthropic()
        _install(monkeypatch, "anthropic", module)
        response = AnthropicProvider(settings=configured).generate(request_obj)
        assert module.last_api_key == "test-anthropic-key"
        assert "test-anthropic-key" not in response.model_dump_json()


# ---------------------------------------------------------------------------
# OpenAI and local
# ---------------------------------------------------------------------------


def _fake_openai(*, text="ok", raise_error=None) -> types.ModuleType:
    module = types.ModuleType("openai")

    class Usage:
        prompt_tokens = 200
        completion_tokens = 40

    class Completion:
        id = "cmpl_1"
        model = "gpt-4o-mini-2024"
        usage = Usage()
        choices = [
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=text), finish_reason="stop"
            )
        ]

    class Completions:
        def create(self, **kwargs):
            module.last_kwargs = kwargs
            if raise_error:
                raise raise_error
            return Completion()

    class Chat:
        completions = Completions()

    class OpenAI:
        def __init__(self, api_key=None, base_url=None):
            module.last_api_key = api_key
            module.last_base_url = base_url
            self.chat = Chat()

    module.OpenAI = OpenAI
    return module


class TestOpenAIProvider:
    def test_generates_and_maps_the_response(self, monkeypatch, configured, request_obj):
        _install(monkeypatch, "openai", _fake_openai(text="SUPPORTED"))
        response = OpenAIProvider(settings=configured).generate(request_obj)
        assert response.text == "SUPPORTED"
        assert response.provider is ProviderKind.OPENAI
        assert response.finish_reason == "stop"

    def test_usage_maps_from_prompt_and_completion_tokens(
        self, monkeypatch, configured, request_obj
    ):
        _install(monkeypatch, "openai", _fake_openai())
        response = OpenAIProvider(settings=configured).generate(request_obj)
        assert response.usage.input_tokens == 200
        assert response.usage.output_tokens == 40
        assert response.usage.estimated_cost_usd is not None

    def test_system_message_is_sent_separately(self, monkeypatch, configured, request_obj):
        module = _fake_openai()
        _install(monkeypatch, "openai", module)
        OpenAIProvider(settings=configured).generate(request_obj)
        roles = [message["role"] for message in module.last_kwargs["messages"]]
        assert roles == ["system", "user"]

    def test_absent_system_message_is_omitted(self, monkeypatch, configured):
        module = _fake_openai()
        _install(monkeypatch, "openai", module)
        OpenAIProvider(settings=configured).generate(ModelRequest(user="x", model="m"))
        assert [m["role"] for m in module.last_kwargs["messages"]] == ["user"]

    def test_null_content_becomes_empty_text(self, monkeypatch, configured, request_obj):
        _install(monkeypatch, "openai", _fake_openai(text=None))
        assert OpenAIProvider(settings=configured).generate(request_obj).text == ""

    def test_sdk_failure_becomes_a_provider_error(self, monkeypatch, configured, request_obj):
        _install(monkeypatch, "openai", _fake_openai(raise_error=RuntimeError("500")))
        with pytest.raises(ProviderCallFailed, match="OpenAI call failed"):
            OpenAIProvider(settings=configured).generate(request_obj)


class TestLocalProvider:
    def test_is_not_remote(self, monkeypatch, configured, request_obj):
        _install(monkeypatch, "openai", _fake_openai())
        provider = LocalOpenAICompatibleProvider(settings=configured)
        assert provider.is_remote is False
        assert provider.kind is ProviderKind.LOCAL

    def test_uses_the_configured_base_url(self, monkeypatch, configured, request_obj):
        module = _fake_openai()
        _install(monkeypatch, "openai", module)
        LocalOpenAICompatibleProvider(settings=configured).generate(request_obj)
        assert module.last_base_url == "http://localhost:11434/v1"

    def test_requires_a_base_url(self):
        blank = Settings(local_base_url=None, openai_api_key=None, _env_file=None)
        with pytest.raises(ProviderNotConfigured, match="LOCAL_BASE_URL"):
            LocalOpenAICompatibleProvider(settings=blank)

    def test_works_without_a_real_api_key(self, monkeypatch):
        """Most local servers ignore the key entirely."""
        module = _fake_openai()
        _install(monkeypatch, "openai", module)
        settings = Settings(
            local_base_url="http://localhost:8000/v1", openai_api_key=None, _env_file=None
        )
        provider = LocalOpenAICompatibleProvider(settings=settings)
        provider.generate(ModelRequest(user="x", model="local-model"))
        assert module.last_api_key == "not-required"


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def _fake_gemini(*, text="ok", raise_error=None) -> types.ModuleType:
    package = types.ModuleType("google")
    module = types.ModuleType("google.generativeai")

    class Usage:
        prompt_token_count = 90
        candidates_token_count = 25

    class Result:
        usage_metadata = Usage()

        def __init__(self, value):
            self.text = value

    class GenerativeModel:
        def __init__(self, model_name=None, system_instruction=None):
            module.last_model_name = model_name
            module.last_system = system_instruction

        def generate_content(self, prompt, generation_config=None):
            module.last_prompt = prompt
            module.last_config = generation_config
            if raise_error:
                raise raise_error
            return Result(text)

    module.GenerativeModel = GenerativeModel
    module.configure = lambda api_key=None: setattr(module, "last_api_key", api_key)
    package.generativeai = module
    return package, module


class TestGeminiProvider:
    def test_generates_and_maps_the_response(self, monkeypatch, configured, request_obj):
        package, module = _fake_gemini(text="SUPPORTED")
        _install(monkeypatch, "google", package)
        _install(monkeypatch, "google.generativeai", module)

        response = GeminiProvider(settings=configured).generate(request_obj)
        assert response.text == "SUPPORTED"
        assert response.provider is ProviderKind.GEMINI
        assert response.usage.input_tokens == 90
        assert response.usage.output_tokens == 25

    def test_system_instruction_uses_the_native_channel(
        self, monkeypatch, configured, request_obj
    ):
        package, module = _fake_gemini()
        _install(monkeypatch, "google", package)
        _install(monkeypatch, "google.generativeai", module)

        GeminiProvider(settings=configured).generate(request_obj)
        assert module.last_system == "You are an adjudicator."
        assert module.last_prompt == "Judge this."

    def test_generation_config_is_passed(self, monkeypatch, configured, request_obj):
        package, module = _fake_gemini()
        _install(monkeypatch, "google", package)
        _install(monkeypatch, "google.generativeai", module)

        GeminiProvider(settings=configured).generate(request_obj)
        assert module.last_config["temperature"] == 0.0
        assert module.last_config["max_output_tokens"] == 256

    def test_sdk_failure_becomes_a_provider_error(self, monkeypatch, configured, request_obj):
        package, module = _fake_gemini(raise_error=RuntimeError("quota"))
        _install(monkeypatch, "google", package)
        _install(monkeypatch, "google.generativeai", module)

        with pytest.raises(ProviderCallFailed, match="Gemini call failed"):
            GeminiProvider(settings=configured).generate(request_obj)


class TestCredentialHandling:
    @pytest.mark.parametrize(
        "provider_class,env_fragment",
        [
            (AnthropicProvider, "ANTHROPIC_API_KEY"),
            (OpenAIProvider, "OPENAI_API_KEY"),
            (GeminiProvider, "GOOGLE_API_KEY"),
        ],
    )
    def test_missing_credential_names_the_variable_to_set(
        self, provider_class, env_fragment
    ):
        blank = Settings(
            anthropic_api_key=None, openai_api_key=None, google_api_key=None,
            local_base_url=None, _env_file=None,
        )
        with pytest.raises(ProviderNotConfigured) as error:
            provider_class(settings=blank)
        assert env_fragment in str(error.value)
        assert "Nothing is sent anywhere" in str(error.value)

    def test_explicit_key_overrides_settings(self, monkeypatch, request_obj):
        module = _fake_openai()
        _install(monkeypatch, "openai", module)
        blank = Settings(openai_api_key=None, _env_file=None)
        OpenAIProvider(api_key="explicit-key", settings=blank).generate(request_obj)
        assert module.last_api_key == "explicit-key"

    def test_provider_config_in_the_response_holds_no_secret(
        self, monkeypatch, configured, request_obj
    ):
        """provider_config is written verbatim into run files."""
        _install(monkeypatch, "openai", _fake_openai())
        response = OpenAIProvider(settings=configured).generate(request_obj)
        assert set(response.provider_config) == {"temperature", "max_tokens", "base_url"}
        assert "test-openai-key" not in str(response.provider_config)
