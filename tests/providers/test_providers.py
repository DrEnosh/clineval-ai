"""Tests for the provider abstraction, registry, mock provider, and pricing.

Every test here runs offline. No test in this repository makes a network call
or requires an API key — the mock provider covers the whole interface, which is
the property that keeps CI free and deterministic.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.enums import ProviderKind
from app.providers import registry
from app.providers.base import (
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderCallFailed,
    ProviderNotConfigured,
    TokenUsage,
)
from app.providers.mock import (
    BEHAVIOUR_INJECTIONS,
    BEHAVIOURS,
    MockProvider,
    build_behaviour_responses,
    judgment_json,
    mock_judge,
    mock_summarizer,
)
from app.providers.pricing import PRICE_TABLE, estimate_cost, is_priced, lookup
from app.providers.remote import (
    AnthropicProvider,
    GeminiProvider,
    LocalOpenAICompatibleProvider,
    OpenAIProvider,
)


@pytest.fixture(scope="module")
def cases():
    from app.benchmark import generate_benchmark

    return generate_benchmark(case_count=6, seed=42)


class TestModelRequest:
    def test_cache_key_is_stable(self):
        request = ModelRequest(user="hello", model="m")
        assert request.cache_key() == request.cache_key()

    def test_cache_key_changes_with_content(self):
        base = ModelRequest(user="hello", model="m")
        assert base.cache_key() != ModelRequest(user="goodbye", model="m").cache_key()
        assert base.cache_key() != ModelRequest(user="hello", model="n").cache_key()
        assert (
            base.cache_key()
            != ModelRequest(user="hello", model="m", temperature=0.7).cache_key()
        )

    def test_metadata_does_not_affect_the_cache_key(self):
        """Metadata is bookkeeping, not model input; including it would defeat
        caching across runs."""
        plain = ModelRequest(user="hello", model="m")
        tagged = ModelRequest(user="hello", model="m", metadata={"case_id": "C1"})
        assert plain.cache_key() == tagged.cache_key()

    def test_extra_key_material_is_honoured(self):
        request = ModelRequest(user="hello", model="m")
        assert request.cache_key() != request.cache_key({"prompt_hash": "abc"})


class TestTokenUsage:
    def test_unknown_usage_reports_unknown(self):
        usage = TokenUsage()
        assert not usage.is_known
        assert usage.merged_total() is None

    def test_total_is_derived_when_absent(self):
        assert TokenUsage(input_tokens=10, output_tokens=5).merged_total() == 15

    def test_explicit_total_wins(self):
        assert TokenUsage(input_tokens=10, output_tokens=5, total_tokens=99).merged_total() == 99


class TestMockProvider:
    def test_is_offline(self):
        assert MockProvider().is_remote is False
        assert MockProvider().kind is ProviderKind.MOCK

    def test_returns_the_scripted_response(self):
        provider = MockProvider(responses={"C1": "summary one"})
        response = provider.generate(
            ModelRequest(user="x", model="m", metadata={"case_id": "C1"})
        )
        assert response.text == "summary one"

    def test_falls_back_to_the_default_response(self):
        provider = MockProvider(default_response="fallback")
        response = provider.generate(ModelRequest(user="x", model="m"))
        assert response.text == "fallback"

    def test_responder_takes_precedence(self):
        provider = MockProvider(responder=lambda request: request.user.upper())
        assert provider.generate(ModelRequest(user="abc", model="m")).text == "ABC"

    def test_records_calls_for_assertions(self):
        provider = MockProvider(default_response="x")
        provider.generate(ModelRequest(user="a", model="m"))
        provider.generate(ModelRequest(user="b", model="m"))
        assert provider.call_count == 2
        assert [call.user for call in provider.calls] == ["a", "b"]

    def test_fail_times_raises_then_succeeds(self):
        provider = MockProvider(default_response="ok", fail_times=2)
        with pytest.raises(ProviderCallFailed):
            provider.generate(ModelRequest(user="x", model="m"))
        with pytest.raises(ProviderCallFailed):
            provider.generate(ModelRequest(user="x", model="m"))
        assert provider.generate(ModelRequest(user="x", model="m")).text == "ok"

    def test_reports_estimated_token_counts(self):
        response = MockProvider(default_response="hello world").generate(
            ModelRequest(user="a longer prompt here", model="m")
        )
        assert response.usage.input_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.input_tokens + response.usage.output_tokens
        )

    def test_never_reports_a_cost(self):
        """A mock has no cost; reporting 0.0 would make aggregates look like
        real measurements."""
        response = MockProvider(default_response="x").generate(
            ModelRequest(user="a", model="m")
        )
        assert response.usage.estimated_cost_usd is None

    def test_marks_output_as_not_real(self):
        response = MockProvider(default_response="x").generate(
            ModelRequest(user="a", model="m")
        )
        assert response.provider_config["is_real_model_output"] is False
        assert response.provider is ProviderKind.MOCK

    def test_describe_is_secret_free(self):
        described = MockProvider().describe()
        assert described["is_real_model_output"] is False
        assert not any("key" in key.lower() for key in described)


class TestMockBehaviours:
    def test_all_documented_behaviours_are_buildable(self, cases):
        for behaviour in BEHAVIOURS:
            responses = build_behaviour_responses(cases, behaviour)
            assert set(responses) == {case.case_id for case in cases}

    def test_unknown_behaviour_is_rejected(self, cases):
        with pytest.raises(ValueError, match="Unknown mock behaviour"):
            build_behaviour_responses(cases, "not_a_behaviour")

    def test_correct_behaviour_returns_the_reference_summary(self, cases):
        responses = build_behaviour_responses(cases, "correct")
        assert responses[cases[0].case_id] == cases[0].reference_summary

    def test_empty_behaviour_returns_nothing(self, cases):
        assert build_behaviour_responses(cases, "empty")[cases[0].case_id] == ""

    def test_defect_behaviours_differ_from_the_reference(self, cases):
        """At least some cases must actually exhibit each defect, or the
        behaviour is not testing anything."""
        for behaviour in BEHAVIOUR_INJECTIONS:
            responses = build_behaviour_responses(cases, behaviour)
            changed = sum(
                1 for case in cases if responses[case.case_id] != case.reference_summary
            )
            assert changed > 0, f"behaviour '{behaviour}' changed nothing"

    def test_paraphrase_preserves_length_roughly(self, cases):
        responses = build_behaviour_responses(cases, "paraphrase")
        for case in cases:
            assert abs(len(responses[case.case_id]) - len(case.reference_summary)) < 80

    def test_abstractive_drops_trend_values(self, cases):
        """The abstractive behaviour is the ambiguity the adjudicator exists
        for: meaning preserved, numbers gone."""
        responses = build_behaviour_responses(cases, "abstractive")
        with_trends = [
            case for case in cases
            if case.structured_case and any(
                trend.clinically_significant for trend in case.structured_case.lab_trends
            )
        ]
        assert with_trends, "fixture needs at least one case with a significant trend"
        assert any(
            responses[case.case_id] != case.reference_summary for case in with_trends
        )

    def test_summarizer_factory_wires_responses(self, cases):
        provider = mock_summarizer(cases, "correct")
        response = provider.generate(
            ModelRequest(user="x", model="m", metadata={"case_id": cases[0].case_id})
        )
        assert response.text == cases[0].reference_summary

    def test_behaviours_are_deterministic(self, cases):
        assert build_behaviour_responses(cases, "omission") == build_behaviour_responses(
            cases, "omission"
        )


class TestMockJudge:
    def test_returns_valid_judgment_json(self):
        from app.evaluation.llm.parser import parse_adjudication

        response = mock_judge("CONTRADICTED", 0.8).generate(
            ModelRequest(user="x", model="j")
        )
        parsed = parse_adjudication(response.text)
        assert parsed.succeeded
        assert parsed.response.judgment.value == "CONTRADICTED"
        assert parsed.response.confidence == 0.8

    def test_judgment_json_helper_shape(self):
        import json

        payload = json.loads(judgment_json("SUPPORTED"))
        assert set(payload) == {"judgment", "confidence", "reason", "evidence"}


class TestPricing:
    def test_known_model_is_priced(self):
        assert is_priced("gpt-4o-mini")
        assert lookup("gpt-4o-mini") is not None

    def test_dated_model_resolves_through_its_family_prefix(self):
        assert lookup("anthropic-sonnet-4-20260101") is not None

    def test_longest_prefix_wins(self):
        """'gpt-4o-mini' must not resolve to the more expensive 'gpt-4o'."""
        assert lookup("gpt-4o-mini") is PRICE_TABLE["gpt-4o-mini"]

    def test_unknown_model_is_unpriced(self):
        assert lookup("some-unreleased-model") is None
        assert estimate_cost("some-unreleased-model", 100, 100) is None

    def test_cost_is_computed_from_token_counts(self):
        cost = estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.15 + 0.6)

    def test_missing_token_counts_yield_no_estimate(self):
        """Treating a missing count as zero would understate cost, which is the
        one direction a cost estimate must not err in."""
        assert estimate_cost("gpt-4o-mini", None, 100) is None
        assert estimate_cost("gpt-4o-mini", 100, None) is None

    def test_empty_model_name_is_unpriced(self):
        assert lookup("") is None


class TestRegistry:
    def test_every_provider_kind_is_registered(self):
        assert set(registry.available()) >= {kind.value for kind in ProviderKind}

    def test_mock_builds_without_configuration(self):
        assert isinstance(registry.build_provider("mock"), MockProvider)

    def test_unknown_provider_raises_with_guidance(self):
        with pytest.raises(KeyError, match="Available"):
            registry.build_provider("not_a_provider")

    @pytest.mark.parametrize(
        "name,provider_class",
        [
            ("anthropic", AnthropicProvider),
            ("openai", OpenAIProvider),
            ("gemini", GeminiProvider),
        ],
    )
    def test_remote_provider_without_credential_is_refused(self, name, provider_class):
        """The guarantee that nothing leaves the machine by default."""
        blank = Settings(
            anthropic_api_key=None, openai_api_key=None, google_api_key=None,
            local_base_url=None, _env_file=None,
        )
        with pytest.raises(ProviderNotConfigured, match="CLINEVAL_"):
            registry.build_provider(name, settings=blank)

    def test_local_provider_without_base_url_is_refused(self):
        blank = Settings(local_base_url=None, openai_api_key=None, _env_file=None)
        with pytest.raises(ProviderNotConfigured, match="LOCAL_BASE_URL"):
            registry.build_provider("local", settings=blank)

    def test_configured_credential_allows_construction(self):
        configured = Settings(anthropic_api_key="test-key-not-real", _env_file=None)
        provider = registry.build_provider("anthropic", settings=configured)
        assert provider.kind is ProviderKind.ANTHROPIC
        assert provider.is_remote is True

    def test_describe_availability_reports_per_provider(self):
        blank = Settings(
            anthropic_api_key=None, openai_api_key=None, google_api_key=None,
            local_base_url=None, _env_file=None,
        )
        availability = registry.describe_availability(blank)
        assert availability["mock"] is True
        assert availability["anthropic"] is False

    def test_duplicate_registration_is_refused(self):
        with pytest.raises(ValueError, match="already registered"):
            registry.register("mock", MockProvider)

    def test_replace_allows_a_deliberate_override(self):
        try:
            registry.register("mock", MockProvider, replace=True)
            assert "mock" in registry.available()
        finally:
            registry.register("mock", MockProvider, replace=True)

    def test_a_new_provider_can_be_added(self):
        class CustomProvider(MockProvider):
            kind = ProviderKind.MOCK

        try:
            registry.register("custom", CustomProvider)
            assert isinstance(registry.build_provider("custom"), CustomProvider)
        finally:
            registry.unregister("custom")


class TestSecrets:
    def test_provider_describe_never_leaks_a_credential(self):
        configured = Settings(anthropic_api_key="super-secret-value", _env_file=None)
        provider = AnthropicProvider(settings=configured)
        described = repr(provider.describe())
        assert "super-secret-value" not in described
        assert provider.describe()["credential_configured"] is True

    def test_model_response_carries_no_credential(self):
        response = MockProvider(default_response="x").generate(
            ModelRequest(user="a", model="m")
        )
        assert "super-secret" not in response.model_dump_json()
        assert "api_key" not in response.model_dump_json()

    def test_settings_resolve_per_provider_keys(self):
        configured = Settings(
            anthropic_api_key="a", openai_api_key="o", google_api_key="g", _env_file=None
        )
        assert configured.api_key_for("anthropic") == "a"
        assert configured.api_key_for("openai") == "o"
        assert configured.api_key_for("gemini") == "g"
        assert configured.api_key_for("unknown-provider") is None

    def test_configured_providers_lists_only_usable_ones(self):
        blank = Settings(
            anthropic_api_key=None, openai_api_key=None, google_api_key=None,
            local_base_url=None, _env_file=None,
        )
        assert blank.configured_providers() == ["mock"]

        configured = Settings(anthropic_api_key="a", _env_file=None)
        assert set(configured.configured_providers()) == {"mock", "anthropic"}


class TestProviderInterface:
    def test_base_provider_is_abstract(self):
        with pytest.raises(TypeError):
            ModelProvider()

    def test_a_custom_provider_satisfies_the_interface(self):
        class EchoProvider(ModelProvider):
            kind = ProviderKind.MOCK
            is_remote = False

            def generate(self, request: ModelRequest) -> ModelResponse:
                return ModelResponse(text=request.user, provider=self.kind, model=request.model)

        response = EchoProvider().generate(ModelRequest(user="echo me", model="m"))
        assert response.text == "echo me"
        assert response.is_empty is False

    def test_empty_response_is_detected(self):
        assert ModelResponse(text="   ", provider=ProviderKind.MOCK, model="m").is_empty
