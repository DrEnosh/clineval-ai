"""Provider registry.

Name -> factory, so a provider is chosen by configuration or a CLI flag rather
than by import. Adding a backend means registering it; nothing above this layer
changes.

`build_provider` is the single entry point. It refuses to construct a remote
provider without a credential, which is what guarantees the default path stays
offline: with nothing configured, only the mock provider can be built.
"""

from __future__ import annotations

from typing import Callable

from app.core.config import Settings, settings as default_settings
from app.core.enums import ProviderKind
from app.providers.base import ModelProvider, ProviderNotConfigured
from app.providers.mock import MockProvider
from app.providers.remote import (
    AnthropicProvider,
    GeminiProvider,
    LocalOpenAICompatibleProvider,
    OpenAIProvider,
)

ProviderFactory = Callable[..., ModelProvider]

_REGISTRY: dict[str, ProviderFactory] = {
    ProviderKind.MOCK.value: MockProvider,
    ProviderKind.ANTHROPIC.value: AnthropicProvider,
    ProviderKind.OPENAI.value: OpenAIProvider,
    ProviderKind.GEMINI.value: GeminiProvider,
    ProviderKind.LOCAL.value: LocalOpenAICompatibleProvider,
}


def register(name: str, factory: ProviderFactory, *, replace: bool = False) -> None:
    if name in _REGISTRY and not replace:
        raise ValueError(
            f"A provider named '{name}' is already registered. Pass replace=True to "
            f"override it deliberately."
        )
    _REGISTRY[name] = factory


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def available() -> list[str]:
    """Every registered provider name, whether or not it is configured."""
    return sorted(_REGISTRY)


def configured(settings: Settings | None = None) -> list[str]:
    """Providers that actually have a usable credential."""
    return (settings or default_settings).configured_providers()


def build_provider(
    name: str, settings: Settings | None = None, **kwargs
) -> ModelProvider:
    """Construct a provider by name.

    Raises `ProviderNotConfigured` with actionable guidance when the provider
    exists but has no credential — the common case, and one that deserves a
    clear message rather than a traceback from inside an SDK.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown provider '{name}'. Available: {', '.join(available())}."
        )

    config = settings or default_settings
    factory = _REGISTRY[name]

    if name == ProviderKind.MOCK.value:
        return factory(**kwargs)

    try:
        return factory(settings=config, **kwargs)
    except ProviderNotConfigured:
        raise
    except TypeError:
        # A registered factory that does not accept `settings`.
        return factory(**kwargs)


def describe_availability(settings: Settings | None = None) -> dict[str, bool]:
    """Which providers could be built right now, for reporting to the user."""
    usable = set(configured(settings))
    return {name: name in usable for name in available()}
