"""Model providers.

One interface, several backends. Evaluation code depends on
`ModelProvider`, never on a specific SDK, so a backend can be added without
touching anything above this layer.

The default path is offline: with no credential configured, only the mock
provider can be built, and no benchmark data leaves the machine.
"""

from app.providers.base import (
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderCallFailed,
    ProviderError,
    ProviderNotConfigured,
    TokenUsage,
)
from app.providers.mock import BEHAVIOURS, MockProvider, mock_judge, mock_summarizer
from app.providers.registry import (
    available,
    build_provider,
    configured,
    describe_availability,
    register,
    unregister,
)
from app.providers.remote import (
    AnthropicProvider,
    GeminiProvider,
    LocalOpenAICompatibleProvider,
    OpenAIProvider,
)

__all__ = [
    "BEHAVIOURS",
    "AnthropicProvider",
    "GeminiProvider",
    "LocalOpenAICompatibleProvider",
    "MockProvider",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "OpenAIProvider",
    "ProviderCallFailed",
    "ProviderError",
    "ProviderNotConfigured",
    "TokenUsage",
    "available",
    "build_provider",
    "configured",
    "describe_availability",
    "mock_judge",
    "mock_summarizer",
    "register",
    "unregister",
]
