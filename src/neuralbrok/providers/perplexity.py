"""
Perplexity AI provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class PerplexityProvider(OpenAICompatibleProvider):
    """Adapter for Perplexity AI inference API.

    Perplexity speaks native OpenAI format.
    Online models with live search capability — best for
    queries requiring up-to-date information.
    """

    SUPPORTED_MODELS = [
        "llama-3.1-sonar-small-128k-online",
        "llama-3.1-sonar-large-128k-online",
        "llama-3.1-sonar-huge-128k-online",
    ]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
