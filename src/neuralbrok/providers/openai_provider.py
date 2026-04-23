"""
OpenAI provider adapter.

Pass-through to OpenAI's native API. Named openai_provider.py
to avoid shadowing the openai Python package.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """Adapter for OpenAI API.

    OpenAI is the canonical format — zero transformation needed.
    """

    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini",
    ]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
