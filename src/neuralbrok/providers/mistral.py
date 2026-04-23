"""
Mistral AI provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class MistralProvider(OpenAICompatibleProvider):
    """Adapter for Mistral AI inference API.

    Mistral speaks native OpenAI format. First-party API for
    Mistral's own model family including Mixtral.
    """

    SUPPORTED_MODELS = [
        "mistral-small-latest",
        "mistral-medium-latest",
        "mistral-large-latest",
        "mistral-7b-instruct",
    ]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
