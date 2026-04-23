"""
OctoAI provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class OctoAIProvider(OpenAICompatibleProvider):
    """Adapter for OctoAI inference API.

    OctoAI speaks native OpenAI format. Efficient serverless
    inference with automatic hardware scaling.
    """

    SUPPORTED_MODELS = [
        "meta-llama-3.1-8b-instruct",
        "meta-llama-3.1-70b-instruct",
        "mixtral-8x22b-instruct",
    ]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
