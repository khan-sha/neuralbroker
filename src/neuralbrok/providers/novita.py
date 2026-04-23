"""
Novita AI provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class NovitaProvider(OpenAICompatibleProvider):
    """Adapter for Novita AI inference API.

    Novita speaks native OpenAI format. Affordable GPU cloud
    with fast cold start times.
    """

    SUPPORTED_MODELS = [
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
    ]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
