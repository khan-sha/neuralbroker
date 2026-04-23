"""
Hyperbolic provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class HyperbolicProvider(OpenAICompatibleProvider):
    """Adapter for Hyperbolic cloud inference API.

    Hyperbolic speaks native OpenAI format. Decentralized
    GPU marketplace focused on open-source models.
    """

    SUPPORTED_MODELS = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
    ]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
