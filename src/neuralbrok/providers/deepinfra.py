"""
DeepInfra provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class DeepInfraProvider(OpenAICompatibleProvider):
    """Adapter for DeepInfra cloud inference API.

    DeepInfra speaks native OpenAI format at their openai-compatible endpoint.
    Hosts a wide variety of open-source models at competitive prices.
    """

    SUPPORTED_MODELS = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
    ]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
