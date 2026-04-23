"""
Cerebras provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class CerebrasProvider(OpenAICompatibleProvider):
    """Adapter for Cerebras cloud inference API.

    Cerebras speaks native OpenAI format. Known for wafer-scale
    processor hardware delivering very high throughput.
    """

    SUPPORTED_MODELS = ["llama3.1-8b", "llama3.1-70b", "llama3.3-70b"]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
