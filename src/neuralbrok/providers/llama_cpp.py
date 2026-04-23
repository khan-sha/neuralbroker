"""
llama.cpp provider adapter.

llama.cpp's server already exposes an OpenAI-compatible endpoint at
/v1/chat/completions, so this is a thin wrapper.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class LlamaCppProvider(OpenAICompatibleProvider):
    """Adapter for llama.cpp server.

    llama.cpp's built-in server speaks OpenAI format natively.
    Default host is localhost:8080.
    """

    # Local server — model depends on what's loaded; empty means no filtering
    SUPPORTED_MODELS: list[str] = []

    def __init__(self, name: str, host: str, api_key: str = ""):
        base_url = f"http://{host}" if "://" not in host else host
        # Ensure we point to /v1 if not already
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key or "no-key",
            provider_type="local",
        )

    async def health_check(self) -> bool:
        """Check if llama.cpp server is running."""
        import httpx
        try:
            # llama.cpp uses /health endpoint
            base = self.base_url.replace("/v1", "")
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{base}/health")
                return r.status_code == 200
        except Exception:
            return False
