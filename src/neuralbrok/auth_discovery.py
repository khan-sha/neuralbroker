"""
Subscription token inheritance + zero-config provider discovery.

Reads existing auth state from installed CLI tools (Claude Code, Codex)
so users don't need to manually configure API keys. Lets a Claude Pro
or ChatGPT Plus subscription serve any OpenAI-compatible client through
NeuralBroker — subscription arbitrage.

WARNING (legal): Using subscription tokens outside their official client
may violate provider ToS. NB exposes the token; it does not enforce policy.
Users assume responsibility.
"""
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredAuth:
    """Auth credentials discovered from a CLI install."""
    provider: str          # "anthropic" | "openai" | "ollama" | "llamacpp"
    auth_type: str         # "oauth_bearer" | "api_key" | "none"
    token: Optional[str]   # bearer token or api key; None for keyless
    source: str            # human-readable source path
    subscription: bool = False  # True = inherited from paid sub (free at margin)
    expires_at: Optional[int] = None  # unix ms
    extra: Optional[dict] = None


def discover_claude_oauth() -> Optional[DiscoveredAuth]:
    """Read Claude Code OAuth token from ~/.claude/.credentials.json."""
    path = Path.home() / ".claude" / ".credentials.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        oauth = data.get("claudeAiOauth", {})
        token = oauth.get("accessToken")
        if not token:
            return None
        return DiscoveredAuth(
            provider="anthropic",
            auth_type="oauth_bearer",
            token=token,
            source=str(path),
            subscription=True,
            expires_at=oauth.get("expiresAt"),
            extra={
                "subscription_type": oauth.get("subscriptionType", "unknown"),
                "rate_limit_tier": oauth.get("rateLimitTier"),
                "scopes": oauth.get("scopes", []),
            },
        )
    except Exception as e:
        logger.debug(f"Failed to read Claude OAuth: {e}")
        return None


def discover_codex_auth() -> Optional[DiscoveredAuth]:
    """Read Codex auth from ~/.codex/auth.json. Either ChatGPT OAuth or raw API key."""
    path = Path.home() / ".codex" / "auth.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        api_key = data.get("OPENAI_API_KEY")
        if api_key:
            return DiscoveredAuth(
                provider="openai",
                auth_type="api_key",
                token=api_key,
                source=str(path),
                subscription=False,
            )
        tokens = data.get("tokens", {})
        access = tokens.get("access_token")
        if access:
            return DiscoveredAuth(
                provider="openai",
                auth_type="oauth_bearer",
                token=access,
                source=str(path),
                subscription=True,
                extra={
                    "auth_mode": data.get("auth_mode"),
                    "account_id": tokens.get("account_id"),
                },
            )
    except Exception as e:
        logger.debug(f"Failed to read Codex auth: {e}")
    return None


def discover_env_keys() -> list[DiscoveredAuth]:
    """Read API keys from environment variables (fallback path)."""
    found = []
    env_map = {
        "ANTHROPIC_API_KEY": "anthropic",
        "OPENAI_API_KEY": "openai",
        "GROQ_API_KEY": "groq",
        "TOGETHER_API_KEY": "together",
        "DEEPINFRA_API_KEY": "deepinfra",
        "MISTRAL_API_KEY": "mistral",
        "GEMINI_API_KEY": "gemini",
        "COHERE_API_KEY": "cohere",
        "OPENROUTER_API_KEY": "openrouter",
        "FIREWORKS_API_KEY": "fireworks",
        "CEREBRAS_API_KEY": "cerebras",
        "PERPLEXITY_API_KEY": "perplexity",
    }
    for var, prov in env_map.items():
        v = os.getenv(var)
        if v:
            found.append(DiscoveredAuth(
                provider=prov, auth_type="api_key", token=v,
                source=f"env:{var}", subscription=False,
            ))
    return found


def discover_ollama(host: str = "http://localhost:11434") -> Optional[DiscoveredAuth]:
    """Ping local Ollama server."""
    try:
        r = httpx.get(f"{host}/api/version", timeout=1.0)
        if r.status_code == 200:
            return DiscoveredAuth(
                provider="ollama", auth_type="none", token=None,
                source=host, subscription=False,
                extra={"version": r.json().get("version")},
            )
    except Exception:
        pass
    return None


def discover_llamacpp(host: str = "http://localhost:8080") -> Optional[DiscoveredAuth]:
    """Ping local llama.cpp server."""
    try:
        r = httpx.get(f"{host}/health", timeout=1.0)
        if r.status_code == 200:
            return DiscoveredAuth(
                provider="llamacpp", auth_type="none", token=None,
                source=host, subscription=False,
            )
    except Exception:
        pass
    return None


def discover_all() -> dict[str, DiscoveredAuth]:
    """Run full discovery. Returns map provider_name → DiscoveredAuth.

    Subscription-tier auths take precedence over env API keys for the same
    provider — they're effectively free at the margin.
    """
    results: dict[str, DiscoveredAuth] = {}

    # Local-first: ollama + llamacpp
    if (o := discover_ollama()) is not None:
        results["ollama"] = o
    if (l := discover_llamacpp()) is not None:
        results["llamacpp"] = l

    # Env-key fallback (lowest priority for cloud providers)
    for auth in discover_env_keys():
        results[auth.provider] = auth

    # Subscription-inherited (overrides env keys for the same provider)
    if (claude := discover_claude_oauth()) is not None:
        results["anthropic"] = claude
        logger.info(
            f"[auth-discovery] Inherited Claude {claude.extra.get('subscription_type', '?').upper()} "
            f"subscription from {claude.source}"
        )
    if (codex := discover_codex_auth()) is not None:
        # Only override if it's an OAuth subscription token; api_key already covered by env
        if codex.subscription or "openai" not in results:
            results["openai"] = codex
            logger.info(
                f"[auth-discovery] Inherited {'ChatGPT subscription' if codex.subscription else 'OpenAI API key'} "
                f"from {codex.source}"
            )

    if results:
        logger.info(
            f"[auth-discovery] {len(results)} provider(s) auto-discovered: "
            f"{', '.join(results.keys())}"
        )
    return results


def is_token_expired(auth: DiscoveredAuth, skew_seconds: int = 60) -> bool:
    """True if OAuth token is past expiry (with skew). Always False for api_key."""
    if auth.expires_at is None or auth.auth_type != "oauth_bearer":
        return False
    return (auth.expires_at / 1000) < (time.time() + skew_seconds)


def refresh_claude_token() -> Optional[DiscoveredAuth]:
    """Re-read Claude credentials from disk (Claude Code refreshes them automatically)."""
    return discover_claude_oauth()
