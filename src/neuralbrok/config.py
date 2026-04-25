"""
Configuration loading and validation.

Supports the expanded config schema with routing modes,
electricity cost calculation, cloud provider pricing, and server settings.
"""
from pathlib import Path
from typing import Optional

import os
import yaml
from pydantic import BaseModel, Field, ConfigDict


class LocalNodeConfig(BaseModel):
    """Configuration for a local backend node (e.g., Ollama, llama.cpp)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    runtime: str = Field(default="ollama")  # ollama | llama_cpp | lm_studio
    host: str = Field(default="localhost:11434")
    vram_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    vram_threshold_gb: Optional[float] = Field(default=None, ge=0.1)
    gpu_index: int = Field(default=0, ge=0)


class CloudProviderConfig(BaseModel):
    """Configuration for a cloud provider backend (e.g., Groq, Together)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    api_key_env: str
    base_url: str = Field(default="")
    cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)


# Default base URLs for known cloud providers
CLOUD_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "openai": "https://api.openai.com/v1",
}

# Default cost per 1k tokens for known providers
CLOUD_COSTS = {
    "groq": 0.00006,
    "together": 0.00020,
    "openai": 0.00060,
}


class RoutingConfig(BaseModel):
    """Routing policy configuration."""

    model_config = ConfigDict(extra="forbid")

    default_mode: str = Field(default="cost")  # cost | speed | fallback
    vram_poll_interval_seconds: float = Field(default=0.5, ge=0.1)
    electricity_kwh_price: float = Field(default=0.14, ge=0.0)
    gpu_tdp_watts: float = Field(default=320.0, ge=0.0)


class ServerConfig(BaseModel):
    """Server configuration."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    api_key_env: str = Field(default="NB_API_KEY")


class CacheConfig(BaseModel):
    """Redis semantic cache configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False)
    redis_url: str = Field(default="redis://localhost:6379")


class IntegrationsConfig(BaseModel):
    """Configuration for AI agent integrations."""

    model_config = ConfigDict(extra="forbid")

    nb_url: str = Field(default="http://localhost:8000")
    api_key_env: str = Field(default="NB_API_KEY")
    auto_setup: list[str] = Field(default_factory=list)


class Config(BaseModel):
    """Root configuration."""

    model_config = ConfigDict(extra="forbid")

    local_nodes: list[LocalNodeConfig] = Field(default_factory=list)
    cloud_providers: list[CloudProviderConfig] = Field(default_factory=list)
    routing: Optional[RoutingConfig] = Field(default_factory=RoutingConfig)
    server: Optional[ServerConfig] = Field(default_factory=ServerConfig)
    cache: Optional[CacheConfig] = Field(default_factory=CacheConfig)
    integrations: Optional[IntegrationsConfig] = Field(default_factory=IntegrationsConfig)


def load_config(path: Optional[str] = None) -> Config:
    """Load and validate configuration from YAML file.

    Priority:
    1. path passed as argument
    2. ~/.neuralbrok/config.yaml
    3. ./config.yaml

    Args:
        path: Optional path to YAML configuration file.

    Returns:
        Validated Config object.

    Raises:
        SystemExit: If no configuration file is found.
        ValueError: If YAML is invalid or fails validation.
    """
    config_path = None
    if path:
        config_path = Path(path)
    else:
        # Check ~/.neuralbrok/config.yaml
        home_config = Path.home() / ".neuralbrok" / "config.yaml"
        if home_config.exists():
            config_path = home_config
        # Check current directory config.yaml
        elif Path("config.yaml").exists():
            config_path = Path("config.yaml")

    if not config_path or not config_path.exists():
        if os.getenv("VERCEL"):
            return Config()
        print("No config found. Run 'neuralbrok setup' to configure.")
        import sys
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}")

    config = Config(**data)

    # Apply default base_url and cost for known cloud providers
    for cp in config.cloud_providers:
        if not cp.base_url and cp.name in CLOUD_BASE_URLS:
            cp.base_url = CLOUD_BASE_URLS[cp.name]
        if cp.cost_per_1k_tokens == 0.0 and cp.name in CLOUD_COSTS:
            cp.cost_per_1k_tokens = CLOUD_COSTS[cp.name]

    if os.environ.get("NB_POLICY_MODE"):
        if not config.routing:
            config.routing = RoutingConfig()
        config.routing.default_mode = os.environ.get("NB_POLICY_MODE")
    return config
