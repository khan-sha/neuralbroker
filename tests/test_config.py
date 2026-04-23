"""Tests for configuration loading and validation."""
import pytest
import tempfile
from pathlib import Path

from neuralbrok.config import load_config, Config, LocalNodeConfig, CloudProviderConfig


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_example_config(self):
        """Test loading the example config file."""
        config = load_config("config.yaml.example")
        assert isinstance(config, Config)
        assert len(config.local_nodes) > 0
        assert len(config.cloud_providers) > 0

    def test_config_file_not_found(self):
        """Test error on missing config file."""
        with pytest.raises(SystemExit) as excinfo:
            load_config("nonexistent.yaml")
        assert excinfo.value.code == 1

    def test_invalid_yaml(self):
        """Test error on invalid YAML."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            temp_path = f.name
            f.write("invalid: yaml: content: [")
            f.flush()

        try:
            with pytest.raises(ValueError, match="Invalid YAML"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)




class TestLocalNodeConfig:
    """Tests for LocalNodeConfig model."""

    def test_valid_node(self):
        """Test creating a valid local node config."""
        node = LocalNodeConfig(
            name="ollama-1",
            runtime="ollama",
            host="localhost:11434",
            vram_threshold=0.85,
        )
        assert node.name == "ollama-1"
        assert node.vram_threshold == 0.85

    def test_custom_threshold(self):
        """Test custom VRAM threshold."""
        node = LocalNodeConfig(
            name="ollama-1",
            runtime="ollama",
            host="localhost:11434",
            vram_threshold=0.90,
        )
        assert node.vram_threshold == 0.90

    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValueError):
            LocalNodeConfig(
                name="ollama-1",
                runtime="ollama",
                host="localhost:11434",
                extra_field="should_fail",
            )


class TestCloudProviderConfig:
    """Tests for CloudProviderConfig model."""

    def test_valid_provider(self):
        """Test creating a valid cloud provider config."""
        provider = CloudProviderConfig(
            name="groq",
            api_key_env="GROQ_API_KEY",
            base_url="https://api.groq.com/openai/v1",
        )
        assert provider.name == "groq"
        assert provider.api_key_env == "GROQ_API_KEY"


class TestConfigModel:
    """Tests for Config model."""

    def test_example_config_structure(self):
        """Test structure of loaded example config."""
        config = load_config("config.yaml.example")

        # Check local nodes
        assert len(config.local_nodes) == 1
        node = config.local_nodes[0]
        assert node.name == "ollama-default"
        assert node.runtime == "ollama"
        assert node.host == "localhost:11434"

        # Check cloud providers
        assert len(config.cloud_providers) == 1
        provider = config.cloud_providers[0]
        assert provider.name == "groq"

        # Check routing config
        assert config.routing is not None
        assert config.routing.default_mode == "cost"
