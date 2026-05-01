"""
Dynamic Agent Plugin Loader.
Equivalent to Ruflo's Plugin Marketplace / 100+ agents architecture.
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

from neuralbrok.agents.builtin import AgentCapability

logger = logging.getLogger(__name__)

class PluginLoader:
    def __init__(self, plugin_dir: str = "~/.neuralbrok/plugins"):
        self.plugin_dir = Path(plugin_dir).expanduser()
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        self.custom_agents: Dict[str, AgentCapability] = {}

    def load_all(self):
        """Scans the plugin directory for YAML agent definitions."""
        self.custom_agents.clear()
        
        if not self.plugin_dir.exists():
            return
            
        for yaml_file in self.plugin_dir.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    
                if not data or "name" not in data or "role" not in data:
                    logger.warning(f"Skipping invalid plugin file: {yaml_file}")
                    continue
                    
                agent = AgentCapability(
                    name=data["name"],
                    role=data["role"],
                    system_prompt=data.get("system_prompt", ""),
                    tools=data.get("tools", []),
                    vram_requirement_gb=data.get("vram_requirement_gb", 4.0)
                )
                
                slug = data.get("slug", data["name"].lower().replace(" ", "-"))
                self.custom_agents[slug] = agent
                logger.info(f"Loaded custom agent plugin: {slug}")
            except Exception as e:
                logger.error(f"Failed to load plugin {yaml_file}: {e}")

    def get_agent(self, slug: str) -> AgentCapability:
        return self.custom_agents.get(slug)

    def install_plugin(self, url: str):
        """Simulates downloading a plugin from a marketplace."""
        logger.info(f"Plugin marketplace not fully implemented. Would download from {url}")
        pass
