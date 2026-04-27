"""
sync.py — Periodic synchronization of local model catalog with Ollama library.
Hardens the 'Zero-Config' experience by ensuring the model registry is always fresh.
"""
import logging
import asyncio
from typing import List, Dict

from neuralbrok.ollama_catalog import (
    fetch_latest_ollama_models, 
    get_trending_ollama_models,
    OllamaModelEntry
)
from neuralbrok.models import build_model_catalog, ModelProfile
from neuralbrok.detect import detect_device

logger = logging.getLogger(__name__)

class RegistrySync:
    def __init__(self):
        self.last_sync = None
        self.trending_cache = []
        self.catalog_cache = []

    async def run_sync(self, profile=None):
        """
        Perform a full synchronization.
        1. Scrape trending models.
        2. Fetch API catalog.
        3. Match against local hardware.
        """
        if not profile:
            profile = detect_device()
        
        logger.info("Starting model registry sync...")
        
        # 1. Scrape trending (user requested) — run in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        self.trending_cache = await loop.run_in_executor(None, get_trending_ollama_models)

        # 2. Fetch full catalog
        self.catalog_cache = await loop.run_in_executor(None, fetch_latest_ollama_models)
        
        # 3. Match against local (to see what's new/installed)
        local_catalog = await build_model_catalog(profile, show_progress=False)
        
        installed_tags = {m.ollama_tag for m in local_catalog if m.is_installed}
        
        new_discoveries = []
        for m in self.catalog_cache:
            if m.tag not in installed_tags:
                # Potential new model to suggest
                new_discoveries.append(m)
        
        logger.info(f"Sync complete. Discovered {len(new_discoveries)} new models on Ollama library.")
        return {
            "trending": self.trending_cache,
            "new_models": [m.__dict__ for m in new_discoveries[:10]],
            "timestamp": asyncio.get_event_loop().time()
        }

async def start_periodic_sync(interval_hours=24):
    """Background task to keep the registry fresh."""
    sync = RegistrySync()
    while True:
        try:
            await sync.run_sync()
        except Exception as e:
            logger.error(f"Periodic sync failed: {e}")
        await asyncio.sleep(interval_hours * 3600)
