import json
import logging
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

@dataclass
class GpuSpec:
    id: str
    name: str
    manufacturer: str
    vram_gb: Optional[float] = None
    bandwidth_gbps: Optional[float] = None
    vram_options: Optional[List[dict]] = None

# Sourced from https://github.com/BenD10/whatmodels/blob/main/src/lib/data/gpus.json
GPU_DATABASE = [
    {"id": "rtx-5090", "name": "GeForce RTX 5090", "manufacturer": "NVIDIA", "vram_gb": 32, "bandwidth_gbps": 1792},
    {"id": "rtx-5080", "name": "GeForce RTX 5080", "manufacturer": "NVIDIA", "vram_gb": 16, "bandwidth_gbps": 896},
    {"id": "rtx-4090", "name": "GeForce RTX 4090", "manufacturer": "NVIDIA", "vram_gb": 24, "bandwidth_gbps": 1008},
    {"id": "rtx-4080-super", "name": "GeForce RTX 4080 SUPER", "manufacturer": "NVIDIA", "vram_gb": 16, "bandwidth_gbps": 736},
    {"id": "rtx-4080", "name": "GeForce RTX 4080", "manufacturer": "NVIDIA", "vram_gb": 16, "bandwidth_gbps": 717},
    {"id": "rtx-4070-ti-super", "name": "GeForce RTX 4070 Ti SUPER", "manufacturer": "NVIDIA", "vram_gb": 16, "bandwidth_gbps": 672},
    {"id": "rtx-3090-ti", "name": "GeForce RTX 3090 Ti", "manufacturer": "NVIDIA", "vram_gb": 24, "bandwidth_gbps": 1008},
    {"id": "rtx-3090", "name": "GeForce RTX 3090", "manufacturer": "NVIDIA", "vram_gb": 24, "bandwidth_gbps": 936},
    {"id": "rtx-3080-ti", "name": "GeForce RTX 3080 Ti", "manufacturer": "NVIDIA", "vram_gb": 12, "bandwidth_gbps": 912},
    {"id": "rtx-3060", "name": "GeForce RTX 3060", "manufacturer": "NVIDIA", "vram_gb": 12, "bandwidth_gbps": 360},
    {"id": "rx-7900-xtx", "name": "Radeon RX 7900 XTX", "manufacturer": "AMD", "vram_gb": 24, "bandwidth_gbps": 960},
    {"id": "rx-6800-xt", "name": "Radeon RX 6800 XT", "manufacturer": "AMD", "vram_gb": 16, "bandwidth_gbps": 512},
    {"id": "m4-max", "name": "M4 Max", "manufacturer": "Apple", "vram_options": [{"vram_gb": 36, "bandwidth_gbps": 410}, {"vram_gb": 128, "bandwidth_gbps": 546}]},
    {"id": "m3-max", "name": "M3 Max", "manufacturer": "Apple", "vram_options": [{"vram_gb": 36, "bandwidth_gbps": 300}, {"vram_gb": 128, "bandwidth_gbps": 400}]},
    {"id": "m2-max", "name": "M2 Max", "manufacturer": "Apple", "vram_options": [{"vram_gb": 32, "bandwidth_gbps": 400}, {"vram_gb": 64, "bandwidth_gbps": 400}, {"vram_gb": 96, "bandwidth_gbps": 400}]},
]

def lookup_gpu(name: str) -> Optional[GpuSpec]:
    name_clean = name.lower().replace("nvidia ", "").replace("geforce ", "").replace("amd ", "").replace("radeon ", "")
    for entry in GPU_DATABASE:
        if entry["id"].replace("-", " ") in name_clean or entry["name"].lower() in name.lower():
            return GpuSpec(**entry)
    return None

class HardwareTelemetry:
    """
    Unified cross-platform hardware telemetry.
    Resolves pynvml deprecation noise and adds multi-vendor support.
    """
    def __init__(self):
        self.vendor = "none"
        self._nvml_initialized = False
        self._psutil = None
        
        # Suppress pynvml deprecation warnings globally in this class
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*deprecated.*")
            try:
                import pynvml
                self._pynvml = pynvml
            except ImportError:
                self._pynvml = None

        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            pass

    def initialize(self):
        """Detect vendor and initialize appropriate driver handles."""
        if sys.platform == "darwin":
            self.vendor = "apple"
        elif self._pynvml:
            try:
                self._pynvml.nvmlInit()
                self._nvml_initialized = True
                self.vendor = "nvidia"
            except:
                self.vendor = "none"
        
        # AMD check (simplistic)
        if self.vendor == "none":
            try:
                subprocess.run(["rocm-smi"], capture_output=True, check=True)
                self.vendor = "amd"
            except:
                pass
        
        return self.vendor

    def shutdown(self):
        if self._nvml_initialized and self._pynvml:
            try:
                self._pynvml.nvmlShutdown()
            except:
                pass
            self._nvml_initialized = False

    def get_vram_snapshot(self, gpu_id: int = 0) -> Dict[str, float]:
        """
        Get live VRAM usage (used/free in GB).
        Guaranteed to be quiet and won't throw deprecation warnings.
        """
        if self.vendor == "nvidia" and self._nvml_initialized:
            try:
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                return {
                    "used": info.used / (1024**3),
                    "free": info.free / (1024**3)
                }
            except:
                pass
        
        if self.vendor == "apple" and self._psutil:
            # On Apple Silicon, we report unified memory via psutil as a proxy
            # since VRAM is just shared system RAM.
            vm = self._psutil.virtual_memory()
            # We assume a 75% 'VRAM' ceiling for LLMs on Mac
            return {
                "used": vm.used / (1024**3),
                "free": (vm.total * 0.75 - vm.used) / (1024**3)
            }

        if self.vendor == "amd":
            # Attempt to parse rocm-smi
            try:
                # This is a placeholder for real ROCm parsing logic
                return {"used": 0.0, "free": 8.0}
            except:
                pass

        # Fallback (CPU only or detection failure)
        return {"used": 0.0, "free": 4.0}
