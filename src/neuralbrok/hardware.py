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
    {"id": "dgx-spark-gb-10", "name": "DGX Spark (GB 10)", "manufacturer": "NVIDIA", "vram_gb": 128, "bandwidth_gbps": 273},
    {"id": "rtx-5090", "name": "GeForce RTX 5090", "manufacturer": "NVIDIA", "vram_gb": 32, "bandwidth_gbps": 1792},
    {"id": "rtx-5080", "name": "GeForce RTX 5080", "manufacturer": "NVIDIA", "vram_gb": 16, "bandwidth_gbps": 896},
    {"id": "rtx-5070-ti", "name": "GeForce RTX 5070 Ti", "manufacturer": "NVIDIA", "vram_gb": 16, "bandwidth_gbps": 896},
    {"id": "rtx-5070", "name": "GeForce RTX 5070", "manufacturer": "NVIDIA", "vram_gb": 12, "bandwidth_gbps": 672},
    {"id": "rtx-5060-ti-16gb", "name": "GeForce RTX 5060 Ti 16GB", "manufacturer": "NVIDIA", "vram_gb": 16, "bandwidth_gbps": 448},
    {"id": "rtx-5060-ti", "name": "GeForce RTX 5060 Ti", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 448},
    {"id": "rtx-5060", "name": "GeForce RTX 5060", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 448},
    {"id": "rtx-5050", "name": "GeForce RTX 5050", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 320},
    {"id": "rtx-4090", "name": "GeForce RTX 4090", "manufacturer": "NVIDIA", "vram_gb": 24, "bandwidth_gbps": 1008},
    {"id": "rtx-4080-super", "name": "GeForce RTX 4080 SUPER", "manufacturer": "NVIDIA", "vram_gb": 16, "bandwidth_gbps": 736},
    {"id": "rtx-4080", "name": "GeForce RTX 4080", "manufacturer": "NVIDIA", "vram_gb": 16, "bandwidth_gbps": 717},
    {"id": "rtx-4070-ti-super", "name": "GeForce RTX 4070 Ti SUPER", "manufacturer": "NVIDIA", "vram_gb": 16, "bandwidth_gbps": 672},
    {"id": "rtx-4070-ti", "name": "GeForce RTX 4070 Ti", "manufacturer": "NVIDIA", "vram_gb": 12, "bandwidth_gbps": 504},
    {"id": "rtx-4070-super", "name": "GeForce RTX 4070 SUPER", "manufacturer": "NVIDIA", "vram_gb": 12, "bandwidth_gbps": 504},
    {"id": "rtx-4070", "name": "GeForce RTX 4070", "manufacturer": "NVIDIA", "vram_gb": 12, "bandwidth_gbps": 504},
    {"id": "rtx-4060-ti-16gb", "name": "GeForce RTX 4060 Ti 16GB", "manufacturer": "NVIDIA", "vram_gb": 16, "bandwidth_gbps": 288},
    {"id": "rtx-4060-ti", "name": "GeForce RTX 4060 Ti", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 288},
    {"id": "rtx-4060", "name": "GeForce RTX 4060", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 272},
    {"id": "rtx-3090-ti", "name": "GeForce RTX 3090 Ti", "manufacturer": "NVIDIA", "vram_gb": 24, "bandwidth_gbps": 1008},
    {"id": "rtx-3090", "name": "GeForce RTX 3090", "manufacturer": "NVIDIA", "vram_gb": 24, "bandwidth_gbps": 936},
    {"id": "rtx-3080-ti", "name": "GeForce RTX 3080 Ti", "manufacturer": "NVIDIA", "vram_gb": 12, "bandwidth_gbps": 912},
    {"id": "rtx-3080-12gb", "name": "GeForce RTX 3080 12GB", "manufacturer": "NVIDIA", "vram_gb": 12, "bandwidth_gbps": 912},
    {"id": "rtx-3080", "name": "GeForce RTX 3080", "manufacturer": "NVIDIA", "vram_gb": 10, "bandwidth_gbps": 760},
    {"id": "rtx-3070-ti", "name": "GeForce RTX 3070 Ti", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 608},
    {"id": "rtx-3070", "name": "GeForce RTX 3070", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 448},
    {"id": "rtx-3060-ti-gddr6x", "name": "GeForce RTX 3060 Ti GDDR6X", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 608},
    {"id": "rtx-3060-ti", "name": "GeForce RTX 3060 Ti", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 448},
    {"id": "rtx-3060", "name": "GeForce RTX 3060", "manufacturer": "NVIDIA", "vram_gb": 12, "bandwidth_gbps": 360},
    {"id": "rtx-3050", "name": "GeForce RTX 3050", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 224},
    {"id": "rtx-3050-6gb", "name": "GeForce RTX 3050 6GB", "manufacturer": "NVIDIA", "vram_gb": 6, "bandwidth_gbps": 168},
    {"id": "rtx-2080-ti", "name": "GeForce RTX 2080 Ti", "manufacturer": "NVIDIA", "vram_gb": 11, "bandwidth_gbps": 616},
    {"id": "rtx-2080-super", "name": "GeForce RTX 2080 SUPER", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 496},
    {"id": "rtx-2080", "name": "GeForce RTX 2080", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 448},
    {"id": "rtx-2070-super", "name": "GeForce RTX 2070 SUPER", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 448},
    {"id": "rtx-2070", "name": "GeForce RTX 2070", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 448},
    {"id": "rtx-2060-super", "name": "GeForce RTX 2060 SUPER", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 448},
    {"id": "rtx-2060-12gb", "name": "GeForce RTX 2060 12GB", "manufacturer": "NVIDIA", "vram_gb": 12, "bandwidth_gbps": 336},
    {"id": "rtx-2060", "name": "GeForce RTX 2060", "manufacturer": "NVIDIA", "vram_gb": 6, "bandwidth_gbps": 336},
    {"id": "gtx-1660-super", "name": "GeForce GTX 1660 SUPER", "manufacturer": "NVIDIA", "vram_gb": 6, "bandwidth_gbps": 336},
    {"id": "gtx-1660-ti", "name": "GeForce GTX 1660 Ti", "manufacturer": "NVIDIA", "vram_gb": 6, "bandwidth_gbps": 288},
    {"id": "gtx-1660", "name": "GeForce GTX 1660", "manufacturer": "NVIDIA", "vram_gb": 6, "bandwidth_gbps": 192},
    {"id": "gtx-1080-ti", "name": "GeForce GTX 1080 Ti", "manufacturer": "NVIDIA", "vram_gb": 11, "bandwidth_gbps": 484},
    {"id": "gtx-1080", "name": "GeForce GTX 1080", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 320},
    {"id": "gtx-1070-ti", "name": "GeForce GTX 1070 Ti", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 256},
    {"id": "gtx-1070", "name": "GeForce GTX 1070", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 256},
    {"id": "gtx-1060", "name": "GeForce GTX 1060 6GB", "manufacturer": "NVIDIA", "vram_gb": 6, "bandwidth_gbps": 192},
    {"id": "rtx-pro-6000", "name": "RTX PRO 6000 Blackwell", "manufacturer": "NVIDIA", "vram_gb": 96, "bandwidth_gbps": 1792},
    {"id": "rtx-pro-5000-72gb", "name": "RTX PRO 5000 Blackwell 72GB", "manufacturer": "NVIDIA", "vram_gb": 72, "bandwidth_gbps": 1344},
    {"id": "rtx-pro-5000", "name": "RTX PRO 5000 Blackwell", "manufacturer": "NVIDIA", "vram_gb": 48, "bandwidth_gbps": 1344},
    {"id": "rtx-pro-4000", "name": "RTX PRO 4000 Blackwell", "manufacturer": "NVIDIA", "vram_gb": 24, "bandwidth_gbps": 432},
    {"id": "rtx-6000-ada", "name": "RTX 6000 Ada", "manufacturer": "NVIDIA", "vram_gb": 48, "bandwidth_gbps": 960},
    {"id": "rtx-5000-ada", "name": "RTX 5000 Ada", "manufacturer": "NVIDIA", "vram_gb": 32, "bandwidth_gbps": 576},
    {"id": "rtx-4500-ada", "name": "RTX 4500 Ada", "manufacturer": "NVIDIA", "vram_gb": 24, "bandwidth_gbps": 432},
    {"id": "rtx-4000-ada", "name": "RTX 4000 Ada", "manufacturer": "NVIDIA", "vram_gb": 20, "bandwidth_gbps": 360},
    {"id": "rtx-2000-ada", "name": "RTX 2000 Ada", "manufacturer": "NVIDIA", "vram_gb": 16, "bandwidth_gbps": 224},
    {"id": "rtx-a6000", "name": "RTX A6000", "manufacturer": "NVIDIA", "vram_gb": 48, "bandwidth_gbps": 768},
    {"id": "rtx-a5000", "name": "RTX A5000", "manufacturer": "NVIDIA", "vram_gb": 24, "bandwidth_gbps": 768},
    {"id": "rtx-a4000", "name": "RTX A4000", "manufacturer": "NVIDIA", "vram_gb": 16, "bandwidth_gbps": 448},
    {"id": "rtx-a2000-12gb", "name": "RTX A2000 12GB", "manufacturer": "NVIDIA", "vram_gb": 12, "bandwidth_gbps": 288},
    {"id": "rtx-a2000", "name": "RTX A2000", "manufacturer": "NVIDIA", "vram_gb": 6, "bandwidth_gbps": 288},
    {"id": "t1000-8gb", "name": "T1000 8GB", "manufacturer": "NVIDIA", "vram_gb": 8, "bandwidth_gbps": 160},
    {"id": "t1000", "name": "T1000", "manufacturer": "NVIDIA", "vram_gb": 4, "bandwidth_gbps": 160},
    {"id": "t600", "name": "T600", "manufacturer": "NVIDIA", "vram_gb": 4, "bandwidth_gbps": 160},
    {"id": "t400-4gb", "name": "T400 4GB", "manufacturer": "NVIDIA", "vram_gb": 4, "bandwidth_gbps": 80},
    {"id": "t400", "name": "T400", "manufacturer": "NVIDIA", "vram_gb": 2, "bandwidth_gbps": 80},
    {"id": "l40s", "name": "L40S", "manufacturer": "NVIDIA", "vram_gb": 48, "bandwidth_gbps": 864},
    {"id": "tesla-p40", "name": "Tesla P40", "manufacturer": "NVIDIA", "vram_gb": 24, "bandwidth_gbps": 346},
    {"id": "rx-9070-xt", "name": "Radeon RX 9070 XT", "manufacturer": "AMD", "vram_gb": 16, "bandwidth_gbps": 640},
    {"id": "rx-9070", "name": "Radeon RX 9070", "manufacturer": "AMD", "vram_gb": 16, "bandwidth_gbps": 640},
    {"id": "rx-9070-gre", "name": "Radeon RX 9070 GRE", "manufacturer": "AMD", "vram_gb": 16, "bandwidth_gbps": 576},
    {"id": "rx-9060-xt-16gb", "name": "Radeon RX 9060 XT 16GB", "manufacturer": "AMD", "vram_gb": 16, "bandwidth_gbps": 320},
    {"id": "rx-9060-xt", "name": "Radeon RX 9060 XT", "manufacturer": "AMD", "vram_gb": 8, "bandwidth_gbps": 320},
    {"id": "rx-9060", "name": "Radeon RX 9060", "manufacturer": "AMD", "vram_gb": 8, "bandwidth_gbps": 288},
    {"id": "rx-7900-xtx", "name": "Radeon RX 7900 XTX", "manufacturer": "AMD", "vram_gb": 24, "bandwidth_gbps": 960},
    {"id": "rx-7900-xt", "name": "Radeon RX 7900 XT", "manufacturer": "AMD", "vram_gb": 20, "bandwidth_gbps": 800},
    {"id": "rx-7900-gre", "name": "Radeon RX 7900 GRE", "manufacturer": "AMD", "vram_gb": 16, "bandwidth_gbps": 576},
    {"id": "rx-7800-xt", "name": "Radeon RX 7800 XT", "manufacturer": "AMD", "vram_gb": 16, "bandwidth_gbps": 624},
    {"id": "rx-7700-xt", "name": "Radeon RX 7700 XT", "manufacturer": "AMD", "vram_gb": 12, "bandwidth_gbps": 432},
    {"id": "rx-7600-xt", "name": "Radeon RX 7600 XT", "manufacturer": "AMD", "vram_gb": 16, "bandwidth_gbps": 288},
    {"id": "rx-7600", "name": "Radeon RX 7600", "manufacturer": "AMD", "vram_gb": 8, "bandwidth_gbps": 288},
    {"id": "rx-6950-xt", "name": "Radeon RX 6950 XT", "manufacturer": "AMD", "vram_gb": 16, "bandwidth_gbps": 576},
    {"id": "rx-6900-xt", "name": "Radeon RX 6900 XT", "manufacturer": "AMD", "vram_gb": 16, "bandwidth_gbps": 512},
    {"id": "rx-6800-xt", "name": "Radeon RX 6800 XT", "manufacturer": "AMD", "vram_gb": 16, "bandwidth_gbps": 512},
    {"id": "rx-6800", "name": "Radeon RX 6800", "manufacturer": "AMD", "vram_gb": 16, "bandwidth_gbps": 512},
    {"id": "rx-6750-xt", "name": "Radeon RX 6750 XT", "manufacturer": "AMD", "vram_gb": 12, "bandwidth_gbps": 432},
    {"id": "rx-6700-xt", "name": "Radeon RX 6700 XT", "manufacturer": "AMD", "vram_gb": 12, "bandwidth_gbps": 384},
    {"id": "rx-6650-xt", "name": "Radeon RX 6650 XT", "manufacturer": "AMD", "vram_gb": 8, "bandwidth_gbps": 280},
    {"id": "rx-6600-xt", "name": "Radeon RX 6600 XT", "manufacturer": "AMD", "vram_gb": 8, "bandwidth_gbps": 256},
    {"id": "rx-6600", "name": "Radeon RX 6600", "manufacturer": "AMD", "vram_gb": 8, "bandwidth_gbps": 224},
    {"id": "rx-5700-xt", "name": "Radeon RX 5700 XT", "manufacturer": "AMD", "vram_gb": 8, "bandwidth_gbps": 448},
    {"id": "rx-5700", "name": "Radeon RX 5700", "manufacturer": "AMD", "vram_gb": 8, "bandwidth_gbps": 448},
    {"id": "rx-5600-xt", "name": "Radeon RX 5600 XT", "manufacturer": "AMD", "vram_gb": 6, "bandwidth_gbps": 288},
    {"id": "ryzen-ai-max", "name": "Ryzen AI Max (Strix Halo)", "manufacturer": "AMD", "vram_options": [{"vram_gb": 32, "bandwidth_gbps": 256}, {"vram_gb": 64, "bandwidth_gbps": 256}, {"vram_gb": 96, "bandwidth_gbps": 256}, {"vram_gb": 128, "bandwidth_gbps": 256}]},
    {"id": "arc-b580", "name": "Arc B580", "manufacturer": "Intel", "vram_gb": 12, "bandwidth_gbps": 456},
    {"id": "arc-b570", "name": "Arc B570", "manufacturer": "Intel", "vram_gb": 10, "bandwidth_gbps": 380},
    {"id": "arc-a770", "name": "Arc A770 16GB", "manufacturer": "Intel", "vram_gb": 16, "bandwidth_gbps": 560},
    {"id": "arc-a770-8gb", "name": "Arc A770 8GB", "manufacturer": "Intel", "vram_gb": 8, "bandwidth_gbps": 512},
    {"id": "arc-a750", "name": "Arc A750", "manufacturer": "Intel", "vram_gb": 8, "bandwidth_gbps": 512},
    {"id": "arc-a580", "name": "Arc A580", "manufacturer": "Intel", "vram_gb": 8, "bandwidth_gbps": 512},
    {"id": "m4-max", "name": "M4 Max", "manufacturer": "Apple", "vram_options": [{"vram_gb": 36, "bandwidth_gbps": 410}, {"vram_gb": 48, "bandwidth_gbps": 546}, {"vram_gb": 64, "bandwidth_gbps": 546}, {"vram_gb": 128, "bandwidth_gbps": 546}]},
    {"id": "m4-pro", "name": "M4 Pro", "manufacturer": "Apple", "vram_options": [{"vram_gb": 24, "bandwidth_gbps": 273}, {"vram_gb": 48, "bandwidth_gbps": 273}]},
    {"id": "m4", "name": "M4", "manufacturer": "Apple", "vram_options": [{"vram_gb": 16, "bandwidth_gbps": 120}, {"vram_gb": 24, "bandwidth_gbps": 120}, {"vram_gb": 32, "bandwidth_gbps": 120}]},
    {"id": "m3-ultra", "name": "M3 Ultra", "manufacturer": "Apple", "vram_options": [{"vram_gb": 192, "bandwidth_gbps": 800}]},
    {"id": "m3-max", "name": "M3 Max", "manufacturer": "Apple", "vram_options": [{"vram_gb": 36, "bandwidth_gbps": 300}, {"vram_gb": 48, "bandwidth_gbps": 400}, {"vram_gb": 64, "bandwidth_gbps": 400}, {"vram_gb": 128, "bandwidth_gbps": 400}]},
    {"id": "m3-pro", "name": "M3 Pro", "manufacturer": "Apple", "vram_options": [{"vram_gb": 18, "bandwidth_gbps": 150}, {"vram_gb": 36, "bandwidth_gbps": 150}]},
    {"id": "m3", "name": "M3", "manufacturer": "Apple", "vram_options": [{"vram_gb": 8, "bandwidth_gbps": 100}, {"vram_gb": 16, "bandwidth_gbps": 100}, {"vram_gb": 24, "bandwidth_gbps": 100}]},
    {"id": "m2-ultra", "name": "M2 Ultra", "manufacturer": "Apple", "vram_options": [{"vram_gb": 64, "bandwidth_gbps": 800}, {"vram_gb": 128, "bandwidth_gbps": 800}, {"vram_gb": 192, "bandwidth_gbps": 800}]},
    {"id": "m2-max", "name": "M2 Max", "manufacturer": "Apple", "vram_options": [{"vram_gb": 32, "bandwidth_gbps": 400}, {"vram_gb": 64, "bandwidth_gbps": 400}, {"vram_gb": 96, "bandwidth_gbps": 400}]},
    {"id": "m2-pro", "name": "M2 Pro", "manufacturer": "Apple", "vram_options": [{"vram_gb": 16, "bandwidth_gbps": 200}, {"vram_gb": 32, "bandwidth_gbps": 200}]},
    {"id": "m2", "name": "M2", "manufacturer": "Apple", "vram_options": [{"vram_gb": 8, "bandwidth_gbps": 100}, {"vram_gb": 16, "bandwidth_gbps": 100}, {"vram_gb": 24, "bandwidth_gbps": 100}]},
    {"id": "m1-ultra", "name": "M1 Ultra", "manufacturer": "Apple", "vram_options": [{"vram_gb": 64, "bandwidth_gbps": 800}, {"vram_gb": 128, "bandwidth_gbps": 800}]},
    {"id": "m1-max", "name": "M1 Max", "manufacturer": "Apple", "vram_options": [{"vram_gb": 32, "bandwidth_gbps": 400}, {"vram_gb": 64, "bandwidth_gbps": 400}]},
    {"id": "m1-pro", "name": "M1 Pro", "manufacturer": "Apple", "vram_options": [{"vram_gb": 16, "bandwidth_gbps": 200}, {"vram_gb": 32, "bandwidth_gbps": 200}]},
    {"id": "m1", "name": "M1", "manufacturer": "Apple", "vram_options": [{"vram_gb": 8, "bandwidth_gbps": 68}, {"vram_gb": 16, "bandwidth_gbps": 68}]},
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
