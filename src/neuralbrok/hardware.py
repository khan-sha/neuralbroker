import json
from dataclasses import dataclass
from typing import Optional, List

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
