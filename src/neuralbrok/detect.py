import platform
import subprocess
import sys
import warnings
import psutil
from dataclasses import dataclass
from typing import List, Optional

from neuralbrok.hardware import lookup_gpu
from neuralbrok.models import resolve_model

@dataclass
class DeviceProfile:
    gpu_vendor: str  # nvidia, apple, amd, none
    gpu_model: str
    vram_gb: float
    ram_gb: float
    cpu_cores: int
    platform: str  # windows, macos, linux
    cuda_version: Optional[str]
    metal_support: bool
    recommended_runtime: str
    recommended_models: List[str]
    recommended_vram_threshold: float
    estimated_electricity_tdp_watts: int
    bandwidth_gbps: Optional[float] = None  # New field from whatmodels

def _get_nvidia_recommendations(vram_gb: float) -> tuple[List[str], float]:
    if vram_gb < 6:
        return [resolve_model("small"), resolve_model("phi-4-mini")], 0.80
    elif vram_gb < 8:
        return [resolve_model("default"), resolve_model("coding"), resolve_model("reasoning")], 0.80
    elif vram_gb < 12:
        return [resolve_model("default"), resolve_model("coding"), resolve_model("reasoning"), "deepseek-coder:6.7b"], 0.80
    elif vram_gb < 16:
        return [resolve_model("default"), resolve_model("coding_large"), resolve_model("qwen3:14b")], 0.85
    elif vram_gb < 24:
        return [resolve_model("reasoning_large"), resolve_model("qwen3:32b")], 0.85
    else:
        return [resolve_model("reasoning_large"), "llama3.1:70b", "qwen3:72b"], 0.90

def _get_nvidia_tdp(model_name: str) -> int:
    name = model_name.upper()
    if "4090" in name: return 450
    if "4080" in name: return 320
    if "4070" in name: return 200
    if "4060" in name: return 115
    if "3090" in name: return 350
    if "3080" in name: return 320
    if "3070" in name: return 220
    if "3060" in name: return 170
    return 250  # safe default

def _get_amd_tdp(model_name: str) -> int:
    name = model_name.upper()
    if "7900 XTX" in name: return 355
    if "7800 XT" in name: return 263
    if "6800 XT" in name: return 300
    return 250

def _get_apple_chip_info(chip_name: str) -> tuple[int, int]:
    # Returns (TDP, memory_multiplier placeholder but we use actual memory)
    name = chip_name.upper()
    base_tdp = 20
    multiplier = 1.0
    if "M4" in name: base_tdp = 35
    elif "M3" in name: base_tdp = 30
    elif "M2" in name: base_tdp = 25
    elif "M1" in name: base_tdp = 20
    
    if "ULTRA" in name: multiplier = 4.0
    elif "MAX" in name: multiplier = 2.5
    elif "PRO" in name: multiplier = 1.5
    
    return int(base_tdp * multiplier)

def detect_device() -> DeviceProfile:
    os_name = sys.platform
    plat = "linux"
    if os_name == "win32": plat = "windows"
    elif os_name == "darwin": plat = "macos"

    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024**3)
    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 4

    # 1. Check Apple Silicon (High Priority)
    if plat == "macos" and "arm" in platform.processor().lower():
        try:
            chip_name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
        except:
            chip_name = "Apple Silicon"
        
        tdp = _get_apple_chip_info(chip_name)
        models, _ = _get_nvidia_recommendations(ram_gb) # Unified recommendations
        
        gpu_spec = lookup_gpu(chip_name)
        bandwidth = None
        if gpu_spec and gpu_spec.vram_options:
            for opt in gpu_spec.vram_options:
                if abs(opt["vram_gb"] - ram_gb) < 4:
                    bandwidth = opt["bandwidth_gbps"]
                    break
        
        return DeviceProfile(
            gpu_vendor="apple", gpu_model=chip_name, vram_gb=ram_gb, ram_gb=ram_gb,
            cpu_cores=cpu_cores, platform=plat, cuda_version=None,
            metal_support=True, recommended_runtime="ollama",
            recommended_models=models, recommended_vram_threshold=0.75,
            estimated_electricity_tdp_watts=tdp, bandwidth_gbps=bandwidth
        )

    # ── Unified Telemetry Detection ──────────────────────────────────
    from neuralbrok.hardware import HardwareTelemetry, lookup_gpu
    telemetry = HardwareTelemetry()
    vendor = telemetry.initialize()
    
    if vendor == "nvidia":
        stats = telemetry.get_vram_snapshot(0)
        vram_gb = stats["used"] + stats["free"]
        
        gpu_name = "NVIDIA GPU"
        try:
            if hasattr(telemetry, "_pynvml") and telemetry._pynvml:
                handle = telemetry._pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = telemetry._pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes): gpu_name = gpu_name.decode("utf-8")
        except: pass

        try:
            cuda_ver = telemetry._pynvml.nvmlSystemGetCudaDriverVersion()
            cuda_version = f"{cuda_ver // 1000}.{(cuda_ver % 1000) // 10}"
        except:
            cuda_version = "Unknown"

        tdp = _get_nvidia_tdp(gpu_name)
        models, threshold = _get_nvidia_recommendations(vram_gb)
        gpu_spec = lookup_gpu(gpu_name)
        bandwidth = gpu_spec.bandwidth_gbps if gpu_spec else 360.0

        telemetry.shutdown()
        return DeviceProfile(
            gpu_vendor="nvidia", gpu_model=gpu_name, vram_gb=vram_gb, ram_gb=ram_gb,
            cpu_cores=cpu_cores, platform=plat, cuda_version=cuda_version,
            metal_support=False, recommended_runtime="ollama",
            recommended_models=models, recommended_vram_threshold=threshold,
            estimated_electricity_tdp_watts=tdp, bandwidth_gbps=bandwidth
        )
    
    elif vendor == "amd":
        stats = telemetry.get_vram_snapshot(0)
        vram_gb = stats["used"] + stats["free"]
        gpu_name = "AMD Radeon GPU"
        
        models, threshold = _get_nvidia_recommendations(vram_gb)
        
        telemetry.shutdown()
        return DeviceProfile(
            gpu_vendor="amd", gpu_model=gpu_name, vram_gb=vram_gb, ram_gb=ram_gb,
            cpu_cores=cpu_cores, platform=plat, cuda_version=None,
            metal_support=False, recommended_runtime="ollama",
            recommended_models=models, recommended_vram_threshold=threshold,
            estimated_electricity_tdp_watts=200, bandwidth_gbps=512.0
        )

    # 4. CPU only fallback
    return DeviceProfile(
        gpu_vendor="none",
        gpu_model="CPU Only",
        vram_gb=0.0,
        ram_gb=ram_gb,
        cpu_cores=cpu_cores,
        platform=plat,
        cuda_version=None,
        metal_support=False,
        recommended_runtime="ollama",
        recommended_models=[resolve_model("small"), resolve_model("phi-4-mini")],
        recommended_vram_threshold=1.0,
        estimated_electricity_tdp_watts=65,
        bandwidth_gbps=40.0
    )
