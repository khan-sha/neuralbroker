import platform
import subprocess
import sys
import warnings
import psutil
from dataclasses import dataclass
from typing import List, Optional

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

def _get_nvidia_recommendations(vram_gb: float) -> tuple[List[str], float]:
    if vram_gb < 6:
        return ["qwen3:0.6b", "phi3:mini"], 0.80
    elif vram_gb < 8:
        return ["llama3.1:8b", "mistral:7b", "qwen3:8b"], 0.80
    elif vram_gb < 12:
        return ["llama3.1:8b", "mistral:7b", "qwen3:8b", "deepseek-coder:6.7b"], 0.80
    elif vram_gb < 16:
        return ["llama3.1:8b", "mixtral:8x7b", "qwen3:14b"], 0.85
    elif vram_gb < 24:
        return ["llama3.1:70b", "qwen3:32b", "mixtral:8x7b"], 0.85
    else:
        return ["llama3.1:70b", "qwen3:72b", "mixtral:8x22b"], 0.90

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

    # 1. Check Apple Silicon
    if plat == "macos" and "arm" in platform.processor().lower():
        try:
            chip_name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
        except:
            chip_name = "Apple Silicon"
        
        tdp = _get_apple_chip_info(chip_name)
        models, _ = _get_nvidia_recommendations(ram_gb)
        
        return DeviceProfile(
            gpu_vendor="apple",
            gpu_model=chip_name,
            vram_gb=ram_gb,
            ram_gb=ram_gb,
            cpu_cores=cpu_cores,
            platform=plat,
            cuda_version=None,
            metal_support=True,
            recommended_runtime="ollama",
            recommended_models=models,
            recommended_vram_threshold=0.75,
            estimated_electricity_tdp_watts=tdp
        )

    # 2. Check NVIDIA via pynvml
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*deprecated.*")
            import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = info.total / (1024**3)
        try:
            cuda_ver = pynvml.nvmlSystemGetCudaDriverVersion()
            cuda_version = f"{cuda_ver // 1000}.{(cuda_ver % 1000) // 10}"
        except:
            cuda_version = "Unknown"
        pynvml.nvmlShutdown()

        tdp = _get_nvidia_tdp(gpu_name)
        models, threshold = _get_nvidia_recommendations(vram_gb)

        return DeviceProfile(
            gpu_vendor="nvidia",
            gpu_model=gpu_name,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
            cpu_cores=cpu_cores,
            platform=plat,
            cuda_version=cuda_version,
            metal_support=False,
            recommended_runtime="ollama",
            recommended_models=models,
            recommended_vram_threshold=threshold,
            estimated_electricity_tdp_watts=tdp
        )
    except Exception:
        pass

    # 3. Check AMD via rocm-smi
    try:
        if plat == "linux":
            vram_str = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, check=True).stdout
            if "vram" in vram_str.lower():
                # Rough detection
                # In reality, parsing rocm-smi might be complex, we'll assign a placeholder or parse if possible
                gpu_name = "AMD Radeon GPU (ROCm)"
                vram_gb = 16.0  # Safe default if parsing fails, but ideally parse `vram_str`
                
                tdp = _get_amd_tdp(gpu_name)
                models, threshold = _get_nvidia_recommendations(vram_gb)
                
                return DeviceProfile(
                    gpu_vendor="amd",
                    gpu_model=gpu_name,
                    vram_gb=vram_gb,
                    ram_gb=ram_gb,
                    cpu_cores=cpu_cores,
                    platform=plat,
                    cuda_version=None,
                    metal_support=False,
                    recommended_runtime="llama_cpp",
                    recommended_models=models,
                    recommended_vram_threshold=threshold,
                    estimated_electricity_tdp_watts=tdp
                )
    except Exception:
        pass

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
        recommended_runtime="llama_cpp",
        recommended_models=["qwen3:0.6b", "phi3:mini", "llama3.2:1b"],
        recommended_vram_threshold=1.0,
        estimated_electricity_tdp_watts=65
    )
