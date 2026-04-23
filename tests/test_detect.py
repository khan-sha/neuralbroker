import sys
import pytest
from unittest.mock import patch, MagicMock

from neuralbrok.detect import detect_device, DeviceProfile
from neuralbrok.autoconfig import generate_config

@pytest.fixture(autouse=True)
def mock_psutil():
    with patch("neuralbrok.detect.psutil") as mock:
        mock.virtual_memory.return_value.total = 32 * (1024**3)
        mock.cpu_count.return_value = 12
        yield mock

def test_nvidia_detection():
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA GeForce RTX 4090"
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value.total = 24 * (1024**3)
    mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12040

    with patch("sys.platform", "linux"):
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            profile = detect_device()
            
            assert profile.gpu_vendor == "nvidia"
            assert "4090" in profile.gpu_model
            assert profile.vram_gb == 24.0
            assert profile.platform == "linux"
            assert profile.cuda_version == "12.4"
            assert profile.recommended_runtime == "ollama"
            assert profile.recommended_vram_threshold == 0.90
            assert profile.estimated_electricity_tdp_watts == 450
            assert "llama3.1:70b" in profile.recommended_models

def test_apple_silicon_detection():
    with patch("sys.platform", "darwin"), \
         patch("neuralbrok.detect.platform.processor", return_value="arm"):
        with patch("neuralbrok.detect.subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.stdout = "Apple M3 Max\n"
            mock_run.return_value = mock_proc
            
            profile = detect_device()
            
            assert profile.gpu_vendor == "apple"
            assert profile.gpu_model == "Apple M3 Max"
            assert profile.platform == "macos"
            assert profile.metal_support is True
            assert profile.recommended_runtime == "ollama"
            assert profile.estimated_electricity_tdp_watts == 75  # 30 * 2.5
            
def test_amd_detection_with_rocm():
    with patch("sys.platform", "linux"):
        # We need to simulate pynvml not existing so it falls back to AMD/CPU
        # pynvml is wrapped in try/except in detect_device, but doing import pynvml
        # We can patch sys.modules to simulate missing pynvml
        with patch.dict("sys.modules", {"pynvml": None}):
            with patch("neuralbrok.detect.subprocess.run") as mock_run:
                mock_proc = MagicMock()
                mock_proc.stdout = "VRAM: 16384"
                mock_run.return_value = mock_proc
                
                profile = detect_device()
                
                assert profile.gpu_vendor == "amd"
                assert profile.recommended_runtime == "llama_cpp"
            
def test_amd_detection_without_rocm_cpu_fallback():
    with patch("sys.platform", "linux"):
        with patch.dict("sys.modules", {"pynvml": None}):
            with patch("neuralbrok.detect.subprocess.run", side_effect=Exception):
                profile = detect_device()
                
                assert profile.gpu_vendor == "none"
                assert profile.gpu_model == "CPU Only"
                assert profile.recommended_runtime == "llama_cpp"
                assert profile.vram_gb == 0.0
                assert profile.recommended_vram_threshold == 1.0

def test_config_generation():
    profile = DeviceProfile(
        gpu_vendor="nvidia",
        gpu_model="RTX 4090",
        vram_gb=24.0,
        ram_gb=32.0,
        cpu_cores=12,
        platform="linux",
        cuda_version="12.4",
        metal_support=False,
        recommended_runtime="ollama",
        recommended_models=["llama3.1:70b"],
        recommended_vram_threshold=0.90,
        estimated_electricity_tdp_watts=450
    )
    
    config_yaml = generate_config(profile)
    
    assert "RTX 4090" in config_yaml
    assert "llama3.1:70b" in config_yaml
    assert "vram_threshold: 0.9" in config_yaml
    assert "gpu_tdp_watts: 450" in config_yaml
    assert "runtime: ollama" in config_yaml
