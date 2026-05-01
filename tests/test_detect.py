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
    with patch("sys.platform", "linux"):
        with patch("neuralbrok.hardware.HardwareTelemetry.initialize", return_value="nvidia"):
            with patch("neuralbrok.hardware.HardwareTelemetry.get_vram_snapshot", return_value={"used": 4.0, "free": 20.0}):
                with patch("neuralbrok.hardware.HardwareTelemetry.shutdown"):
                    with patch("neuralbrok.hardware.lookup_gpu") as mock_lookup:
                        mock_spec = MagicMock()
                        mock_spec.bandwidth_gbps = 1000.0
                        mock_lookup.return_value = mock_spec
                        
                        profile = detect_device()
                        
                        assert profile.gpu_vendor == "nvidia"
                        assert profile.vram_gb == 24.0
                        assert profile.platform == "linux"
                        assert profile.recommended_runtime == "ollama"

def test_apple_silicon_detection():
    with patch("sys.platform", "darwin"), \
         patch("neuralbrok.detect.platform.processor", return_value="arm"):
        with patch("neuralbrok.detect._get_cpu_name", return_value="Apple M3 Max"):
            profile = detect_device()
            
            assert profile.gpu_vendor == "apple"
            assert profile.gpu_model == "Apple M3 Max"
            assert profile.platform == "macos"
            assert profile.metal_support is True
            assert profile.recommended_runtime == "ollama"
            
def test_amd_detection_with_rocm():
    with patch("sys.platform", "linux"):
        with patch("neuralbrok.hardware.HardwareTelemetry.initialize", return_value="amd"):
            with patch("neuralbrok.hardware.HardwareTelemetry.get_vram_snapshot", return_value={"used": 2.0, "free": 14.0}):
                profile = detect_device()
                
                assert profile.gpu_vendor == "amd"
                assert profile.recommended_runtime == "ollama"
            
def test_amd_detection_without_rocm_cpu_fallback():
    with patch("sys.platform", "linux"):
        with patch("neuralbrok.hardware.HardwareTelemetry.initialize", return_value="none"):
            profile = detect_device()
            
            assert profile.gpu_vendor == "none"
            assert "CPU" in profile.gpu_model
            assert profile.recommended_runtime == "ollama"
