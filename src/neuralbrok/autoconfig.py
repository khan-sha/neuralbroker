import os
from neuralbrok.detect import DeviceProfile

def generate_config(profile: DeviceProfile) -> str:
    models_str = "\n".join([f"#   - {m}" for m in profile.recommended_models])
    
    config_yaml = f"""# NeuralBroker Auto-Generated Configuration
# Hardware Detected: {profile.gpu_model} ({profile.platform})
# Recommended models for your hardware:
{models_str}

local_nodes:
  - name: local
    runtime: {profile.recommended_runtime}
    host: localhost:11434
    vram_threshold: {profile.recommended_vram_threshold}

cloud_providers:
  - name: groq
    api_key_env: GROQ_KEY
    base_url: https://api.groq.com/openai/v1
  - name: together
    api_key_env: TOGETHER_KEY
  - name: openai
    api_key_env: OPENAI_KEY

routing:
  default_mode: cost
  electricity_kwh_price: 0.14
  gpu_tdp_watts: {profile.estimated_electricity_tdp_watts}
  vram_poll_interval_seconds: 0.5
"""
    return config_yaml

def write_initial_config(profile: DeviceProfile):
    config_dir = os.path.expanduser("~/.neuralbrok")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        config_str = generate_config(profile)
        with open(config_path, "w") as f:
            f.write(config_str)
