import argparse
import sys
import os
from neuralbrok.detect import detect_device
from neuralbrok.autoconfig import generate_config

def main():
    parser = argparse.ArgumentParser(description="NeuralBroker Device Setup")
    parser.parse_args()

    print("NeuralBroker · device detection")
    print("────────────────────────────────────────")
    
    profile = detect_device()
    
    # Format platform
    platform_info = profile.platform.capitalize()
    if profile.cuda_version:
        platform_info += f" · CUDA {profile.cuda_version}"
    elif profile.metal_support:
        platform_info += " · Metal"
    
    vram_str = f"{profile.vram_gb:.1f}GB VRAM" if profile.vram_gb > 0 else "System RAM"
    
    print(f"GPU          {profile.gpu_model} · {vram_str}")
    print(f"Platform     {platform_info}")
    print(f"Runtime      {profile.recommended_runtime} (recommended)")
    print(f"Threshold    {int(profile.recommended_vram_threshold * 100)}% VRAM before cloud spill")
    
    tdp = profile.estimated_electricity_tdp_watts
    cost_per_hr = (tdp / 1000) * 0.14
    print(f"TDP          {tdp}W · ${cost_per_hr:.3f}/hr at $0.14/kWh")
    
    print("\nRecommended models for your hardware:")
    for model in profile.recommended_models[:3]:
        if profile.recommended_runtime == "ollama":
            print(f"  → ollama pull {model}")
        else:
            print(f"  → download {model} for llama.cpp")
            
    print("")
    ans = input("Write config to ~/.neuralbrok/config.yaml? [y/n]: ").strip().lower()
    if ans == 'y':
        config_dir = os.path.expanduser("~/.neuralbrok")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "config.yaml")
        
        config_str = generate_config(profile)
        with open(config_path, "w") as f:
            f.write(config_str)
        print(f"✓ Wrote config to {config_path}")
        
        print("\nNext steps:")
        for model in profile.recommended_models[:3]:
            if profile.recommended_runtime == "ollama":
                print(f"  $ ollama pull {model}")
                
        print("\nThen update your SDK:")
        print("  client = OpenAI(base_url=\"http://localhost:8000/v1\", api_key=\"nb_live_...\")")
    else:
        print("Setup aborted.")

if __name__ == "__main__":
    main()
