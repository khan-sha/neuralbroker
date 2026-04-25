import json
import re

with open(r'C:\Users\shaya\.gemini\antigravity\brain\c0ac5058-1bb0-443e-81a0-b029c0c4d33d\.system_generated\steps\56\content.md', 'r') as f:
    text = f.read()
json_str = text.split('---', 1)[-1].strip()

try:
    gpus_data = json.loads(json_str)
except Exception:
    import urllib.request
    url = "https://raw.githubusercontent.com/BenD10/whatmodels/main/src/lib/data/gpus.json"
    with urllib.request.urlopen(url) as response:
        gpus_data = json.loads(response.read().decode())

gpu_db_str = "GPU_DATABASE = [\n"
for g in gpus_data:
    if "vram_options" in g:
        vram_options = json.dumps(g["vram_options"])
        gpu_db_str += f'    {{"id": "{g["id"]}", "name": "{g["name"]}", "manufacturer": "{g["manufacturer"]}", "vram_options": {vram_options}}},\n'
    else:
        gpu_db_str += f'    {{"id": "{g["id"]}", "name": "{g["name"]}", "manufacturer": "{g["manufacturer"]}", "vram_gb": {g["vram_gb"]}, "bandwidth_gbps": {g["bandwidth_gbps"]}}},\n'
gpu_db_str += "]\n"

with open('src/neuralbrok/hardware.py', 'r', encoding='utf-8') as f:
    hardware_content = f.read()

hardware_content = re.sub(r'GPU_DATABASE = \[.*?\]\n', gpu_db_str, hardware_content, flags=re.DOTALL)

with open('src/neuralbrok/hardware.py', 'w', encoding='utf-8') as f:
    f.write(hardware_content)

print("GPU DB injected successfully!")
