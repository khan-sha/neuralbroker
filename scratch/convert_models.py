import json
import re
from pathlib import Path

# Load json
with open(r'C:\Users\shaya\.gemini\antigravity\brain\c0ac5058-1bb0-443e-81a0-b029c0c4d33d\.system_generated\steps\12\content.md', 'r') as f:
    text = f.read()

# Extract JSON string from markdown content
json_str = text.split('---', 1)[-1].strip()
try:
    # If the view_file truncated the content we might need to load it from the URL again, or we can just fetch it here.
    models_data = json.loads(json_str)
except Exception:
    import urllib.request
    url = "https://raw.githubusercontent.com/BenD10/whatmodels/main/src/lib/data/models.json"
    with urllib.request.urlopen(url) as response:
        models_data = json.loads(response.read().decode())

def map_capabilities(features):
    caps = ["chat"]
    for f in features:
        if f == "tool_use":
            caps.append("tools")
            caps.append("agentic")
        if f == "vision":
            caps.append("vision")
        if f == "reasoning":
            caps.append("reasoning")
            caps.append("math")
    return list(set(caps))

def get_ollama_tag(id_str):
    # simple mapping from id to ollama tag
    # e.g., llama-3.2-1b-q4 -> llama3.2:1b
    # qwen-2.5-3b-q4 -> qwen2.5:3b
    # deepseek-r1-distill-qwen-7b-q4 -> deepseek-r1:7b
    
    id_str = id_str.lower()
    if id_str.endswith('-q8') or id_str.endswith('-q4') or id_str.endswith('-fp16'):
        parts = id_str.rsplit('-', 1)
        base = parts[0]
    else:
        base = id_str
        
    if "llama-3.2-vision" in base:
        base = base.replace("llama-3.2-vision", "llama3.2-vision")
    elif "llama-3.2" in base:
        base = base.replace("llama-3.2", "llama3.2")
    elif "llama-3.3" in base:
        base = base.replace("llama-3.3", "llama3.3")
    elif "llama-3.1" in base:
        base = base.replace("llama-3.1", "llama3.1")
    elif "qwen-2.5-vl" in base:
        base = base.replace("qwen-2.5-vl", "qwen2.5-vl")
    elif "qwen-2.5" in base:
        base = base.replace("qwen-2.5", "qwen2.5")
    elif "qwen3-coder" in base:
        pass
    elif "qwen3-next" in base:
        pass
    elif "qwen3" in base:
        pass
    elif "deepseek-r1-distill" in base:
        # e.g. deepseek-r1-distill-qwen-7b
        return "deepseek-r1:" + base.split('-')[-1]
    
    # split by the last hyphen to separate model and size
    if "-" in base:
        parts = base.rsplit('-', 1)
        return f"{parts[0]}:{parts[1]}"
    
    return base

def get_family(name):
    name = name.lower()
    if 'llama' in name: return 'llama'
    if 'qwen' in name: return 'qwen'
    if 'deepseek' in name: return 'deepseek'
    if 'mistral' in name: return 'mistral'
    if 'gemma' in name: return 'gemma'
    if 'phi' in name: return 'phi'
    return 'unknown'

profiles = []
for m in models_data:
    # Only pick Q4_K_M for brevity, or include both?
    # NeuralBroker typically only wants 1 representation per tag.
    # Let's pick Q4_K_M if possible. We'll group by tag
    pass

# We group by model base name
grouped = {}
for m in models_data:
    if m["quantization"] not in ["Q4_K_M", "fp16", "f16"]:
        continue
    
    base_id = m["id"]
    if base_id.endswith('-q4') or base_id.endswith('-fp16'):
        base_id = base_id.rsplit('-', 1)[0]
        
    if base_id not in grouped:
        grouped[base_id] = m

print(f"Loaded {len(grouped)} unique models.")

with open('scratch/models_output.txt', 'w') as out:
    for base_id, m in grouped.items():
        ollama_tag = get_ollama_tag(base_id)
        fam = get_family(m["name"])
        caps = map_capabilities(m.get("features", []))
        if "coding" in m["name"].lower() or "coder" in m["name"].lower() or "swe_bench_score" in m and m["swe_bench_score"]:
            caps.append("coding")
            caps.append("code")
        caps = list(set(caps))
        
        ctx = m.get("max_context_k", 32)
        vram_est = m.get("weight_gb", 0) + (m.get("kv_per_1k_gb", 0) * 4) # rough estimate for 4k ctx
        vram_gb = round(vram_est, 1) if vram_est else m.get("params_b", 7) * 0.7
        ram_gb = round(vram_gb + 2.0, 1)
        
        # default tok_per_sec dummy
        toks = "{'rtx3090': 50, 'rtx4090': 80, 'm2max': 30, 'cpu': 2.0}"
        
        notes = m.get("notes", "").replace('"', "'")
        
        out.write(f'    ModelProfile("{ollama_tag}", "{fam}", {m["params_b"]}, "{m["quantization"]}", {vram_gb}, {ram_gb}, {ctx}, {toks}, 2.0, {caps}, {caps}, "{ollama_tag}", "{notes}", weight_gb={m.get("weight_gb", 0)}, kv_per_1k_gb={m.get("kv_per_1k_gb", 0)}, layers={m.get("layers", 0)}),\n')

print("Wrote to scratch/models_output.txt")
