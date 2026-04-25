import json

with open(r'C:\Users\shaya\.gemini\antigravity\brain\c0ac5058-1bb0-443e-81a0-b029c0c4d33d\.system_generated\steps\12\content.md', 'r') as f:
    text = f.read()
json_str = text.split('---', 1)[-1].strip()

try:
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
    id_str = id_str.lower()
    if id_str.endswith('-q8') or id_str.endswith('-q4') or id_str.endswith('-fp16'):
        parts = id_str.rsplit('-', 1)
        base = parts[0]
    else:
        base = id_str
        
    base = base.replace("llama-3.2-vision", "llama3.2-vision")
    base = base.replace("llama-3.2", "llama3.2")
    base = base.replace("llama-3.3", "llama3.3")
    base = base.replace("llama-3.1", "llama3.1")
    base = base.replace("qwen-2.5-vl", "qwen2.5-vl")
    base = base.replace("qwen-2.5", "qwen2.5")
    
    if "deepseek-r1-distill" in base:
        return "deepseek-r1:" + base.split('-')[-1]
    
    if "-" in base and "deepseek-coder" not in base and "phi-4" not in base:
        parts = base.rsplit('-', 1)
        return f"{parts[0]}:{parts[1]}"
    
    # fallback
    if ":" not in base:
        if "-" in base:
            parts = base.rsplit('-', 1)
            return f"{parts[0]}:{parts[1]}"
        return f"{base}:latest"
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

grouped = {}
for m in models_data:
    if m["quantization"] not in ["Q4_K_M", "fp16", "f16"]:
        continue
    base_id = m["id"]
    if base_id.endswith('-q4') or base_id.endswith('-fp16'):
        base_id = base_id.rsplit('-', 1)[0]
    if base_id not in grouped:
        grouped[base_id] = m

known_local_str = "KNOWN_LOCAL_MODELS = [\n"
models_str = "MODELS = [\n"

for base_id, m in grouped.items():
    ollama_tag = get_ollama_tag(base_id)
    fam = get_family(m["name"])
    caps = map_capabilities(m.get("features", []))
    if "coding" in m["name"].lower() or "coder" in m["name"].lower() or m.get("swe_bench_score"):
        caps.extend(["coding", "code"])
    caps = list(set(caps))
    
    ctx = m.get("max_context_k", 32)
    vram_est = m.get("weight_gb", 0) + (m.get("kv_per_1k_gb", 0) * 4)
    vram_gb = round(vram_est, 1) if vram_est else round(m.get("params_b", 7) * 0.7, 1)
    ram_gb = round(vram_gb + 2.0, 1)
    
    # for KNOWN_LOCAL_MODELS
    known_local_str += f'    {{"tag": "{ollama_tag}", "params_b": {m["params_b"]}, "vram_gb": {vram_gb}, "capabilities": {caps}}},\n'
    
    # for MODELS
    toks = "{'rtx3090': 50, 'rtx4090': 80, 'm2max': 30, 'cpu': 2.0}"
    notes = m.get("notes", "").replace('"', "'")
    models_str += f'    ModelProfile("{ollama_tag}", "{fam}", {m["params_b"]}, "{m["quantization"]}", {vram_gb}, {ram_gb}, {ctx}, {toks}, 2.0, {caps}, {caps}, "{ollama_tag}", "{notes}", weight_gb={m.get("weight_gb", 0)}, kv_per_1k_gb={m.get("kv_per_1k_gb", 0)}, layers={m.get("layers", 0)}),\n'

known_local_str += "]\n"
models_str += "]\n"

with open('scratch/generated.py', 'w') as out:
    out.write(known_local_str + "\n" + models_str)
