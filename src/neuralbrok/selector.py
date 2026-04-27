import re
import math
from neuralbrok.models import ModelProfile, get_tok_per_sec, resolve_model, MODEL_REGISTRY
from neuralbrok.detect import detect_device

# Real benchmark weights per workload — derived from AA Intelligence Index + Arena Elo data.
# Maps workload → {capability_tag: bonus_points}
# Bonuses reflect where model families actually outperform vs pure parameter count.
_WORKLOAD_BONUS: dict[str, dict[str, int]] = {
    "coding":       {"coding": 25, "code": 20, "agentic": 10, "tools": 8},
    "reasoning":    {"reasoning": 25, "math": 20, "agentic": 8},
    "math":         {"math": 30, "reasoning": 15},
    "long_context": {"long_context": 20, "agentic": 5},
    "chat":         {"chat": 15, "tools": 8, "agentic": 8},
    "tools":        {"tools": 25, "agentic": 20, "coding": 10},
    "agentic":      {"agentic": 25, "tools": 20, "coding": 10, "reasoning": 8},
    "vision":       {"vision": 30},
    "fast_response":{"fast_response": 20, "chat": 10},
    "rag":          {"long_context": 15, "reasoning": 10, "chat": 8},
}

# AA Intelligence Index scores for known local model families (0-60 scale).
# Used as quality anchor when model.intelligence_score not set in registry.
_FAMILY_INTELLIGENCE: dict[str, float] = {
    "deepseek-r1":      27.0,  # AA: 27 (0528 version)
    "qwq":              30.0,  # reasoning-focused, similar to deepseek-r1
    "qwen3":            30.0,  # qwen3 8b-32b range
    "qwen3.5":          32.0,  # AA: qwen3.5-9b = 32
    "qwen2.5-coder":    22.0,  # coding specialist
    "qwen2.5":          18.0,  # general chat
    "gemma-4":          22.0,  # gemma4 family
    "gemma-3":          16.0,
    "llama-4":          18.0,  # AA: llama-4-maverick = 18
    "llama3.3":         14.0,  # AA: 14
    "mistral":          15.0,
    "phi-4":            18.0,
    "gpt-oss":          25.0,  # GPT-OSS 20B on Groq
    "glm":              20.0,  # GLM-5.1 = 51 on Intelligence Index
    "nemotron":         22.0,
}


def _get_intelligence(model: ModelProfile) -> float:
    """Return intelligence score: explicit registry value beats family estimate."""
    if model.intelligence_score > 0:
        return model.intelligence_score
    name_lc = model.name.lower()
    for family, score in _FAMILY_INTELLIGENCE.items():
        if family in name_lc:
            # Scale by log2(params) within family — larger = smarter, diminishing returns
            scale = 1.0 + 0.08 * math.log2(max(model.params_b, 1.0))
            return min(score * scale, 55.0)
    # Unknown model: pure log2(params) estimate, capped at 30
    return min(math.log2(max(model.params_b, 1.0)) * 3.5, 30.0)


class SmartModelSelector:
    def __init__(self, device_key: str, available_vram_gb: float, runnable: list[ModelProfile], hw_profile=None):
        self.device_key = device_key
        self.available_vram_gb = available_vram_gb
        self.runnable = runnable
        self.hw_profile = hw_profile or detect_device()

    def resolve_model(self, name: str) -> str:
        """Resolve a model name or alias to its registry-recommended tag."""
        return resolve_model(name)

    def for_workload(self, workload: list[str]) -> list[ModelProfile]:
        scored = []
        bandwidth = self.hw_profile.bandwidth_gbps or 40.0

        for model in self.runnable:
            # Base: intelligence score (0-55) accounts for 60% of raw score.
            # log2(params) contributes 40% — bigger still helps within same family.
            intel = _get_intelligence(model)
            score = intel * 0.9 + math.log2(max(model.params_b, 1.0)) * 2.5

            # Workload-specific bonuses — capability match weighted by workload relevance
            for w in workload:
                bonus_map = _WORKLOAD_BONUS.get(w, {"chat": 10})
                for cap, pts in bonus_map.items():
                    if cap in model.capabilities:
                        score += pts
                    if cap in model.recommended_for:
                        score += pts * 0.5

            # Speed signal: bandwidth / weight estimates tok/s
            weight = model.weight_gb if model.weight_gb > 0 else model.vram_gb
            est_tok_s = bandwidth / (weight + 1.0)

            if est_tok_s > 80:
                score += 18   # very fast (small models on good HW)
            elif est_tok_s > 50:
                score += 12
            elif est_tok_s > 25:
                score += 5
            elif est_tok_s < 8:
                score -= 20   # unusably slow for interactive use

            # Long-context bonus — use actual ctx_k, not just ≥128k threshold
            if "long_context" in workload or "rag" in workload:
                if model.ctx_k >= 1000:
                    score += 30    # 1M context (llama-4, qwen3.5-122b+)
                elif model.ctx_k >= 262:
                    score += 22    # 262k (qwen3.5 family)
                elif model.ctx_k >= 128:
                    score += 15    # 128k standard

            # MoE bonus — activation-efficient: good for all workloads, not just fast_response.
            # Pattern: 35b:a3b or 30b-a3b
            is_moe = re.search(r'\d+b[-:][aA]\d+', model.name, re.IGNORECASE) is not None
            if is_moe:
                score += 12                            # always efficient at same VRAM cost
                if "fast_response" in workload:
                    score += 10                        # extra for latency-sensitive tasks

            # VRAM headroom — reward models that leave breathing room (capped to prevent dominance)
            vram_needed = model.weight_gb + (model.kv_per_1k_gb * 4.0) if model.weight_gb > 0 else model.vram_gb
            headroom = self.available_vram_gb - vram_needed
            score += min(max(0.0, headroom * 1.5), 15.0)

            scored.append((model, score))
            
        if not scored: return []
        max_score = max(s for m, s in scored)
        min_score = min(s for m, s in scored)
        
        normalized = []
        for model, score in scored:
            if max_score > min_score:
                norm_score = ((score - min_score) / (max_score - min_score)) * 100
            else:
                norm_score = 100.0
            model._temp_score = norm_score
            normalized.append(model)
            
        normalized.sort(key=lambda m: m._temp_score, reverse=True)
        return normalized[:3]

    def rank_all(self) -> list[tuple[ModelProfile, float, str]]:
        scored = self.for_workload(["chat", "coding", "reasoning", "tools"])
        bandwidth = self.hw_profile.bandwidth_gbps or 40.0
        
        result = []
        for m in scored:
            weight = m.weight_gb if m.weight_gb > 0 else m.vram_gb
            est_tok_s = bandwidth / (weight + 1.0)
            reason = f"Best suited based on VRAM · ~{int(est_tok_s)} tok/s on your {self.hw_profile.gpu_model}"
            result.append((m, m._temp_score, reason))
        return result

    def best_single(self, workload: list[str]) -> ModelProfile:
        res = self.for_workload(workload)
        return res[0] if res else None
