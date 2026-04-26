import re
import math
from neuralbrok.models import ModelProfile, get_tok_per_sec, resolve_model, MODEL_REGISTRY
from neuralbrok.detect import detect_device

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
            # Log-scale base score — avoids 70B always beating 7B regardless of fit.
            # log2(1)=0, log2(7)≈2.8→11, log2(32)=5→20, log2(70)≈6.1→24
            score = math.log2(max(model.params_b, 1.0)) * 4.0

            for w in workload:
                if w in model.capabilities:
                    score += 15
                if w in model.recommended_for:
                    score += 20

            # tokens_per_sec ≈ bandwidth_GB_s / model_weight_GB
            weight = model.weight_gb if model.weight_gb > 0 else model.vram_gb
            est_tok_s = bandwidth / (weight + 1.0)

            if est_tok_s > 60:
                score += 15
            elif est_tok_s > 30:
                score += 8
            elif est_tok_s < 10:
                score -= 15

            if "long_context" in workload and model.ctx_k >= 128:
                score += 25

            # Headroom: reward models that leave breathing room, cap to avoid dominating
            vram_needed = model.weight_gb + (model.kv_per_1k_gb * 4.0) if model.weight_gb > 0 else model.vram_gb
            headroom = self.available_vram_gb - vram_needed
            score += min(max(0.0, headroom * 2.0), 20.0)

            # MoE detection: matches both `30b-a3b` (hyphen) and `35b:a3b` (colon) formats
            is_moe = re.search(r'[-:][aA]\d+', model.name) is not None
            if is_moe and "fast_response" in workload:
                score += 15

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
