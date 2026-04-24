from neuralbrok.models import ModelProfile, get_tok_per_sec

class SmartModelSelector:
    def __init__(self, device_key: str, available_vram_gb: float, runnable: list[ModelProfile]):
        self.device_key = device_key
        self.available_vram_gb = available_vram_gb
        self.runnable = runnable

    def for_workload(self, workload: list[str]) -> list[ModelProfile]:
        scored = []
        for model in self.runnable:
            score = model.params_b
            
            for w in workload:
                if w in model.capabilities:
                    score += 15
                if w in model.recommended_for:
                    score += 20
            
            tok_s = get_tok_per_sec(model, self.device_key)
            if tok_s > 60:
                score += 10
            elif tok_s > 30:
                score += 5
            elif tok_s < 10:
                score -= 10
                
            if "long_context" in workload and model.ctx_k >= 128:
                score += 25
                
            headroom = self.available_vram_gb - model.vram_gb
            score += headroom * 2
            
            is_moe = "a" in model.name and "-" in model.name
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
        result = []
        for m in scored:
            tok_s = get_tok_per_sec(m, self.device_key)
            reason = f"Best suited based on VRAM · {int(tok_s)} tok/s on your hardware"
            result.append((m, m._temp_score, reason))
        return result

    def best_single(self, workload: list[str]) -> ModelProfile:
        res = self.for_workload(workload)
        return res[0] if res else None
