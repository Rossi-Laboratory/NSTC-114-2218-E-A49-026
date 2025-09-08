import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEGate(nn.Module):
    """Simple gating network for Mixture-of-Experts.
    Produces soft mixture weights over E experts, and optionally selects top-k active.
    """
    def __init__(self, d_model: int, n_experts: int, k_active: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.k_active = k_active
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_experts),
        )

    def forward(self, fused_latent: torch.Tensor):
        # fused_latent: [B, D]
        logits = self.fc(fused_latent)  # [B, E]
        # soft weights
        probs = F.softmax(logits, dim=-1)

        if self.k_active is None or self.k_active >= self.n_experts:
            return probs, None  # all experts soft mix

        # top-k mask (hard routing)
        topk_vals, topk_idx = torch.topk(probs, k=self.k_active, dim=-1)  # [B, K]
        mask = torch.zeros_like(probs)
        mask.scatter_(1, topk_idx, 1.0)
        # renormalize only top-k
        routed = (probs * mask)
        routed = routed / (routed.sum(dim=-1, keepdim=True) + 1e-8)
        return routed, topk_idx
