# models/vla_moe.py
"""
VLAMoE: Vision-Language-Action model with Mixture-of-Experts for manipulation.
- Encoders: tiny vision/text encoders + MLP encoders for proprio/tactile.
- Fusion: multi-head latent attention to fuse modality tokens.
- Routing: MoE gate selects/mixes expert heads.
- Output: 7D action vector [dx, dy, dz, droll, dpitch, dyaw, gripper].
"""

import torch
import torch.nn as nn
from einops import rearrange

from .moe_gate import MoEGate
from .multihead_latent_attn import MultiHeadLatentAttention
from .experts import (
    GraspingExpert,
    PlacementExpert,
    TooluseExpert,
    TactileExpert,
)

# Pluggable expert registry. Add new experts here and list their key names in the config.
EXPERT_REGISTRY = {
    "grasping": GraspingExpert,
    "placement": PlacementExpert,
    "tooluse": TooluseExpert,
    "tactile": TactileExpert,
}


class TinyVisionEncoder(nn.Module):
    """
    Very small CNN to produce a set of spatial tokens for the fusion layer.
    Input:  [B, 3, H, W]
    Output: [B, T, D] where T = H'*W' after downsampling
    """
    def __init__(self, in_channels=3, base=32, d_model=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, 2, 1), nn.GELU(),
            nn.Conv2d(base, base * 2, 3, 2, 1), nn.GELU(),
            nn.Conv2d(base * 2, base * 4, 3, 2, 1), nn.GELU(),
            nn.Conv2d(base * 4, d_model, 3, 2, 1), nn.GELU(),
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        f = self.conv(x)                           # [B, D, H', W']
        tokens = rearrange(f, "b d h w -> b (h w) d")
        return self.proj(tokens)                   # [B, T, D]


class TinyTextEncoder(nn.Module):
    """
    Minimal text encoder: embedding + mean pooling + projection to d_model.
    Input:  [B, L] token ids
    Output: [B, 1, D] single token summarizing the command
    """
    def __init__(self, vocab_size=512, max_len=16, emb_dim=128, d_model=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.proj = nn.Linear(emb_dim, d_model)
        self.max_len = max_len

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        e = self.emb(token_ids)                    # [B, L, E]
        pooled = e.mean(dim=1)                     # [B, E]
        return self.proj(pooled).unsqueeze(1)      # [B, 1, D]


class MLPEncoder(nn.Module):
    """
    Small MLP to map low-dim states (proprio/tactile) to the model dimension.
    Input:  [B, F]
    Output: [B, 1, D]
    """
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).unsqueeze(1)            # [B, 1, D]


class VLAMoE(nn.Module):
    """
    Vision-Language-Action model with a Mixture-of-Experts head.
    Config (dict) must contain:
      d_model, n_latent, n_heads,
      vision: {in_channels, base_channels},
      language: {vocab_size, max_len, emb_dim},
      proprio_dim, tactile_dim,
      experts: {names: [...], k_active: int},
      action_dim
    """
    def __init__(self, cfg: dict):
        super().__init__()
        d_model = cfg["d_model"]
        self.action_dim = cfg["action_dim"]

        # Encoders
        self.vision = TinyVisionEncoder(
            in_channels=cfg["vision"]["in_channels"],
            base=cfg["vision"]["base_channels"],
            d_model=d_model,
        )
        self.language = TinyTextEncoder(
            vocab_size=cfg["language"]["vocab_size"],
            max_len=cfg["language"]["max_len"],
            emb_dim=cfg["language"]["emb_dim"],
            d_model=d_model,
        )
        self.proprio = MLPEncoder(cfg["proprio_dim"], d_model)
        self.tactile = MLPEncoder(cfg["tactile_dim"], d_model)

        # Fusion (latent queries attending to concatenated modality tokens)
        self.fuse = MultiHeadLatentAttention(
            d_model=d_model,
            n_heads=cfg["n_heads"],
            n_latent=cfg["n_latent"],
        )

        # MoE gate
        self.expert_names = list(cfg["experts"]["names"])
        self.k_active = cfg["experts"]["k_active"]
        n_experts = len(self.expert_names)
        self.gate = MoEGate(d_model, n_experts=n_experts, k_active=self.k_active)

        # Experts (instantiated dynamically from registry based on names in config)
        modules = {}
        for name in self.expert_names:
            if name not in EXPERT_REGISTRY:
                raise KeyError(
                    f"Unknown expert '{name}'. Available: {list(EXPERT_REGISTRY.keys())}"
                )
            modules[name] = EXPERT_REGISTRY[name](d_model, self.action_dim)
        self.experts = nn.ModuleDict(modules)

    def forward(
        self,
        image: torch.Tensor,
        text_ids: torch.Tensor,
        proprio: torch.Tensor,
        tactile: torch.Tensor,
    ):
        # Encode modalities
        v_tokens = self.vision(image)              # [B, Tv, D]
        l_token = self.language(text_ids)          # [B, 1,  D]
        p_token = self.proprio(proprio)            # [B, 1,  D]
        t_token = self.tactile(tactile)            # [B, 1,  D]

        # Concatenate tokens and fuse via latent attention
        tokens = torch.cat([v_tokens, l_token, p_token, t_token], dim=1)  # [B, T, D]
        fused = self.fuse(tokens)                  # [B, D]

        # MoE routing
        weights, _ = self.gate(fused)              # weights: [B, E]

        # Compute all expert outputs and mix by routing weights
        expert_outs = [self.experts[name](fused) for name in self.expert_names]  # list of [B, A]
        expert_out = torch.stack(expert_outs, dim=1)                              # [B, E, A]
        action = (expert_out * weights.unsqueeze(-1)).sum(dim=1)                  # [B, A]
        return action, weights
