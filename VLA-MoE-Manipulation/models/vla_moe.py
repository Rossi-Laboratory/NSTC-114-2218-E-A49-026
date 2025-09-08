import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .moe_gate import MoEGate
from .multihead_latent_attn import MultiHeadLatentAttention
from .experts.grasping_expert import GraspingExpert
from .experts.placement_expert import PlacementExpert
from .experts.tooluse_expert import TooluseExpert
from .experts.tactile_expert import TactileExpert

class TinyVisionEncoder(nn.Module):
    def __init__(self, in_channels=3, base=32, d_model=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, 2, 1), nn.GELU(),
            nn.Conv2d(base, base*2, 3, 2, 1), nn.GELU(),
            nn.Conv2d(base*2, base*4, 3, 2, 1), nn.GELU(),
            nn.Conv2d(base*4, d_model, 3, 2, 1), nn.GELU(),
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B,3,H,W]
        f = self.conv(x)  # [B, D, H', W']
        B, D, H, W = f.shape
        tokens = rearrange(f, 'b d h w -> b (h w) d')  # [B, T, D]
        return self.proj(tokens)

class TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size=512, max_len=16, emb_dim=128, d_model=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.proj = nn.Linear(emb_dim, d_model)
        self.max_len = max_len

    def forward(self, token_ids):
        # token_ids: [B, L]
        e = self.emb(token_ids)  # [B, L, E]
        pooled = e.mean(dim=1)   # average pooling (toy)
        return self.proj(pooled).unsqueeze(1)  # [B, 1, D]

class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.net(x).unsqueeze(1)  # [B, 1, D]

class VLAMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg['d_model']
        self.action_dim = cfg['action_dim']

        # encoders
        self.vision = TinyVisionEncoder(
            in_channels=cfg['vision']['in_channels'],
            base=cfg['vision']['base_channels'],
            d_model=d_model
        )
        self.language = TinyTextEncoder(
            vocab_size=cfg['language']['vocab_size'],
            max_len=cfg['language']['max_len'],
            emb_dim=cfg['language']['emb_dim'],
            d_model=d_model
        )
        self.proprio = MLPEncoder(cfg['proprio_dim'], d_model)
        self.tactile = MLPEncoder(cfg['tactile_dim'], d_model)

        # fusion
        self.fuse = MultiHeadLatentAttention(
            d_model=d_model,
            n_heads=cfg['n_heads'],
            n_latent=cfg['n_latent']
        )

        # MoE gate
        self.expert_names = cfg['experts']['names']
        self.k_active = cfg['experts']['k_active']
        n_experts = len(self.expert_names)
        self.gate = MoEGate(d_model, n_experts=n_experts, k_active=self.k_active)

        # experts
        self.experts = nn.ModuleDict({
            'grasping': GraspingExpert(d_model, self.action_dim),
            'placement': PlacementExpert(d_model, self.action_dim),
            'tooluse': TooluseExpert(d_model, self.action_dim),
            'tactile': TactileExpert(d_model, self.action_dim),
        })

    def forward(self, image, text_ids, proprio, tactile):
        # encoders
        v_tokens = self.vision(image)                 # [B, Tv, D]
        l_token  = self.language(text_ids)            # [B, 1,  D]
        p_token  = self.proprio(proprio)              # [B, 1,  D]
        t_token  = self.tactile(tactile)              # [B, 1,  D]

        # concatenate modality tokens
        tokens = torch.cat([v_tokens, l_token, p_token, t_token], dim=1)  # [B, T, D]
        fused = self.fuse(tokens)  # [B, D]

        # gating
        weights, topk_idx = self.gate(fused)  # [B, E]
        # expert outputs (all), then mix by weights
        expert_outs = []
        for name in self.expert_names:
            expert_outs.append(self.experts[name](fused))  # [B, A]
        expert_out = torch.stack(expert_outs, dim=1)  # [B, E, A]

        # mix
        weights_expanded = weights.unsqueeze(-1)  # [B, E, 1]
        action = (expert_out * weights_expanded).sum(dim=1)  # [B, A]
        return action, weights
