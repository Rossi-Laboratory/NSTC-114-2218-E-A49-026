# models/experts/placement_expert.py
"""
PlacementExpert
---------------
Expert head specialized for placement / stacking / precise pose adjustment.

Design:
- Residual MLP trunk with lightweight Squeeze-Excitation (SE) gating.
- Separate heads for translation (dx, dy, dz), rotation (droll, dpitch, dyaw),
  and gripper (open/close). An additional stability head modulates the overall
  magnitude to promote gentle, stable placements.
- All outputs are squashed to [-1, 1] using tanh, then scaled by
  a stability factor in [min_stability, 1].

Input:
    x: torch.Tensor of shape [B, D]  (fused latent from the backbone)

Output:
    action: torch.Tensor of shape [B, A] with A == action_dim
            (ordered as [dx, dy, dz, droll, dpitch, dyaw, gripper])
"""

from typing import Optional
import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    """Simple residual MLP block with LayerNorm and GELU."""
    def __init__(self, d_model: int, hidden_mult: int = 4, drop: float = 0.0):
        super().__init__()
        hidden = d_model * hidden_mult
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        h = self.drop(h)
        return x + h


class SEGate(nn.Module):
    """
    Squeeze-Excitation style gating over the channel (feature) dimension.
    Produces coefficients in [0, 1] to softly emphasize placement-relevant features.
    """
    def __init__(self, d_model: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, d_model // reduction)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.fc(x)  # [B, D] in [0,1]
        return x * gate


class PlacementExpert(nn.Module):
    """
    Placement expert head.

    Args:
        d_model: feature dimension of the fused latent.
        action_dim: size of the action vector; expected 7 in the current project.
        depth: number of residual MLP blocks in the trunk.
        hidden_mult: width multiplier for the residual MLP hidden size.
        dropout: dropout probability in residual blocks.
        min_stability: lower bound of stability scaling (in (0,1]); action is
                       multiplied by s in [min_stability, 1].
    """
    def __init__(
        self,
        d_model: int,
        action_dim: int,
        depth: int = 3,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        min_stability: float = 0.5,
    ):
        super().__init__()
        assert action_dim >= 7, "PlacementExpert expects action_dim >= 7."

        self.action_dim = action_dim
        self.min_stability = float(min_stability)

        # Optional task token nudges the trunk toward "placement" semantics
        self.task_token = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # Trunk: Residual MLP blocks + SE gate to highlight useful channels
        blocks = []
        for _ in range(depth):
            blocks.append(ResidualMLPBlock(d_model, hidden_mult=hidden_mult, drop=dropout))
            blocks.append(SEGate(d_model, reduction=4))
        self.trunk = nn.Sequential(*blocks)

        # Prediction heads
        self.head_trans = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),  # dx, dy, dz
        )
        self.head_rot = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),  # droll, dpitch, dyaw
        )
        self.head_grip = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),  # gripper
        )
        # Stability/confidence in [0,1]; used to gently scale the outputs
        self.head_stability = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] fused latent features

        Returns:
            action: [B, action_dim] = [dx, dy, dz, droll, dpitch, dyaw, gripper, ... (zeros if A>7)]
        """
        b, d = x.shape
        h = x + self.task_token.expand(b, -1)  # bias features toward placement
        h = self.trunk(h)

        # Heads
        t = torch.tanh(self.head_trans(h))  # [-1, 1]^3
        r = torch.tanh(self.head_rot(h))    # [-1, 1]^3
        g = torch.tanh(self.head_grip(h))   # [-1, 1]^1

        # Stability scaling in [min_stability, 1]
        s = self.head_stability(h)                          # [B, 1] in [0,1]
        s = self.min_stability + (1.0 - self.min_stability) * s  # [B,1] in [min,1]

        base = torch.cat([t, r, g], dim=-1)  # [B, 7]
        scaled = base * s  # encourage gentle, stable adjustments

        # If action_dim > 7, pad with zeros to keep interface generic
        if self.action_dim > 7:
            pad = torch.zeros(b, self.action_dim - 7, device=x.device, dtype=x.dtype)
            scaled = torch.cat([scaled, pad], dim=-1)

        return scaled
