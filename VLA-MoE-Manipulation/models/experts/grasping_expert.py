# models/experts/placement_expert.py
"""
PlacementExpert (enhanced)
--------------------------
Expert head specialized for precise placement / stacking.

Design highlights
- Residual MLP trunk with:
  * LayerNorm + GELU
  * LayerScale (stabilizes deep residuals)
  * DropPath (stochastic depth) for regularization
  * Squeeze-Excitation (SE) gating to emphasize placement-relevant channels
- Multi-head predictions:
  * Translation (dx, dy, dz)
  * Rotation   (droll, dpitch, dyaw)
  * Gripper    (open/close)
  * Stability  (in [min_stability, 1]) to softly scale the whole action
- Physical constraints:
  * Per-axis max translation (meters)
  * Per-axis max rotation   (radians)
  * Optional gravity bias to favor gentle downward dz during placement
- Output range:
  * Translation ∈ [-max_trans[i], +max_trans[i]]
  * Rotation    ∈ [-max_rot[i],   +max_rot[i]]
  * Gripper     ∈ [-1, +1]

Input:
    x: torch.Tensor, shape [B, D] (fused latent)

Output:
    action: torch.Tensor, shape [B, action_dim]
            concatenated as [dx, dy, dz, droll, dpitch, dyaw, gripper, ...pad]
"""

from typing import Iterable, Optional, Tuple
import torch
import torch.nn as nn


# ---------------------------
# Building blocks
# ---------------------------
class DropPath(nn.Module):
    """Stochastic depth as in timm; no-op at eval."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # work with (B, ...) shape
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class ResidualMLPBlock(nn.Module):
    """
    Residual MLP with LayerScale, DropPath, SE gating.
    y = x + DropPath( LayerScale( SE( MLP(LN(x)) )))
    """
    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        drop: float = 0.0,
        drop_path: float = 0.0,
        layerscale_init: float = 1e-3,
        se_reduction: int = 4,
    ):
        super().__init__()
        hidden = d_model * hidden_mult
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)

        # Squeeze-Excitation over channel dim
        self.se = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(1, d_model // se_reduction)),
            nn.GELU(),
            nn.Linear(max(1, d_model // se_reduction), d_model),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.droppath = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        # LayerScale: learnable per-channel scaling of the residual branch
        self.gamma = nn.Parameter(layerscale_init * torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        h = self.dropout(h)
        # SE gating
        gate = self.se(h)
        h = h * gate
        # LayerScale + DropPath
        h = self.droppath(self.gamma * h)
        return x + h


# ---------------------------
# Expert head
# ---------------------------
class PlacementExpert(nn.Module):
    """
    Args:
        d_model: feature dimension of input x.
        action_dim: total action dimension (>= 7).
        depth: number of residual blocks.
        hidden_mult: expansion in residual MLP.
        dropout: dropout inside MLP.
        drop_path: stochastic depth probability.
        min_stability: lower bound for stability scaling (0 < s <= 1).
        max_trans: per-axis max translation (meters), tuple of 3 floats.
        max_rot: per-axis max rotation (radians),   tuple of 3 floats.
        gravity_bias: non-negative small bias (meters) subtracted from dz to
                      gently encourage downward motion during placement.
        temperature: global multiplier for action magnitude (post-squash).
    """
    def __init__(
        self,
        d_model: int,
        action_dim: int,
        depth: int = 4,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.1,
        min_stability: float = 0.5,
        max_trans: Tuple[float, float, float] = (0.05, 0.05, 0.05),  # 5 cm
        max_rot: Tuple[float, float, float] = (0.25, 0.25, 0.25),    # ~14 deg
        gravity_bias: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        assert action_dim >= 7, "PlacementExpert expects action_dim >= 7."
        self.action_dim = action_dim
        self.min_stability = float(min_stability)
        self.temperature = float(temperature)

        # Save physical limits as buffers (non-trainable, on the right device)
        self.register_buffer("max_trans", torch.tensor(max_trans, dtype=torch.float32))
        self.register_buffer("max_rot", torch.tensor(max_rot, dtype=torch.float32))
        self.gravity_bias = float(gravity_bias)

        # A small learned task token biases features toward placement semantics
        self.task_token = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # Trunk
        blocks = []
        for i in range(depth):
            # Linearly decay drop_path across depth for better stability
            dp = drop_path * float(i) / max(1, depth - 1)
            blocks.append(
                ResidualMLPBlock(
                    d_model=d_model,
                    hidden_mult=hidden_mult,
                    drop=dropout,
                    drop_path=dp,
                    layerscale_init=1e-3,
                    se_reduction=4,
                )
            )
        self.trunk = nn.Sequential(*blocks)

        # Heads
        self.head_trans = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.head_rot = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.head_grip = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.head_stability = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),  # [0,1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _squash_translation(self, t_raw: torch.Tensor) -> torch.Tensor:
        """
        Map unconstrained logits to bounded translations:
        tanh -> [-1,1] then scale by per-axis max_trans (meters).
        """
        t = torch.tanh(t_raw) * self.max_trans  # [B, 3]
        # gravity bias: gently encourage negative dz during placement
        if self.gravity_bias > 0:
            t = t.clone()
            t[:, 2] = t[:, 2] - self.gravity_bias
            # keep within bounds after bias
            t[:, 2] = torch.clamp(t[:, 2], -self.max_trans[2], self.max_trans[2])
        return t

    def _squash_rotation(self, r_raw: torch.Tensor) -> torch.Tensor:
        """
        Map logits to bounded euler deltas:
        tanh -> [-1,1] then scale by per-axis max_rot (radians).
        """
        r = torch.tanh(r_raw) * self.max_rot  # [B, 3]
        return r

    def _squash_gripper(self, g_raw: torch.Tensor) -> torch.Tensor:
        """
        Map logits to [-1, 1] via symmetric sigmoid (2*sigmoid - 1).
        Use tanh-equivalent but with potentially different gradients.
        """
        g = 2.0 * torch.sigmoid(g_raw) - 1.0  # [B, 1]
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] fused latent features

        Returns:
            action: [B, action_dim] concatenated as
                    [dx, dy, dz, droll, dpitch, dyaw, gripper, ...zeros]
        """
        B, D = x.shape
        h = x + self.task_token.expand(B, -1)
        h = self.trunk(h)

        # Heads
        t_raw = self.head_trans(h)     # [B, 3]
        r_raw = self.head_rot(h)       # [B, 3]
        g_raw = self.head_grip(h)      # [B, 1]
        s = self.head_stability(h)     # [B, 1] in [0,1]
        s = self.min_stability + (1.0 - self.min_stability) * s  # [min, 1]

        # Squash to physical ranges
        t = self._squash_translation(t_raw)        # meters
        r = self._squash_rotation(r_raw)           # radians
        g = self._squash_gripper(g_raw)            # [-1, 1]

        base = torch.cat([t, r, g], dim=-1)        # [B, 7]
        # Stability scaling + temperature
        out = base * s * self.temperature

        # If action_dim > 7, pad with zeros (keep interface generic)
        if self.action_dim > 7:
            pad = torch.zeros(B, self.action_dim - 7, device=x.device, dtype=x.dtype)
            out = torch.cat([out, pad], dim=-1)
        return out
