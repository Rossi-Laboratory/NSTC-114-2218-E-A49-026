# models/experts/tactile_expert.py
"""
TactileExpert (enhanced)
------------------------
Expert head specialized for contact-aware manipulation where tactile/force feedback
matters (e.g., insertion, gentle alignment, force-limited interaction).

Key ideas
- Residual MLP trunk with LayerScale, DropPath, and SE gating.
- Predicts a base 7D action (translation, rotation, gripper), then modulates it
  using contact-aware factors inferred from the fused latent (which already
  encodes tactile/proprio tokens in the backbone).
- Normal/tangential decomposition of translation with separate compliance scales:
  * normal compliance  c_n   ∈ [c_n_min, 1]
  * tangential compl. c_t   ∈ [c_t_min, 1], further reduced by slip risk
- Rotation damping c_rot ∈ [c_rot_min, 1] to avoid aggressive orientation changes.
- Slip risk (in [0,1]) reduces tangential motion; intended to mitigate slipping.
- Optional gripper bias/scale for gentle force control.
- Physical constraints: per-axis max translation (meters) and rotation (radians).

Output
- [dx, dy, dz, droll, dpitch, dyaw, gripper] (and zero-padded if A>7).
"""

from typing import Tuple
import torch
import torch.nn as nn


# ---------------------------
# Building blocks
# ---------------------------
class DropPath(nn.Module):
    """Stochastic depth; acts as identity during eval."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask


class ResidualMLPBlock(nn.Module):
    """
    Residual MLP with LayerScale, DropPath, and SE gating.
    y = x + DropPath( gamma * SE( MLP(LN(x)) ) )
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
        self.dropout = nn.Dropout(drop) if drop > 0 else nn.Identity()

        self.se = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(1, d_model // se_reduction)),
            nn.GELU(),
            nn.Linear(max(1, d_model // se_reduction), d_model),
            nn.Sigmoid(),
        )

        self.gamma = nn.Parameter(layerscale_init * torch.ones(d_model))
        self.droppath = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        h = self.dropout(h)
        h = h * self.se(h)
        h = self.droppath(self.gamma * h)
        return x + h


# ---------------------------
# Expert head
# ---------------------------
class TactileExpert(nn.Module):
    """
    Args:
        d_model: fused latent feature dimension.
        action_dim: total action dimension (>= 7).
        depth: number of residual blocks.
        hidden_mult: expansion factor for MLP width.
        dropout: dropout probability in MLP.
        drop_path: stochastic depth probability (0..1).
        # Physical limits (per-axis):
        max_trans: (meters)  tuple of 3 floats for dx, dy, dz bounds.
        max_rot:   (radians) tuple of 3 floats for droll, dpitch, dyaw bounds.
        # Compliance/damping:
        c_n_min:  lower bound for normal-direction compliance   (0<c<=1).
        c_t_min:  lower bound for tangential-direction compliance(0<c<=1).
        c_rot_min:lower bound for rotation damping              (0<c<=1).
        slip_gain: scales how strongly slip risk reduces tangential motion.
        # Gripper:
        gripper_bias_max: absolute bias applied to gripper in [-bias, +bias].
        temperature: global magnitude multiplier after modulation.
    """
    def __init__(
        self,
        d_model: int,
        action_dim: int,
        depth: int = 4,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.1,
        # limits
        max_trans: Tuple[float, float, float] = (0.03, 0.03, 0.03),   # 3 cm
        max_rot:   Tuple[float, float, float] = (0.20, 0.20, 0.20),   # ~11.5 deg
        # compliance/damping
        c_n_min:   float = 0.3,
        c_t_min:   float = 0.2,
        c_rot_min: float = 0.4,
        slip_gain: float = 0.6,
        # gripper
        gripper_bias_max: float = 0.15,
        temperature: float = 1.0,
    ):
        super().__init__()
        assert action_dim >= 7, "TactileExpert expects action_dim >= 7."
        self.action_dim = action_dim

        # Buffers for physical limits
        self.register_buffer("max_trans", torch.tensor(max_trans, dtype=torch.float32))
        self.register_buffer("max_rot",   torch.tensor(max_rot,   dtype=torch.float32))

        # Compliance and damping bounds
        self.c_n_min   = float(c_n_min)
        self.c_t_min   = float(c_t_min)
        self.c_rot_min = float(c_rot_min)
        self.slip_gain = float(slip_gain)

        # Gripper and temperature
        self.gripper_bias_max = float(gripper_bias_max)
        self.temperature = float(temperature)

        # Small learned task token to bias features toward tactile semantics
        self.task_token = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # Trunk with linearly increasing droppath
        blocks = []
        for i in range(depth):
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

        # Base action heads
        self.head_trans = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),  # dx, dy, dz (pre-squash logits)
        )
        self.head_rot = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),  # droll, dpitch, dyaw (pre-squash logits)
        )
        self.head_grip = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),  # gripper (pre-squash)
        )

        # Contact-aware modulation heads
        self.head_contact_n = nn.Sequential(  # contact normal (unit 3D vector)
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.head_c_n = nn.Sequential(        # normal compliance in [c_n_min, 1]
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.head_c_t = nn.Sequential(        # tangential compliance in [c_t_min, 1]
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.head_c_rot = nn.Sequential(      # rotation damping in [c_rot_min, 1]
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.head_slip = nn.Sequential(       # slip risk ∈ [0,1]
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.head_grip_bias = nn.Sequential(  # gripper bias ∈ [-bias, +bias]
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------------------
    # Squashers to physical range
    # ---------------------------
    def _squash_translation(self, t_raw: torch.Tensor) -> torch.Tensor:
        return torch.tanh(t_raw) * self.max_trans  # [B,3] meters

    def _squash_rotation(self, r_raw: torch.Tensor) -> torch.Tensor:
        return torch.tanh(r_raw) * self.max_rot    # [B,3] radians

    def _squash_gripper(self, g_raw: torch.Tensor) -> torch.Tensor:
        # symmetric sigmoid to [-1, 1]
        return 2.0 * torch.sigmoid(g_raw) - 1.0    # [B,1]

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] fused latent (already encodes tactile/proprio modalities)

        Returns:
            action: [B, action_dim] = [dx, dy, dz, droll, dpitch, dyaw, gripper, ...]
        """
        B, D = x.shape
        h = x + self.task_token.expand(B, -1)
        h = self.trunk(h)

        # Base (pre-modulation) actions
        t_raw = self.head_trans(h)  # [B,3]
        r_raw = self.head_rot(h)    # [B,3]
        g_raw = self.head_grip(h)   # [B,1]

        # Physical squashing
        t = self._squash_translation(t_raw)
        r = self._squash_rotation(r_raw)
        g = self._squash_gripper(g_raw)

        # Contact-normal prediction and normalization
        n_raw = self.head_contact_n(h)                # [B,3]
        n = torch.nn.functional.normalize(n_raw, dim=-1, eps=1e-6)  # unit vector

        # Decompose translation into normal/tangential components
        # t_parallel = (t·n) n ; t_tangent = t - t_parallel
        dot = (t * n).sum(dim=-1, keepdim=True)      # [B,1]
        t_parallel = dot * n                          # [B,3]
        t_tangent = t - t_parallel                    # [B,3]

        # Compliance/damping scalars
        c_n   = self.head_c_n(h)                      # [B,1] in (0,1)
        c_t   = self.head_c_t(h)                      # [B,1] in (0,1)
        c_rot = self.head_c_rot(h)                    # [B,1] in (0,1)
        slip  = self.head_slip(h)                     # [B,1] in [0,1]

        # Map to [c_min, 1]
        c_n   = self.c_n_min   + (1.0 - self.c_n_min)   * c_n
        c_t   = self.c_t_min   + (1.0 - self.c_t_min)   * c_t
        c_rot = self.c_rot_min + (1.0 - self.c_rot_min) * c_rot

        # Reduce tangential motion with slip risk
        # effective c_t = c_t * (1 - slip_gain * slip)
        c_t_eff = torch.clamp(c_t * (1.0 - self.slip_gain * slip), min=0.0, max=1.0)

        # Apply modulation
        t_mod = c_n * t_parallel + c_t_eff * t_tangent  # [B,3]
        r_mod = c_rot * r                                # [B,3]

        # Gripper bias (scaled tanh -> [-bias, +bias])
        g_bias = self.gripper_bias_max * self.head_grip_bias(h)  # [B,1]
        g_mod = torch.clamp(g + g_bias, -1.0, 1.0)

        # Temperature scaling
        out7 = torch.cat([t_mod, r_mod, g_mod], dim=-1) * self.temperature  # [B,7]

        # Pad to action_dim if needed
        if self.action_dim > 7:
            pad = torch.zeros(B, self.action_dim - 7, device=x.device, dtype=x.dtype)
            out7 = torch.cat([out7, pad], dim=-1)
        return out7
