# models/experts/tooluse_expert.py
"""
TooluseExpert (enhanced)
------------------------
Expert head for manipulation tasks involving tool use
(e.g., rotation of a screwdriver, insertion of a peg, cutting, levering).

Highlights
- Residual MLP trunk with LayerScale + DropPath.
- Predicts base 7D action (translation, rotation, gripper).
- Tool-specific modulation factors:
  * force_gain     : scales translation magnitude
  * insertion_gain : biases axial translation (dz)
  * torque_bias    : biases rotation (yaw)
- Physical constraints: per-axis max translation (meters) and max rotation (radians).
- Output aligned with [dx, dy, dz, droll, dpitch, dyaw, gripper].

Input:
    x: [B, D] fused latent features

Output:
    action: [B, action_dim]
"""

from typing import Tuple
import torch
import torch.nn as nn


# ---------------------------
# Building blocks
# ---------------------------
class DropPath(nn.Module):
    """Stochastic depth (identity at eval)."""
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
    """Residual MLP with LayerNorm, GELU, LayerScale, and DropPath."""
    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        drop: float = 0.0,
        drop_path: float = 0.0,
        layerscale_init: float = 1e-3,
    ):
        super().__init__()
        hidden = d_model * hidden_mult
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.gamma = nn.Parameter(layerscale_init * torch.ones(d_model))
        self.droppath = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        h = self.dropout(h)
        h = self.droppath(self.gamma * h)
        return x + h


# ---------------------------
# Expert head
# ---------------------------
class TooluseExpert(nn.Module):
    """
    Args:
        d_model: fused latent feature dim.
        action_dim: >=7.
        depth: trunk depth.
        hidden_mult: expansion factor.
        dropout: dropout prob.
        drop_path: stochastic depth prob.
        max_trans: per-axis translation limit (m).
        max_rot: per-axis rotation limit (rad).
        force_gain_max: scaling factor for force application.
        insertion_bias_max: axial translation bias (dz).
        torque_bias_max: yaw torque bias (dyaw).
        temperature: global multiplier.
    """
    def __init__(
        self,
        d_model: int,
        action_dim: int,
        depth: int = 4,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.1,
        max_trans: Tuple[float, float, float] = (0.04, 0.04, 0.06),   # tool may allow larger dz
        max_rot:   Tuple[float, float, float] = (0.35, 0.35, 0.5),    # ~20-30 deg
        force_gain_max: float = 1.5,
        insertion_bias_max: float = 0.02,
        torque_bias_max: float = 0.15,
        temperature: float = 1.0,
    ):
        super().__init__()
        assert action_dim >= 7, "TooluseExpert expects action_dim >= 7."
        self.action_dim = action_dim

        self.register_buffer("max_trans", torch.tensor(max_trans, dtype=torch.float32))
        self.register_buffer("max_rot",   torch.tensor(max_rot,   dtype=torch.float32))
        self.force_gain_max = float(force_gain_max)
        self.insertion_bias_max = float(insertion_bias_max)
        self.torque_bias_max = float(torque_bias_max)
        self.temperature = float(temperature)

        # task token
        self.task_token = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # trunk
        blocks = []
        for i in range(depth):
            dp = drop_path * float(i) / max(1, depth - 1)
            blocks.append(ResidualMLPBlock(d_model, hidden_mult, dropout, dp))
        self.trunk = nn.Sequential(*blocks)

        # base heads
        self.head_trans = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
            nn.GELU(), nn.Linear(d_model, 3),
        )
        self.head_rot = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
            nn.GELU(), nn.Linear(d_model, 3),
        )
        self.head_grip = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, 1),
        )

        # modulation heads
        self.head_force = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, 1), nn.Sigmoid()
        )
        self.head_insert = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, 1), nn.Tanh()
        )
        self.head_torque = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, 1), nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _squash_translation(self, t_raw: torch.Tensor) -> torch.Tensor:
        return torch.tanh(t_raw) * self.max_trans

    def _squash_rotation(self, r_raw: torch.Tensor) -> torch.Tensor:
        return torch.tanh(r_raw) * self.max_rot

    def _squash_gripper(self, g_raw: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sigmoid(g_raw) - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        h = x + self.task_token.expand(B, -1)
        h = self.trunk(h)

        # base actions
        t = self._squash_translation(self.head_trans(h))  # [B,3]
        r = self._squash_rotation(self.head_rot(h))       # [B,3]
        g = self._squash_gripper(self.head_grip(h))       # [B,1]

        # tool-use modulation
        f_gain = 1.0 + self.force_gain_max * self.head_force(h)    # [B,1] in [1, 1+max]
        insert = self.insertion_bias_max * self.head_insert(h)     # [B,1] bias along dz
        torque = self.torque_bias_max * self.head_torque(h)        # [B,1] bias to yaw

        t = t * f_gain
        t[:, 2:3] = t[:, 2:3] + insert   # dz bias
        r[:, 2:3] = r[:, 2:3] + torque   # yaw bias

        out7 = torch.cat([t, r, g], dim=-1) * self.temperature

        if self.action_dim > 7:
            pad = torch.zeros(B, self.action_dim - 7, device=x.device, dtype=x.dtype)
            out7 = torch.cat([out7, pad], dim=-1)
        return out7
