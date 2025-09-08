# models/experts/grasping_expert.py
"""
GraspingExpert (enhanced)
-------------------------
Expert head specialized for grasping objects.

Design
- Residual MLP trunk with LayerScale, DropPath, and SE gating.
- Multi-head outputs:
  * Translation (dx, dy, dz)  -> bounded by max_trans
  * Rotation    (droll, dpitch, dyaw) -> bounded by max_rot
  * Gripper     (open/close) in [-1,1]
- Additional modulation:
  * Force scale (grip strength) in [min_force, 1]
  * Optional approach bias toward -z (downward) to stabilize approach
- Output order: [dx, dy, dz, droll, dpitch, dyaw, gripper]

Input
    x: [B, D] fused latent features
Output
    action: [B, action_dim] (>=7)
"""

from typing import Tuple
import torch
import torch.nn as nn


# ---------------------------
# Building blocks
# ---------------------------
class DropPath(nn.Module):
    """Stochastic depth; identity during eval."""
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
    """Residual MLP with LayerNorm, GELU, SE gating, LayerScale, and DropPath."""
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
class GraspingExpert(nn.Module):
    """
    Args:
        d_model: latent feature dim
        action_dim: >=7
        depth: trunk depth
        hidden_mult: MLP expansion
        dropout: dropout prob
        drop_path: stochastic depth prob
        max_trans: per-axis max translation (m)
        max_rot: per-axis max rotation (rad)
        min_force: min gripper force scale
        approach_bias: bias added to dz (negative means downward approach)
        temperature: global multiplier
    """
    def __init__(
        self,
        d_model: int,
        action_dim: int,
        depth: int = 4,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.1,
        max_trans: Tuple[float, float, float] = (0.05, 0.05, 0.08),  # up to 8 cm in z
        max_rot: Tuple[float, float, float] = (0.3, 0.3, 0.3),       # ~17 deg
        min_force: float = 0.3,
        approach_bias: float = -0.01,
        temperature: float = 1.0,
    ):
        super().__init__()
        assert action_dim >= 7, "GraspingExpert expects action_dim >= 7."
        self.action_dim = action_dim

        self.register_buffer("max_trans", torch.tensor(max_trans, dtype=torch.float32))
        self.register_buffer("max_rot",   torch.tensor(max_rot,   dtype=torch.float32))

        self.min_force = float(min_force)
        self.approach_bias = float(approach_bias)
        self.temperature = float(temperature)

        # task token
        self.task_token = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # trunk
        blocks = []
        for i in range(depth):
            dp = drop_path * float(i) / max(1, depth - 1)
            blocks.append(
                ResidualMLPBlock(d_model, hidden_mult, dropout, dp)
            )
        self.trunk = nn.Sequential(*blocks)

        # heads
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
        self.head_force = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),  # [0,1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # squashers
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

        # base predictions
        t = self._squash_translation(self.head_trans(h))
        r = self._squash_rotation(self.head_rot(h))
        g = self._squash_gripper(self.head_grip(h))

        # apply approach bias on dz
        t = t.clone()
        t[:, 2:3] = t[:, 2:3] + self.approach_bias
        t[:, 2:3] = torch.clamp(t[:, 2:3], -self.max_trans[2], self.max_trans[2])

        # force modulation
        f = self.head_force(h)  # [0,1]
        f = self.min_force + (1.0 - self.min_force) * f
        g = g * f  # modulate gripper output

        out7 = torch.cat([t, r, g], dim=-1) * self.temperature

        # pad if action_dim > 7
        if self.action_dim > 7:
            pad = torch.zeros(B, self.action_dim - 7, device=x.device, dtype=x.dtype)
            out7 = torch.cat([out7, pad], dim=-1)
        return out7
