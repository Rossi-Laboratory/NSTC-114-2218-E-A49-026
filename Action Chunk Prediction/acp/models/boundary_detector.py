# acp/models/boundary_detector.py
import torch
import torch.nn as nn

class BoundaryDetector(nn.Module):
    """Simple boundary classifier: [B,T,D] -> logits [B,T]."""
    def __init__(self, d_model=256, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: [B,T,D]
        return self.net(x).squeeze(-1)
