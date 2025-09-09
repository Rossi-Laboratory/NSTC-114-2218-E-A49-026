# acp/models/vq/vector_quantizer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """Basic VQ layer with straight-through estimator."""
    def __init__(self, num_codes: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        self.codebook = nn.Embedding(num_codes, dim)
        nn.init.uniform_(self.codebook.weight, -1.0/dim, 1.0/dim)

    def forward(self, x):  # x: [B,N,D]
        B, N, D = x.shape
        flat = x.reshape(B*N, D)  # [BN,D]
        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1, keepdim=True).t()
        )  # [BN,K]
        indices = torch.argmin(distances, dim=1)  # [BN]
        z_q = self.codebook(indices).view(B, N, D)
        # commitment + codebook losses
        loss = F.mse_loss(z_q.detach(), x) + self.beta * F.mse_loss(z_q, x.detach())
        # straight-through
        z_q = x + (z_q - x).detach()
        return z_q, loss, indices.view(B, N)
