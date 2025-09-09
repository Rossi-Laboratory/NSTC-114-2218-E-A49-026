# acp/models/vq/codebook.py
"""
Codebook Utilities for VQ
=========================

This module defines a reusable codebook class for vector quantization.
It supports both:
- Directly learnable embeddings (optimized by gradient).
- Exponential Moving Average (EMA) updates for stability.

Usage
-----
codebook = Codebook(num_codes=128, dim=256, ema=True)

z_q, indices, loss = codebook(z)   # z: [B,N,D]
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(nn.Module):
    def __init__(self, num_codes: int, dim: int, ema: bool = True, decay: float = 0.99, eps: float = 1e-5):
        """
        Args:
            num_codes: number of discrete codes in the codebook
            dim: dimensionality of each code vector
            ema: whether to use EMA updates (True = EMA-VQ, False = learnable embeddings)
            decay: EMA decay factor
            eps: numerical stability
        """
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.ema = ema
        self.decay = decay
        self.eps = eps

        # Initialize codebook
        embed = torch.randn(num_codes, dim) / dim**0.5
        self.register_buffer("embed", embed)  # [K,D]

        if ema:
            self.register_buffer("cluster_size", torch.zeros(num_codes))
            self.register_buffer("embed_avg", embed.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input z with nearest neighbor lookup.

        Args:
            z: [B,N,D] input features

        Returns:
            z_q: [B,N,D] quantized embeddings (with straight-through)
            indices: [B,N] chosen code indices
            loss: scalar quantization loss
        """
        B, N, D = z.shape
        flat = z.reshape(B * N, D)  # [BN,D]

        # Compute distances to codebook
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.t()
            + self.embed.pow(2).sum(1, keepdim=True).t()
        )  # [BN,K]

        indices = torch.argmin(dist, dim=1)  # [BN]
        z_q = self.embed[indices].view(B, N, D)

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # Loss = commitment + codebook
        commitment = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = commitment + codebook_loss

        if self.ema and self.training:
            # EMA updates
            onehot = F.one_hot(indices, self.num_codes).type_as(z)  # [BN,K]
            cluster_size = onehot.sum(0)  # [K]
            embed_sum = flat.t() @ onehot  # [D,K]

            # Update buffers
            self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self.embed_avg.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)

            # Normalize
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.copy_(embed_normalized)

        return z_q_st, indices.view(B, N), loss

    def get_codebook(self) -> torch.Tensor:
        """Return the current codebook [K,D]."""
        return self.embed.detach().clone()

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings by indices [..] -> [..,D]."""
        return self.embed[indices]
