# acp/models/vq/quantizer_utils.py
"""
Quantizer Utilities
===================

Helper functions for vector quantization (VQ).
These utilities provide:
- Distance computations (L2, cosine)
- Codebook initialization (uniform, kmeans)
- Code usage analysis (entropy, diversity)
- Straight-through estimator helper
- Optional visualization helpers
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F


# -------------------------
# Distance metrics
# -------------------------
def l2_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise L2 distance between x and y.

    Args:
        x: [N,D]
        y: [M,D]

    Returns:
        dist: [N,M]
    """
    return (
        x.pow(2).sum(1, keepdim=True)
        - 2 * x @ y.t()
        + y.pow(2).sum(1, keepdim=True).t()
    )


def cosine_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute pairwise cosine distance = 1 - cosine similarity.

    Args:
        x: [N,D]
        y: [M,D]

    Returns:
        dist: [N,M]
    """
    x_norm = F.normalize(x, dim=-1, eps=eps)
    y_norm = F.normalize(y, dim=-1, eps=eps)
    return 1.0 - x_norm @ y_norm.t()


# -------------------------
# Codebook initialization
# -------------------------
def init_codebook_uniform(num_codes: int, dim: int, scale: float = 1.0) -> torch.Tensor:
    """
    Uniform random initialization [-scale, scale].

    Returns:
        [num_codes, dim] tensor
    """
    return torch.empty(num_codes, dim).uniform_(-scale, scale)


def init_codebook_kmeans(x: torch.Tensor, num_codes: int, iters: int = 10) -> torch.Tensor:
    """
    Simple KMeans initialization.

    Args:
        x: [N,D] data
        num_codes: K
        iters: number of iterations

    Returns:
        centroids: [K,D]
    """
    N, D = x.shape
    indices = torch.randperm(N)[:num_codes]
    centroids = x[indices]

    for _ in range(iters):
        dist = l2_distance(x, centroids)  # [N,K]
        assign = torch.argmin(dist, dim=1)  # [N]
        for k in range(num_codes):
            mask = assign == k
            if mask.sum() > 0:
                centroids[k] = x[mask].mean(0)
    return centroids


# -------------------------
# Code usage analysis
# -------------------------
def compute_code_usage(indices: torch.Tensor, num_codes: int) -> Tuple[torch.Tensor, float]:
    """
    Compute histogram and entropy of code usage.

    Args:
        indices: [B,N] code indices
        num_codes: total number of codes

    Returns:
        hist: [K] counts
        entropy: float, usage entropy
    """
    flat = indices.flatten()
    hist = torch.bincount(flat, minlength=num_codes).float()
    probs = hist / (hist.sum() + 1e-6)
    entropy = -(probs * (probs + 1e-6).log()).sum().item()
    return hist, entropy


def code_diversity(hist: torch.Tensor) -> float:
    """
    Compute diversity = fraction of codes used at least once.

    Args:
        hist: [K] counts

    Returns:
        float in [0,1]
    """
    return (hist > 0).float().mean().item()


# -------------------------
# Straight-through helper
# -------------------------
def straight_through(z_e: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    """
    Straight-through estimator: pass gradient of z_e while using values of z_q.

    Args:
        z_e: [B,N,D] original embeddings
        z_q: [B,N,D] quantized embeddings

    Returns:
        [B,N,D] tensor
    """
    return z_e + (z_q - z_e).detach()


# -------------------------
# Optional visualization
# -------------------------
def project_to_2d(x: torch.Tensor, method: str = "pca") -> torch.Tensor:
    """
    Project embeddings to 2D for visualization.

    Args:
        x: [N,D]
        method: "pca" | "tsne" (requires sklearn)

    Returns:
        [N,2] projected coords
    """
    x_np = x.detach().cpu().numpy()
    if method == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2).fit_transform(x_np)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, init="random").fit_transform(x_np)
    else:
        raise ValueError(f"Unknown method={method}")
    return torch.tensor(proj, dtype=torch.float32)
