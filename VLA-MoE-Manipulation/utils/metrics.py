# utils/metrics.py
"""
Metrics utilities for VLA-MoE Manipulation
------------------------------------------

Includes:
- compute_success_rate : binary success based on L2 distance threshold
- compute_mse          : mean squared error
- compute_rmse         : root mean squared error
- compute_mae          : mean absolute error
- compute_r2           : coefficient of determination
- normalized_error     : per-dimension normalized error
- MetricTracker        : helper to accumulate and average metrics
"""

from typing import Dict
import numpy as np
import torch


# ----------------------------
# Basic metrics
# ----------------------------
def compute_success_rate(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.25) -> float:
    """
    Success if L2 distance < thresh.
    pred   : [B, A]
    target : [B, A]
    """
    diff = torch.norm(pred - target, dim=-1)
    return float((diff < thresh).float().mean().item())


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).item())


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).item())


def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    R² = 1 - SSE/SST
    """
    target_mean = torch.mean(target)
    sse = torch.sum((pred - target) ** 2)
    sst = torch.sum((target - target_mean) ** 2)
    r2 = 1.0 - sse / (sst + 1e-8)
    return float(r2.item())


def normalized_error(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """
    Dimension-wise normalized error: |pred - target| / (|target| + eps).
    Returns numpy array [A].
    """
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    eps = 1e-6
    return np.mean(np.abs(p - t) / (np.abs(t) + eps), axis=0)


# ----------------------------
# Metric tracker
# ----------------------------
class MetricTracker:
    """
    Tracks and averages multiple metrics over time.
    Example:
        tracker = MetricTracker("loss", "success", "mse")
        tracker.update(loss=0.3, success=0.8, mse=0.1)
        avg = tracker.avg()  # dict with averages
    """
    def __init__(self, *keys: str):
        self.data = {k: [] for k in keys}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.data:
                self.data[k].append(float(v))

    def avg(self) -> Dict[str, float]:
        return {k: (sum(vs) / len(vs) if len(vs) > 0 else 0.0) for k, vs in self.data.items()}

    def latest(self) -> Dict[str, float]:
        return {k: (vs[-1] if len(vs) > 0 else 0.0) for k, vs in self.data.items()}

    def reset(self):
        for k in self.data.keys():
            self.data[k] = {}

    def __str__(self):
        avg_dict = self.avg()
        return " | ".join([f"{k}: {v:.4f}" for k, v in avg_dict.items()])
