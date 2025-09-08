# inference/eval_manipulation.py
"""
Evaluation utilities for manipulation tasks with VLA-MoE.

Provides functions to compute:
- success rate
- mean squared error (MSE) on actions
- root mean squared error (RMSE)
- normalized distance error (per action dimension)
- top-k expert usage statistics
- aggregated evaluation over a dataset
"""

import numpy as np
import torch
from typing import Dict, List, Tuple


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error between actions."""
    return float(((pred - target) ** 2).mean())


def rmse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error between actions."""
    return float(np.sqrt(((pred - target) ** 2).mean()))


def success_rate(
    pred: np.ndarray, target: np.ndarray, thresh: float = 0.25
) -> float:
    """
    Binary success if L2 distance < thresh.
    pred: [N, A]
    target: [N, A]
    """
    diff = np.linalg.norm(pred - target, axis=-1)
    return float((diff < thresh).mean())


def normalized_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Dimension-wise normalized error: |pred - target| / (|target| + eps).
    Returns array of shape [A].
    """
    eps = 1e-6
    return np.mean(np.abs(pred - target) / (np.abs(target) + eps), axis=0)


def expert_usage(weights: np.ndarray, topk: int = 1) -> Dict[str, float]:
    """
    Compute expert usage distribution.

    Args:
        weights: [N, E], softmax probabilities
        topk: count an expert as "used" if in top-k

    Returns:
        dict mapping expert_index -> usage frequency
    """
    N, E = weights.shape
    top_idx = np.argsort(-weights, axis=1)[:, :topk]  # [N, topk]
    counts = np.bincount(top_idx.flatten(), minlength=E)
    usage = counts / counts.sum()
    return {f"expert_{i}": float(usage[i]) for i in range(E)}


def evaluate_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    thresh: float = 0.25,
) -> Dict[str, float]:
    """
    Evaluate one batch of predictions.

    Args:
        pred: [B, A]
        target: [B, A]
        weights: [B, E]

    Returns:
        dict of metrics
    """
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    w = weights.detach().cpu().numpy()

    metrics = {
        "mse": mse_loss(p, t),
        "rmse": rmse_loss(p, t),
        "success_rate": success_rate(p, t, thresh=thresh),
    }

    # dimension-wise normalized error
    norm_err = normalized_error(p, t)
    for i, v in enumerate(norm_err):
        metrics[f"norm_err_dim{i}"] = float(v)

    # expert usage
    usage = expert_usage(w, topk=1)
    for k, v in usage.items():
        metrics[f"usage_{k}"] = v

    return metrics


def aggregate_results(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate list of metric dicts by averaging values.
    """
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        agg[k] = float(np.mean(vals))
    return agg


if __name__ == "__main__":
    # quick self-test
    B, A, E = 8, 7, 4
    pred = torch.rand(B, A)
    target = torch.rand(B, A)
    weights = torch.softmax(torch.rand(B, E), dim=-1)

    metrics = evaluate_batch(pred, target, weights)
    print("Sample metrics:")
    for k, v in metrics.items():
        print(f"{k:15s}: {v:.4f}")
