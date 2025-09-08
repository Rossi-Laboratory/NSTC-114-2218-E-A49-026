# training/loss_functions.py
"""
Loss functions for VLA-MoE Manipulation
---------------------------------------

Includes:
- action_loss      : regression loss on predicted vs target actions
- moe_balance_loss : load balancing loss for MoE experts
- moe_entropy_loss : encourages sparse expert selection
- total_loss       : weighted sum of all components
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def action_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "l2",
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Action regression loss.

    Args:
        pred   : [B, A] predicted actions
        target : [B, A] ground truth actions
        loss_type: "l2" (default), "l1", or "huber"
        mask   : [B, A] optional mask (1=include, 0=ignore)

    Returns:
        loss (scalar tensor)
    """
    if mask is not None:
        pred = pred * mask
        target = target * mask

    if loss_type == "l2":
        loss = F.mse_loss(pred, target)
    elif loss_type == "l1":
        loss = F.l1_loss(pred, target)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(pred, target)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss


def moe_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """
    Load balancing loss for Mixture-of-Experts.
    Encourages equal expert usage.

    Args:
        weights : [B, E] softmax weights over experts

    Returns:
        scalar tensor
    """
    # mean weight per expert
    expert_mean = weights.mean(dim=0)  # [E]
    # target is uniform distribution
    uniform = torch.full_like(expert_mean, 1.0 / weights.size(1))
    return F.mse_loss(expert_mean, uniform)


def moe_entropy_loss(weights: torch.Tensor) -> torch.Tensor:
    """
    Entropy regularization for MoE weights.
    Encourages sparsity (low entropy).

    Args:
        weights : [B, E] softmax weights

    Returns:
        scalar tensor
    """
    eps = 1e-8
    ent = - (weights * (weights + eps).log()).sum(dim=1)  # [B]
    return ent.mean()


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    cfg: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combine all loss terms.

    Args:
        pred   : [B, A] action predictions
        target : [B, A] ground truth actions
        weights: [B, E] expert mixture weights
        cfg    : dict with weightings, e.g.
                 {
                   "lambda_action": 1.0,
                   "lambda_balance": 0.01,
                   "lambda_entropy": 0.001,
                   "loss_type": "l2"
                 }

    Returns:
        (total_loss, log_dict)
    """
    lam_action = cfg.get("lambda_action", 1.0)
    lam_balance = cfg.get("lambda_balance", 0.0)
    lam_entropy = cfg.get("lambda_entropy", 0.0)
    loss_type = cfg.get("loss_type", "l2")

    l_action = action_loss(pred, target, loss_type=loss_type)
    l_balance = moe_balance_loss(weights)
    l_entropy = moe_entropy_loss(weights)

    total = lam_action * l_action + lam_balance * l_balance + lam_entropy * l_entropy

    logs = {
        "loss_total": float(total.item()),
        "loss_action": float(l_action.item()),
        "loss_balance": float(l_balance.item()),
        "loss_entropy": float(l_entropy.item()),
    }
    return total, logs
