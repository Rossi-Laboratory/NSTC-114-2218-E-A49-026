# acp/training/losses.py
"""
Loss Functions for Action Chunk Prediction (ACP)
================================================

This module implements loss functions used to train ACP models:
- Boundary BCE loss
- VQ quantization losses (commitment, codebook)
- Reconstruction loss (optional, if ground-truth actions available)
- Diversity regularization for codebook usage
"""

from typing import Dict, Optional
import torch
import torch.nn.functional as F


# -------------------------
# Boundary loss
# -------------------------
def boundary_bce_loss(boundary_logits: torch.Tensor, boundary_targets: torch.Tensor) -> torch.Tensor:
    """
    Binary cross entropy loss for boundary prediction.

    Args:
        boundary_logits: [B,T] raw logits from boundary head
        boundary_targets: [B,T] {0,1} ground truth boundaries

    Returns:
        scalar tensor
    """
    return F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets.float())


# -------------------------
# VQ losses
# -------------------------
def vq_loss(vq_loss_value: torch.Tensor, lambda_vq: float = 1.0) -> torch.Tensor:
    """
    Weight the vector quantization loss.

    Args:
        vq_loss_value: scalar from VQ forward pass
        lambda_vq: weight

    Returns:
        scalar tensor
    """
    return lambda_vq * vq_loss_value


# -------------------------
# Reconstruction loss
# -------------------------
def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mode: str = "mse") -> torch.Tensor:
    """
    Reconstruction loss for decoded actions.

    Args:
        pred: [B,T,A] predicted actions
        target: [B,T,A] ground truth actions
        mode: "mse" | "l1" | "smoothl1"

    Returns:
        scalar tensor
    """
    if mode == "mse":
        return F.mse_loss(pred, target)
    elif mode == "l1":
        return F.l1_loss(pred, target)
    elif mode == "smoothl1":
        return F.smooth_l1_loss(pred, target, beta=0.1)
    else:
        raise ValueError(f"Unknown mode={mode}")


# -------------------------
# Diversity regularizer
# -------------------------
def diversity_loss(code_hist: torch.Tensor, num_codes: int, lambda_div: float = 1e-3) -> torch.Tensor:
    """
    Encourage codebook diversity: penalize imbalance.

    Args:
        code_hist: [K] histogram of code usage
        num_codes: number of codes in codebook
        lambda_div: weight

    Returns:
        scalar tensor
    """
    probs = code_hist / (code_hist.sum() + 1e-6)
    entropy = -(probs * (probs + 1e-6).log()).sum()
    max_entropy = torch.log(torch.tensor(float(num_codes)))
    return lambda_div * (max_entropy - entropy)  # penalize low entropy


# -------------------------
# Loss aggregator
# -------------------------
def aggregate_losses(outputs: Dict, targets: Dict, cfg: Dict) -> Dict[str, torch.Tensor]:
    """
    Aggregate multiple losses based on config.

    Args:
        outputs: dict from model forward
            - "boundary_logits": [B,T]
            - "vq_loss": scalar
            - (optional) "recon": [B,T,A]
        targets: dict
            - "boundary": [B,T]
            - (optional) "actions": [B,T,A]
            - (optional) "code_hist": [K]
        cfg: loss weights, e.g.,
            {
              "lambda_boundary": 1.0,
              "lambda_vq": 1.0,
              "lambda_recon": 1.0,
              "lambda_div": 1e-3
            }

    Returns:
        dict with individual and total loss
    """
    logs = {}
    total = 0.0

    # Boundary loss
    if "boundary_logits" in outputs and "boundary" in targets:
        b_loss = boundary_bce_loss(outputs["boundary_logits"], targets["boundary"])
        logs["loss_boundary"] = b_loss
        total = total + cfg.get("lambda_boundary", 1.0) * b_loss

    # VQ loss
    if "vq_loss" in outputs:
        vq_l = vq_loss(outputs["vq_loss"], cfg.get("lambda_vq", 1.0))
        logs["loss_vq"] = vq_l
        total = total + vq_l

    # Reconstruction loss
    if "recon" in outputs and "actions" in targets:
        r_loss = reconstruction_loss(outputs["recon"], targets["actions"], mode=cfg.get("recon_mode", "mse"))
        logs["loss_recon"] = r_loss
        total = total + cfg.get("lambda_recon", 1.0) * r_loss

    # Diversity regularizer
    if "code_hist" in targets:
        d_loss = diversity_loss(targets["code_hist"], num_codes=len(targets["code_hist"]),
                                lambda_div=cfg.get("lambda_div", 1e-3))
        logs["loss_div"] = d_loss
        total = total + d_loss

    logs["loss_total"] = total
    return logs
