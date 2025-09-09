# acp/training/scheduler.py
"""
Learning Rate Scheduler Utilities
=================================

This module provides various learning rate schedulers for training
Action Chunk Prediction (ACP). It includes:

- StepLR
- MultiStepLR
- Cosine Annealing (with optional warmup)
- OneCycleLR
- Linear Warmup (custom LambdaLR)

Usage
-----
cfg = {
    "scheduler": {
        "type": "cosine",
        "warmup_steps": 500,
        "total_steps": 10000
    }
}
scheduler = build_scheduler(optimizer, cfg["scheduler"])
"""

import math
from typing import Dict, Optional
import torch
from torch.optim import Optimizer


# -------------------------
# Warmup helpers
# -------------------------
def linear_warmup_lambda(current_step: int, warmup_steps: int):
    """Linear warmup from 0 → 1 during warmup_steps."""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0


def cosine_lambda(current_step: int, warmup_steps: int, total_steps: int):
    """Cosine decay after warmup."""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# -------------------------
# Scheduler builder
# -------------------------
def build_scheduler(optimizer: Optimizer, cfg: Optional[Dict]):
    """
    Build scheduler from config.

    Args:
        optimizer: torch optimizer
        cfg: dict with keys:
            - type: "step" | "multistep" | "cosine" | "onecycle" | "linear_warmup"
            - step_size: int
            - gamma: float
            - milestones: list[int]
            - warmup_steps: int
            - total_steps: int
            - max_lr: float (for onecycle)

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None
    """
    if cfg is None:
        return None

    s_type = cfg.get("type", None)
    if s_type is None:
        return None

    s_type = s_type.lower()

    if s_type == "step":
        step_size = cfg.get("step_size", 10)
        gamma = cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif s_type == "multistep":
        milestones = cfg.get("milestones", [30, 60, 90])
        gamma = cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif s_type == "cosine":
        warmup = cfg.get("warmup_steps", 0)
        total = cfg.get("total_steps", 1000)
        lr_lambda = lambda step: cosine_lambda(step, warmup, total)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif s_type == "linear_warmup":
        warmup = cfg.get("warmup_steps", 500)
        total = cfg.get("total_steps", 1000)
        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            return max(0.0, 1.0 - (step - warmup) / float(max(1, total - warmup)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif s_type == "onecycle":
        max_lr = cfg.get("max_lr", 1e-3)
        total = cfg.get("total_steps", 1000)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total,
            pct_start=cfg.get("pct_start", 0.3),
            anneal_strategy=cfg.get("anneal_strategy", "cos"),
        )

    else:
        raise ValueError(f"Unknown scheduler type={s_type}")
