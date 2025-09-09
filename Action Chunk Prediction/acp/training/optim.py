# acp/training/optim.py
"""
Optimizer & Scheduler Builders
==============================

This module provides utilities to build optimizers and learning rate
schedulers from a config dictionary. It supports commonly used choices
for Action Chunk Prediction (ACP).

Usage
-----
cfg = {
    "optim": {
        "type": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2
    },
    "scheduler": {
        "type": "cosine",
        "warmup_steps": 500,
        "total_steps": 10000
    }
}

optim = build_optimizer(model, cfg["optim"])
sched = build_scheduler(optim, cfg["scheduler"])
"""

from typing import Dict, Optional
import torch
from torch.optim import Optimizer


# -------------------------
# Optimizer
# -------------------------
def build_optimizer(model: torch.nn.Module, cfg: Dict) -> Optimizer:
    """
    Build optimizer from config.

    Args:
        model: nn.Module
        cfg: dict with keys:
            - type: "adam" | "adamw" | "sgd"
            - lr: float
            - weight_decay: float
            - momentum: float (for SGD)

    Returns:
        torch.optim.Optimizer
    """
    opt_type = cfg.get("type", "adamw").lower()
    lr = cfg.get("lr", 3e-4)
    wd = cfg.get("weight_decay", 1e-2)

    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "sgd":
        momentum = cfg.get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type={opt_type}")


# -------------------------
# Scheduler
# -------------------------
def build_scheduler(optim: Optimizer, cfg: Optional[Dict]):
    """
    Build learning rate scheduler from config.

    Args:
        optim: torch optimizer
        cfg: dict with keys:
            - type: "step" | "cosine" | "onecycle" | None
            - step_size: int
            - gamma: float
            - warmup_steps: int
            - total_steps: int

    Returns:
        torch.optim.lr_scheduler or None
    """
    if cfg is None:
        return None

    sched_type = cfg.get("type", None)
    if sched_type is None:
        return None

    sched_type = sched_type.lower()

    if sched_type == "step":
        step_size = cfg.get("step_size", 10)
        gamma = cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)

    elif sched_type == "cosine":
        T = cfg.get("total_steps", 1000)
        warmup = cfg.get("warmup_steps", 0)

        def lr_lambda(step: int):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, T - warmup))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))

        return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    elif sched_type == "onecycle":
        max_lr = cfg.get("max_lr", 1e-3)
        total = cfg.get("total_steps", 1000)
        return torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=max_lr, total_steps=total)

    else:
        raise ValueError(f"Unknown scheduler type={sched_type}")


# -------------------------
# Example convenience wrapper
# -------------------------
def build_optim_and_sched(model: torch.nn.Module, cfg: Dict):
    """
    Convenience function to build both optimizer and scheduler.

    Args:
        model: nn.Module
        cfg: dict with subkeys "optim", "scheduler"

    Returns:
        (optimizer, scheduler)
    """
    optim_cfg = cfg.get("optim", {"type": "adamw", "lr": 3e-4})
    sched_cfg = cfg.get("scheduler", None)
    optim = build_optimizer(model, optim_cfg)
    sched = build_scheduler(optim, sched_cfg)
    return optim, sched
