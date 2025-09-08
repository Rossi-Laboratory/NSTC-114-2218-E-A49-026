# training/__init__.py
"""
Training package for VLA-MoE Manipulation.

This module exposes:
- Trainer class (training loop wrapper)
- train function (convenient entry point)
- common loss functions
"""

from .trainer import Trainer, train
from .loss_functions import moe_loss, action_loss

__all__ = [
    "Trainer",
    "train",
    "moe_loss",
    "action_loss",
]
