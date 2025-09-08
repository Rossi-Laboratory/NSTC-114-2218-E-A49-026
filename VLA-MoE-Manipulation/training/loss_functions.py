import torch
import torch.nn.functional as F

def action_mse_loss(pred: torch.Tensor, target: torch.Tensor):
    return F.mse_loss(pred, target)

def load_balance_loss(weights: torch.Tensor):
    """Encourage diverse expert usage across the batch by maximizing entropy
    of the average routing distribution.
    """
    avg_w = weights.mean(dim=0)  # [E]
    entropy = - (avg_w * (avg_w.clamp(min=1e-8).log())).sum()
    # We want higher entropy -> lower loss
    return -entropy
