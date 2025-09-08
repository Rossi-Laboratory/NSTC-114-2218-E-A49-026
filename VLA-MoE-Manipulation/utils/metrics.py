import torch

def rmse(pred: torch.Tensor, target: torch.Tensor):
    return torch.sqrt(((pred - target) ** 2).mean())
