# acp/utils/checkpoint.py
import torch, os
def save(path, **state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
def load(path, map_location='cpu'):
    return torch.load(path, map_location=map_location)
