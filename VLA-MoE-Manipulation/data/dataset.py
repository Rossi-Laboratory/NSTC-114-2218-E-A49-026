import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

def _stable_hash(token: str) -> int:
    # Simple deterministic hash (FNV-1a 32-bit style)
    h = 2166136261
    for c in token:
        h ^= ord(c)
        h = (h * 16777619) & 0xFFFFFFFF
    return h

def tokenize(text: str, vocab_size: int, max_len: int):
    tokens = text.lower().strip().split()
    ids = []
    for t in tokens[:max_len]:
        ids.append(_stable_hash(t) % vocab_size)
    # pad
    while len(ids) < max_len:
        ids.append(0)
    return torch.tensor(ids, dtype=torch.long)

class SyntheticManipulationDataset(Dataset):
    """A tiny synthetic dataset that emits random (but structured) samples.    Each sample:
      - image: 3xHxW
      - text_ids: max_len
      - proprio: proprio_dim
      - tactile: tactile_dim
      - action: action_dim
    """
    def __init__(self, num_samples=1024, image_size=128, vocab_size=512, max_text_len=16,
                 proprio_dim=7, tactile_dim=4, action_dim=7):
        self.num_samples = num_samples
        self.image_size = image_size
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.proprio_dim = proprio_dim
        self.tactile_dim = tactile_dim
        self.action_dim = action_dim

        # tiny prompt bank
        self.prompts = [
            "pick up the red block",
            "place the green cube in the box",
            "rotate the screw clockwise",
            "insert the peg into the hole",
            "grasp the blue cylinder",
            "place the yellow brick on the stack",
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        H = W = self.image_size
        # random image
        image = torch.rand(3, H, W)
        # choose a random command
        cmd = self.prompts[idx % len(self.prompts)]
        text_ids = tokenize(cmd, self.vocab_size, self.max_text_len)
        # proprio & tactile
        proprio = torch.randn(self.proprio_dim)
        tactile = torch.randn(self.tactile_dim)
        # synth action is correlated with hashed text for toy supervision
        rng = np.random.default_rng(seed=idx)
        base = (np.array([_stable_hash(w) for w in cmd.split()]) % 13).mean() / 13.0
        action = torch.tensor(rng.normal(loc=base, scale=0.25, size=(self.action_dim,)), dtype=torch.float32)
        action = action.clamp(-1.0, 1.0)
        sample = {
            "image": image,
            "text_ids": text_ids,
            "proprio": proprio,
            "tactile": tactile,
            "action": action,
            "command": cmd,
        }
        return sample
