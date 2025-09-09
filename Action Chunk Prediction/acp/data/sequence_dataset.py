# acp/data/sequence_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    """Loads preprocessed sequences of actions/features: dict with 'seq' and 'boundary' targets."""
    def __init__(self, tensor_path: str):
        data = torch.load(tensor_path)
        self.seqs = data['seqs']            # [N, T, D]
        self.boundaries = data['boundaries']# [N, T] (0/1)
    def __len__(self): return self.seqs.size(0)
    def __getitem__(self, i):
        return {"seq": self.seqs[i], "boundary": self.boundaries[i]}
