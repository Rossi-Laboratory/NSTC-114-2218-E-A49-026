# data/__init__.py
"""
Data module entrypoint for VLA-MoE-Manipulation.

Exports:
- SyntheticManipulationDataset: toy dataset to validate the pipeline
- tokenize: text → token ids (stable hashing)
- build_dataset(cfg, split="train"): create a dataset from a config dict
- build_dataloader(cfg, batch_size, shuffle=True, num_workers=0, device="cpu"):
    convenience wrapper returning a PyTorch DataLoader
"""

from typing import Dict, Optional
from torch.utils.data import DataLoader

from .dataset import (
    SyntheticManipulationDataset,
    tokenize,
)

__all__ = [
    "SyntheticManipulationDataset",
    "tokenize",
    "build_dataset",
    "build_dataloader",
]


def _get_num_samples(cfg: Dict, split: str) -> int:
    """
    Helper to pick the right sample count by split.
    Prefer keys like num_samples_train / num_samples_val if present,
    otherwise fall back to num_samples.
    """
    if not isinstance(cfg, dict):
        return 0
    key_split = f"num_samples_{split}"
    if key_split in cfg:
        return int(cfg[key_split])
    return int(cfg.get("num_samples", 0))


def build_dataset(cfg: Dict, split: str = "train"):
    """
    Build a dataset from a config dict (e.g., configs/train.yaml -> cfg["dataset"]).

    Args:
        cfg: dict with keys like
             - name: "synthetic"
             - num_samples, image_size, vocab_size, max_text_len,
               proprio_dim, tactile_dim, action_dim
        split: "train" | "val" | "test" (affects num_samples picking)

    Returns:
        torch.utils.data.Dataset
    """
    if not isinstance(cfg, dict):
        raise ValueError("build_dataset expects a dict (e.g., cfg['dataset']).")

    name = str(cfg.get("name", "synthetic")).lower()

    if name == "synthetic":
        ds = SyntheticManipulationDataset(
            num_samples=_get_num_samples(cfg, split),
            image_size=int(cfg.get("image_size", 128)),
            vocab_size=int(cfg.get("vocab_size", 512)),
            max_text_len=int(cfg.get("max_text_len", 16)),
            proprio_dim=int(cfg.get("proprio_dim", 7)),
            tactile_dim=int(cfg.get("tactile_dim", 4)),
            action_dim=int(cfg.get("action_dim", 7)),
        )
        return ds

    raise ValueError(f"Unknown dataset name: {name!r}. Supported: 'synthetic'.")


def build_dataloader(
    cfg: Dict,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    """
    Convenience function to build a DataLoader given a dataset config.

    Example:
        ds_cfg = full_cfg['dataset']
        train_loader = build_dataloader(ds_cfg, batch_size=8, shuffle=True, num_workers=2)

    Args:
        cfg: dataset config dict (same fields as build_dataset)
        batch_size: per-iteration batch size
        shuffle: shuffle samples
        num_workers: DataLoader workers
        pin_memory: pin memory (useful for CUDA)
        drop_last: drop remainder batch

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = build_dataset(cfg, split="train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
