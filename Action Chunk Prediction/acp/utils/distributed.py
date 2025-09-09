# acp/utils/distributed.py
"""
Distributed Training Utilities
==============================

Helpers for multi-GPU training with PyTorch DistributedDataParallel (DDP).
"""

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler


# -------------------------
# Initialization
# -------------------------
def init_distributed(backend: str = "nccl", port: str = "12355"):
    """
    Initialize torch.distributed if environment variables are set.

    Typical launch:
        torchrun --nproc_per_node=4 acp/training/train.py --config configs/model_acp.yaml

    Args:
        backend: "nccl" | "gloo"
        port: TCP port for rendezvous
    """
    if dist.is_initialized():
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", port)

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank % torch.cuda.device_count())
        print(f"[DDP] Initialized (rank={rank}, world_size={world_size})")
    else:
        print("[DDP] Environment variables not set. Running in single-process mode.")


def cleanup_distributed():
    """Destroy the process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


# -------------------------
# Rank / World size helpers
# -------------------------
def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


# -------------------------
# Sampler / DataLoader wrapper
# -------------------------
def get_dataloader(dataset, batch_size: int, shuffle: bool = True, num_workers: int = 4, pin_memory: bool = True):
    """
    Wrap dataset with DistributedSampler if in DDP mode.

    Args:
        dataset: torch Dataset
        batch_size: int
        shuffle: bool
        num_workers: int
        pin_memory: bool

    Returns:
        DataLoader
    """
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=pin_memory)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)


# -------------------------
# Utility wrappers
# -------------------------
def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(t: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    All-reduce a tensor across processes.

    Args:
        t: torch.Tensor
        average: whether to divide by world_size

    Returns:
        reduced tensor (on all ranks)
    """
    if dist.is_initialized():
        rt = t.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        if average:
            rt /= get_world_size()
        return rt
    return t
