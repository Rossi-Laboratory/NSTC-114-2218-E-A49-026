# acp/models/fast/chunk_sampler.py
"""
FAST-style Chunk Sampler
========================

This module implements utilities for sampling subsequences ("chunks")
from a longer action sequence during training.

Why?
----
- Long trajectories (T ~ 100-1000 steps) are inefficient to feed entirely.
- FAST-style training samples *chunks* of variable length to expose the
  boundary detector and aggregator to diverse temporal contexts.
- Chunks can be aligned with predicted / ground truth boundaries, or
  sampled uniformly at random.

Key Functions
-------------
- sample_uniform_chunks: sample random subsequences of length L.
- sample_boundary_aligned: use boundary labels to sample chunks that
  begin/end at boundaries.
- batch_sample_chunks: convenience for a batch of sequences.
"""

from typing import List, Tuple, Dict
import torch


def sample_uniform_chunks(seq: torch.Tensor, L: int, num_chunks: int) -> List[torch.Tensor]:
    """
    Uniformly sample subsequences of fixed length L.

    Args:
        seq: [T, D] tensor
        L: int, chunk length
        num_chunks: number of subsequences to sample

    Returns:
        List of [L, D] tensors (padded/truncated if necessary)
    """
    T, D = seq.shape
    chunks = []
    for _ in range(num_chunks):
        if T <= L:
            # If too short, pad
            pad = torch.zeros(L - T, D, device=seq.device, dtype=seq.dtype)
            chunks.append(torch.cat([seq, pad], dim=0))
        else:
            start = torch.randint(0, T - L + 1, (1,)).item()
            chunks.append(seq[start:start+L])
    return chunks


def sample_boundary_aligned(seq: torch.Tensor, boundaries: torch.Tensor, L: int) -> torch.Tensor:
    """
    Sample one subsequence aligned with ground-truth boundaries.

    Args:
        seq: [T, D] tensor
        boundaries: [T] binary tensor (1 if end of a chunk)
        L: desired chunk length (approximate)

    Returns:
        [L, D] tensor (padded/truncated)
    """
    T, D = seq.shape
    idxs = torch.nonzero(boundaries, as_tuple=False).flatten().tolist()
    if len(idxs) < 2:
        # No boundaries -> fallback to uniform
        return sample_uniform_chunks(seq, L, 1)[0]

    # Pick a random boundary interval
    i = torch.randint(0, len(idxs) - 1, (1,)).item()
    s, e = idxs[i], idxs[i+1]
    seg = seq[s:e+1]
    if seg.size(0) >= L:
        return seg[:L]
    else:
        pad = torch.zeros(L - seg.size(0), D, device=seq.device, dtype=seq.dtype)
        return torch.cat([seg, pad], dim=0)


def batch_sample_chunks(batch: Dict[str, torch.Tensor], L: int, mode: str = "uniform", num_chunks: int = 1) -> torch.Tensor:
    """
    Apply chunk sampling to a batch of sequences.

    Args:
        batch: dict with keys:
            - "seq": [B, T, D]
            - "boundary": [B, T] (optional, required if mode="boundary")
        L: chunk length
        mode: "uniform" | "boundary"
        num_chunks: number of chunks per sequence

    Returns:
        [B * num_chunks, L, D] tensor
    """
    seqs = batch["seq"]
    B, T, D = seqs.shape
    out = []
    for b in range(B):
        seq = seqs[b]
        if mode == "uniform":
            chunks = sample_uniform_chunks(seq, L, num_chunks)
        elif mode == "boundary":
            boundaries = batch["boundary"][b]
            chunks = [sample_boundary_aligned(seq, boundaries, L) for _ in range(num_chunks)]
        else:
            raise ValueError(f"Unknown mode={mode}")
        out.extend(chunks)
    return torch.stack(out, dim=0)  # [B*num_chunks, L, D]
