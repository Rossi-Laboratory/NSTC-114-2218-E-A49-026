# acp/utils/metrics.py
"""
Metrics for Action Chunk Prediction (ACP)
=========================================

This module provides evaluation metrics for:
- Boundary prediction (precision, recall, f1, accuracy)
- Chunk-level IoU (pred vs ground-truth segments)
- VQ codebook usage (entropy, diversity)
"""

from typing import List, Dict, Tuple
import torch


# -------------------------
# Boundary metrics
# -------------------------
def boundary_metrics(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5) -> Dict[str, float]:
    """
    Compute boundary detection metrics.

    Args:
        logits: [B,T] predicted logits
        targets: [B,T] ground truth {0,1}
        thresh: threshold for positive

    Returns:
        dict with precision, recall, f1, accuracy
    """
    preds = (torch.sigmoid(logits) > thresh).int()
    targs = targets.int()

    tp = ((preds == 1) & (targs == 1)).sum().item()
    fp = ((preds == 1) & (targs == 0)).sum().item()
    fn = ((preds == 0) & (targs == 1)).sum().item()
    tn = ((preds == 0) & (targs == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
    }


# -------------------------
# Chunk IoU
# -------------------------
def segments_from_boundary(boundary: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Convert boundary mask [T] (0/1) into list of (start, end) segments.
    """
    T = boundary.numel()
    segs = []
    start = 0
    for t in range(T):
        if boundary[t].item() == 1:
            segs.append((start, t))
            start = t + 1
    if start < T:
        segs.append((start, T - 1))
    return segs


def chunk_iou(pred_boundary: torch.Tensor, true_boundary: torch.Tensor) -> float:
    """
    Compute IoU between predicted and ground-truth segments.

    Args:
        pred_boundary: [T] binary {0,1}
        true_boundary: [T] binary {0,1}

    Returns:
        IoU in [0,1]
    """
    pred_segs = segments_from_boundary(pred_boundary)
    true_segs = segments_from_boundary(true_boundary)

    # Convert to sets of frame indices
    pred_set = set()
    for s, e in pred_segs:
        pred_set.update(range(s, e + 1))
    true_set = set()
    for s, e in true_segs:
        true_set.update(range(s, e + 1))

    inter = len(pred_set & true_set)
    union = len(pred_set | true_set)
    return inter / (union + 1e-8)


# -------------------------
# VQ metrics
# -------------------------
def vq_code_usage(indices: torch.Tensor, num_codes: int) -> Dict[str, float]:
    """
    Compute entropy & diversity of code usage.

    Args:
        indices: [B,N] code indices
        num_codes: int

    Returns:
        dict with entropy, diversity
    """
    flat = indices.flatten()
    hist = torch.bincount(flat, minlength=num_codes).float()
    probs = hist / (hist.sum() + 1e-6)
    entropy = -(probs * (probs + 1e-6).log()).sum().item()
    max_entropy = torch.log(torch.tensor(float(num_codes))).item()
    diversity = (hist > 0).float().mean().item()

    return {
        "entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": entropy / (max_entropy + 1e-8),
        "diversity": diversity,
    }


# -------------------------
# Aggregator
# -------------------------
def evaluate(outputs: Dict, targets: Dict, num_codes: int = 128, thresh: float = 0.5) -> Dict[str, float]:
    """
    Aggregate evaluation metrics.

    Args:
        outputs: dict from model forward
            - "boundary_logits": [B,T]
            - "codes": [B,N] (optional)
        targets: dict
            - "boundary": [B,T]
        num_codes: size of codebook
        thresh: threshold for boundary classification

    Returns:
        dict with all metrics
    """
    metrics = {}

    if "boundary_logits" in outputs and "boundary" in targets:
        m = boundary_metrics(outputs["boundary_logits"], targets["boundary"], thresh=thresh)
        metrics.update({f"boundary_{k}": v for k, v in m.items()})

        # chunk IoU (first example only, for efficiency)
        pred = (torch.sigmoid(outputs["boundary_logits"][0]) > thresh).int()
        true = targets["boundary"][0].int()
        metrics["chunk_iou"] = chunk_iou(pred, true)

    if "codes" in outputs and outputs["codes"] is not None:
        m = vq_code_usage(outputs["codes"], num_codes=num_codes)
        metrics.update({f"vq_{k}": v for k, v in m.items()})

    return metrics
