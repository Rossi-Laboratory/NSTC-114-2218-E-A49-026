# acp/inference/export.py
"""
Model Export Utilities for Action Chunk Prediction (ACP)
=======================================================

This module provides functions and CLI to export a trained ACP model
(ActionChunkPredictor) to TorchScript or ONNX formats.

Usage
-----
TorchScript export:
    python -m acp.inference.export \
        --config configs/model_acp.yaml \
        --acp_ckpt outputs/checkpoint_best.pth \
        --format torchscript \
        --save outputs/acp_model.pt

ONNX export:
    python -m acp.inference.export \
        --config configs/model_acp.yaml \
        --acp_ckpt outputs/checkpoint_best.pth \
        --format onnx \
        --save outputs/acp_model.onnx
"""

import os
import argparse
import torch
import yaml

from acp.models.action_chunk_predictor import ActionChunkPredictor


def load_acp(config_path: str, ckpt_path: str, device: str = "cpu") -> ActionChunkPredictor:
    """Load ACP model from config + checkpoint."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = ActionChunkPredictor(
        d_model=cfg["model"]["d_model"],
        use_vq=cfg["model"]["use_vq"],
        codebook_size=cfg["model"].get("codebook_size", 128),
        depth=cfg["encoder"]["depth"],
        heads=cfg["encoder"]["heads"],
        dropout=cfg["encoder"]["dropout"],
    ).to(device)

    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state["model"] if "model" in state else state
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded checkpoint {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint not found at {ckpt_path}, exporting untrained model.")

    model.eval()
    return model


def export_torchscript(model: ActionChunkPredictor, save_path: str, seq_len: int = 32, d_model: int = 256):
    """Export to TorchScript (.pt)."""
    dummy = torch.randn(1, seq_len, d_model)
    traced = torch.jit.trace(model, dummy)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    traced.save(save_path)
    print(f"[INFO] TorchScript model saved to {save_path}")


def export_onnx(model: ActionChunkPredictor, save_path: str, seq_len: int = 32, d_model: int = 256):
    """Export to ONNX (.onnx)."""
    dummy = torch.randn(1, seq_len, d_model)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        save_path,
        input_names=["seq"],
        output_names=["boundary_logits", "emb", "codes"],
        opset_version=13,
        dynamic_axes={"seq": {1: "T"}},
    )
    print(f"[INFO] ONNX model saved to {save_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/model_acp.yaml")
    ap.add_argument("--acp_ckpt", type=str, default="outputs/checkpoint_best.pth")
    ap.add_argument("--format", type=str, choices=["torchscript", "onnx"], default="torchscript")
    ap.add_argument("--save", type=str, default="outputs/acp_model.pt")
    ap.add_argument("--seq_len", type=int, default=32, help="dummy sequence length for export tracing")
    ap.add_argument("--device", type=str, default="cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    model = load_acp(args.config, args.acp_ckpt, device=args.device)

    if args.format == "torchscript":
        export_torchscript(model, args.save, seq_len=args.seq_len, d_model=args.model_d_model if hasattr(args,"model_d_model") else 256)
    else:
        export_onnx(model, args.save, seq_len=args.seq_len, d_model=args.model_d_model if hasattr(args,"model_d_model") else 256)


if __name__ == "__main__":
    main()
