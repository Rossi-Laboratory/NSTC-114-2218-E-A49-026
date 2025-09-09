# acp/inference/decode.py
"""
Action Chunk Decoding Utilities
===============================

Two decoding modes:

1) VQ-based decode (no VLA dependency):
   - Load ACP checkpoint to access the VQ codebook (embedding table).
   - Optionally load a tiny MLP decoder (embedding -> 7D action).
   - Reconstruct a per-step action sequence by repeating each chunk's decoded action
     over its segment length (derived from boundary logits).

2) VLA-bridge decode (uses your existing VLA-MoE-Manipulation):
   - Use acp/integration/vla_moe_bridge.py to import InferenceSession from your VLA repo.
   - For each chunk, call session.predict(command, image_path=...) repeatedly to produce
     a fixed number of low-level steps (steps_per_chunk). This avoids re-implementing VLA.

Inputs
------
- Inference JSON produced by acp/inference/run_inference.py, containing:
  {
    "boundary_logits": [[p_0, p_1, ..., p_{T-1}]],   # probs for sequence 0
    "codes": [[c_0, c_1, ..., c_{N-1}]] or null      # optional, if VQ enabled
  }

Outputs
-------
- Decoded JSON at --save (default: outputs/decoded_actions.json):
  {
    "actions": [[a_0,...,a_6], ..., [a_{T-1},...,a_{T-1,6}]],
    "segments": [[start0, end0], [start1, end1], ...],
    "per_step_codes": [k_0, k_1, ..., k_{T-1}] or null,
    "mode": "vq" | "vla"
  }

CLI Examples
------------
VQ decode (uses ACP ckpt & optional MLP decoder):
    python -m acp.inference.decode \
        --mode vq \
        --infer_json outputs/last_infer.json \
        --config configs/model_acp.yaml \
        --acp_ckpt outputs/checkpoint_best.pth \
        --save outputs/decoded_actions.json

VLA decode (uses your VLA repo):
    python -m acp.inference.decode \
        --mode vla \
        --infer_json outputs/last_infer.json \
        --vla_repo_root . \
        --vla_infer_cfg third_party/VLA-MoE-Manipulation/configs/inference.yaml \
        --vla_model_cfg third_party/VLA-MoE-Manipulation/configs/model_moe.yaml \
        --command "pick up the red block" \
        --image data/raw/sample.png \
        --steps_per_chunk 5 \
        --save outputs/decoded_actions.json
"""

from __future__ import annotations

import os
import json
import argparse
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from acp.models.action_chunk_predictor import ActionChunkPredictor
from acp.integration.vla_moe_bridge import load_vla_inference


# ----------------------------
# Helpers: boundaries & segments
# ----------------------------
def logits_to_boundary_bool(logits: torch.Tensor, thresh: float = 0.5) -> torch.BoolTensor:
    """
    Convert logits [B,T] -> boolean ends [B,T], True where a chunk ends at position t.
    """
    probs = torch.sigmoid(logits)
    return probs > thresh


def segments_from_ends(ends_bool: torch.BoolTensor) -> List[List[Tuple[int, int]]]:
    """
    ends_bool: [B,T] with True at end of a segment (inclusive index).
    Returns per-batch list of (start, end) segments.
    """
    B, T = ends_bool.shape
    all_segments: List[List[Tuple[int, int]]] = []
    for b in range(B):
        segs = []
        start = 0
        for t in range(T):
            if ends_bool[b, t]:
                segs.append((start, t))
                start = t + 1
        if start < T:
            segs.append((start, T - 1))
        if len(segs) == 0:
            segs = [(0, T - 1)]
        all_segments.append(segs)
    return all_segments


def expand_codes_to_steps(codes: torch.Tensor, segments: List[Tuple[int, int]], T: int) -> List[int]:
    """
    Expand chunk codes [N] to per-step timeline [T] using (start,end) segments order.
    If codes is None, returns [-1]*T.
    """
    per_step = [-1] * T
    if codes is None:
        return per_step
    codes_list = codes.squeeze(0).tolist()  # [N]
    for i, (s, e) in enumerate(segments):
        k = codes_list[i] if i < len(codes_list) else -1
        for t in range(s, e + 1):
            per_step[t] = int(k)
    return per_step


# ----------------------------
# Simple MLP decoder (optional)
# ----------------------------
class SimpleActionDecoder(nn.Module):
    """
    Tiny MLP: embedding[D] -> action[A]. Intended to be trained separately if desired.
    If a checkpoint is provided (--decoder_mlp), we will load state_dict from it.
    """
    def __init__(self, emb_dim: int, action_dim: int = 7, hidden_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = max(emb_dim * hidden_mult, action_dim * 2)
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,N,D] or [B,T,D]
        return self.net(z)


def load_decoder_mlp(path: Optional[str], emb_dim: int, action_dim: int) -> SimpleActionDecoder:
    dec = SimpleActionDecoder(emb_dim, action_dim)
    if path and os.path.isfile(path):
        sd = torch.load(path, map_location="cpu")
        # allow raw state_dict or wrapped
        state = sd.get("model", sd)
        dec.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded decoder MLP from {path}")
    else:
        print("[WARN] No decoder MLP provided; using randomly initialized MLP. "
              "Decoded actions may be meaningless without training.")
    return dec


# ----------------------------
# VQ-based decoding
# ----------------------------
@torch.no_grad()
def decode_vq(
    infer_json: str,
    config_yaml: str,
    acp_ckpt: str,
    save_path: str,
    boundary_thresh: float = 0.5,
    action_dim: int = 7,
    decoder_mlp: Optional[str] = None,
    device: str = "cpu",
) -> None:
    """
    Reconstruct per-step actions from ACP outputs using the ACP's own VQ codebook + MLP decoder.

    Steps:
        1) Load inference JSON -> boundary logits & codes.
        2) Load ACP checkpoint to access VQ codebook & emb dim.
        3) Convert logits -> segments; align codes to segments.
        4) For each chunk code, fetch codebook embedding; feed MLP -> chunk action.
        5) Repeat per segment length -> per-step action sequence.
        6) Save JSON with actions, segments, and per_step_codes.
    """
    with open(infer_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract boundary & codes (only the first sequence for now)
    probs = torch.tensor(data["boundary_logits"], dtype=torch.float32)  # [B=1, T]
    B, T = probs.shape
    codes = None
    if data.get("codes", None) is not None:
        codes = torch.tensor(data["codes"], dtype=torch.long)  # [B, N]
    print(f"[INFO] Loaded inference JSON: T={T}, has_codes={codes is not None}")

    # Load ACP checkpoint to get VQ codebook + emb dim
    ckpt = torch.load(acp_ckpt, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    # Infer emb dim from codebook
    code_weight = None
    for k, v in state_dict.items():
        if k.endswith("vq.codebook.weight"):
            code_weight = v  # [K, D]
            break
    if code_weight is None:
        raise RuntimeError("VQ codebook not found in ACP checkpoint. Ensure use_vq=true during training.")
    K, D = code_weight.shape
    print(f"[INFO] Found VQ codebook: K={K}, D={D}")

    # Build a decoder MLP (load if provided)
    decoder = load_decoder_mlp(decoder_mlp, emb_dim=D, action_dim=action_dim).to(device).eval()

    # Boundaries -> segments
    ends_bool = logits_to_boundary_bool(probs, thresh=boundary_thresh)  # [1,T]
    segments = segments_from_ends(ends_bool)[0]  # [(s,e), ...]
    print(f"[INFO] Segments: {segments}")

    # Codes -> per-step code ids
    per_step_codes = expand_codes_to_steps(codes, segments, T) if codes is not None else None

    # For each segment i, decode one action and repeat
    actions: List[List[float]] = []
    for i, (s, e) in enumerate(segments):
        seg_len = e - s + 1
        if codes is not None and i < codes.shape[1]:
            idx = int(codes[0, i].item())
            z = code_weight[idx:idx + 1].unsqueeze(0)  # [1,1,D]
        else:
            # If no code available, fallback to average of neighbors (here: zero vec)
            z = torch.zeros(1, 1, D)

        a = decoder(z.to(device)).squeeze(0).squeeze(0)  # [A]
        a = a.detach().cpu().tolist()
        for _ in range(seg_len):
            actions.append([float(x) for x in a])

    # If total steps < T (rare), pad; if > T, trim
    if len(actions) < T:
        pad = [0.0] * action_dim
        actions.extend([pad] * (T - len(actions)))
    elif len(actions) > T:
        actions = actions[:T]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "vq",
                "actions": actions,
                "segments": segments,
                "per_step_codes": per_step_codes,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] Saved decoded actions to {save_path}")


# ----------------------------
# VLA-bridge decoding
# ----------------------------
@torch.no_grad()
def decode_vla(
    infer_json: str,
    save_path: str,
    vla_repo_root: str,
    vla_infer_cfg: str,
    vla_model_cfg: str,
    command: str,
    image_path: str,
    steps_per_chunk: int = 5,
    boundary_thresh: float = 0.5,
    device: str = "cpu",
) -> None:
    """
    Use your existing VLA-MoE-Manipulation to synthesize low-level actions for each chunk.
    We do NOT re-implement VLA. We call InferenceSession from your VLA repo.

    Strategy:
        - Parse boundary logits -> segments.
        - For each segment, invoke VLA N=steps_per_chunk times to produce low-level steps
          (you may adapt this to your task, e.g., single step per chunk or variable N).
    """
    with open(infer_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    probs = torch.tensor(data["boundary_logits"], dtype=torch.float32)  # [1,T]
    B, T = probs.shape
    ends_bool = logits_to_boundary_bool(probs, thresh=boundary_thresh)
    segments = segments_from_ends(ends_bool)[0]  # [(s,e), ...]

    # Load InferenceSession from VLA repo via the bridge
    InferenceSession = load_vla_inference(vla_repo_root)
    sess = InferenceSession(vla_infer_cfg, vla_model_cfg, device=device)

    actions: List[List[float]] = []
    for i, (s, e) in enumerate(segments):
        # Simple policy: for each chunk, query VLA steps_per_chunk times
        for _ in range(steps_per_chunk):
            act, _weights = sess.predict(command, image_path=image_path)
            # act is expected to be [7] or [A]; ensure list of floats
            actions.append([float(x) for x in act])

    # Align to T by trimming or padding with last action
    if len(actions) > T:
        actions = actions[:T]
    elif len(actions) < T:
        tail = actions[-1] if len(actions) > 0 else [0.0] * 7
        actions.extend([tail] * (T - len(actions)))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "vla",
                "actions": actions,
                "segments": segments,
                "per_step_codes": None,  # VLA path does not rely on codes
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] Saved decoded actions to {save_path}")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["vq", "vla"], required=True,
                    help="Decoding mode: 'vq' uses ACP VQ codebook; 'vla' calls VLA-MoE-Manipulation.")
    ap.add_argument("--infer_json", type=str, required=True, help="Inference JSON from acp/inference/run_inference.py")
    ap.add_argument("--save", type=str, default="outputs/decoded_actions.json")

    # VQ mode args
    ap.add_argument("--config", type=str, default="configs/model_acp.yaml", help="ACP config (for sanity)")
    ap.add_argument("--acp_ckpt", type=str, default="outputs/checkpoint_best.pth", help="ACP checkpoint path")
    ap.add_argument("--decoder_mlp", type=str, default="", help="Optional MLP decoder checkpoint (embedding->action)")
    ap.add_argument("--boundary_thresh", type=float, default=0.5)
    ap.add_argument("--action_dim", type=int, default=7)
    ap.add_argument("--device", type=str, default="cpu")

    # VLA mode args
    ap.add_argument("--vla_repo_root", type=str, default=".", help="Repo root containing third_party/VLA-MoE-Manipulation")
    ap.add_argument("--vla_infer_cfg", type=str, default="third_party/VLA-MoE-Manipulation/configs/inference.yaml")
    ap.add_argument("--vla_model_cfg", type=str, default="third_party/VLA-MoE-Manipulation/configs/model_moe.yaml")
    ap.add_argument("--command", type=str, default="pick up the red block")
    ap.add_argument("--image", type=str, default="data/raw/sample.png")
    ap.add_argument("--steps_per_chunk", type=int, default=5)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.mode == "vq":
        decode_vq(
            infer_json=args.infer_json,
            config_yaml=args.config,
            acp_ckpt=args.acp_ckpt,
            save_path=args.save,
            boundary_thresh=args.boundary_thresh,
            action_dim=args.action_dim,
            decoder_mlp=args.decoder_mlp if args.decoder_mlp else None,
            device=args.device,
        )
    else:
        decode_vla(
            infer_json=args.infer_json,
            save_path=args.save,
            vla_repo_root=args.vla_repo_root,
            vla_infer_cfg=args.vla_infer_cfg,
            vla_model_cfg=args.vla_model_cfg,
            command=args.command,
            image_path=args.image,
            steps_per_chunk=args.steps_per_chunk,
            boundary_thresh=args.boundary_thresh,
            device=args.device,
        )


if __name__ == "__main__":
    main()
