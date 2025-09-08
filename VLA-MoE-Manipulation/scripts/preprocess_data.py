#!/usr/bin/env python
"""
Preprocess Data Script for VLA-MoE Manipulation
-----------------------------------------------
This script performs preprocessing for raw multimodal data:
- Reads raw samples (image path + text command + proprio/tactile/action labels).
- Tokenizes text commands.
- Resizes and normalizes images.
- Saves processed tensors and metadata to `data/processed/`.

Usage:
    python scripts/preprocess_data.py \
        --input data/raw/train.json \
        --output data/processed/train.pt \
        --image_size 128 --vocab_size 512 --max_text_len 16
"""

import os
import json
import argparse
from typing import Dict, List

import torch
import numpy as np
from PIL import Image

from data.dataset import tokenize


def preprocess(
    input_json: str,
    output_path: str,
    image_size: int = 128,
    vocab_size: int = 512,
    max_text_len: int = 16,
) -> None:
    """Preprocess raw dataset into a torch .pt file."""
    print(f"[INFO] Loading raw data: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed = {
        "images": [],
        "text_ids": [],
        "proprio": [],
        "tactile": [],
        "actions": [],
        "commands": [],
    }

    for idx, sample in enumerate(raw_data):
        # --- Image ---
        img_path = sample.get("image")
        if img_path is None or not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
        img_arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
        img_arr = np.transpose(img_arr, (2, 0, 1))            # CHW
        processed["images"].append(img_arr)

        # --- Text ---
        command = sample.get("command", "")
        ids = tokenize(command, vocab_size, max_text_len)
        processed["text_ids"].append(ids)
        processed["commands"].append(command)

        # --- Proprio / Tactile ---
        processed["proprio"].append(sample.get("proprio", [0.0] * 7))
        processed["tactile"].append(sample.get("tactile", [0.0] * 4))

        # --- Action ---
        processed["actions"].append(sample.get("action", [0.0] * 7))

        if idx % 100 == 0:
            print(f"[INFO] Processed {idx}/{len(raw_data)} samples...")

    # Convert to tensors
    dataset = {
        "images": torch.tensor(np.stack(processed["images"]), dtype=torch.float32),
        "text_ids": torch.tensor(np.stack(processed["text_ids"]), dtype=torch.long),
        "proprio": torch.tensor(np.stack(processed["proprio"]), dtype=torch.float32),
        "tactile": torch.tensor(np.stack(processed["tactile"]), dtype=torch.float32),
        "actions": torch.tensor(np.stack(processed["actions"]), dtype=torch.float32),
        "commands": processed["commands"],  # keep as list of strings
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dataset, output_path)
    print(f"[INFO] Saved processed dataset to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to save .pt file")
    parser.add_argument("--image_size", type=int, default=128, help="Resize image size (H=W)")
    parser.add_argument("--vocab_size", type=int, default=512, help="Vocabulary size for tokenize()")
    parser.add_argument("--max_text_len", type=int, default=16, help="Max text length for tokenize()")
    args = parser.parse_args()

    preprocess(
        input_json=args.input,
        output_path=args.output,
        image_size=args.image_size,
        vocab_size=args.vocab_size,
        max_text_len=args.max_text_len,
    )
