#!/usr/bin/env python
"""
Train Script for VLA-MoE Manipulation
-------------------------------------
- Loads configs (train.yaml, model_moe.yaml)
- Builds dataset and dataloader
- Initializes VLAMoE model
- Runs training loop with optimizer + scheduler
- Supports resume from checkpoint
- Saves logs and checkpoints to out_dir
"""

import os
import argparse
import yaml
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import build_dataloader
from models.vla_moe import VLAMoE
from utils.checkpoint import save_checkpoint, load_checkpoint
from training.loss_functions import total_loss


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    cfg,
    epoch: int,
    log_interval: int = 50,
):
    model.train()
    total_step = len(dataloader)
    epoch_loss = 0.0

    for step, batch in enumerate(dataloader):
        imgs = batch["image"].to(device)
        text_ids = batch["text_ids"].to(device)
        proprio = batch["proprio"].to(device)
        tactile = batch["tactile"].to(device)
        actions = batch["actions"].to(device)

        optimizer.zero_grad()
        pred, weights = model(imgs, text_ids, proprio, tactile)
        loss, logs = total_loss(pred, actions, weights, cfg["loss"])
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if step % log_interval == 0:
            print(
                f"[Epoch {epoch}] Step {step}/{total_step} | "
                f"Loss: {logs['loss_total']:.4f} | "
                f"Action: {logs['loss_action']:.4f} | "
                f"Balance: {logs['loss_balance']:.4f} | "
                f"Entropy: {logs['loss_entropy']:.4f}"
            )

    return epoch_loss / total_step


def validate(model, dataloader, device, cfg):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device)
            text_ids = batch["text_ids"].to(device)
            proprio = batch["proprio"].to(device)
            tactile = batch["tactile"].to(device)
            actions = batch["actions"].to(device)

            pred, weights = model(imgs, text_ids, proprio, tactile)
            loss, _ = total_loss(pred, actions, weights, cfg["loss"])
            val_loss += loss.item()
    return val_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--model_cfg", type=str, default="configs/model_moe.yaml")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load configs
    cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_cfg)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # dataset + dataloader
    train_loader = build_dataloader(
        cfg["dataset"], batch_size=cfg["train"]["batch_size"], shuffle=True
    )
    val_loader = build_dataloader(
        cfg["dataset"], batch_size=cfg["train"]["batch_size"], shuffle=False
    )

    # model
    model = VLAMoE(model_cfg).to(device)

    # optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["train"].get("lr_step", 10), gamma=0.5
    )

    # resume
    start_epoch = 1
    best_val = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = load_checkpoint(args.resume, model, optimizer)
        start_epoch = ckpt.get("epoch", 1)
        best_val = ckpt.get("best_val", float("inf"))
        print(f"[INFO] Resumed from {args.resume} at epoch {start_epoch}")

    # training loop
    num_epochs = cfg["train"]["epochs"]
    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg, epoch)
        val_loss = validate(model, val_loader, device, cfg)
        scheduler.step()

        print(
            f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time()-t0:.1f}s"
        )

        # save checkpoint
        ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch}.pth")
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            epoch,
            best_val,
        )

        # best model
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.out_dir, "checkpoint_best.pth")
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_val,
            )
            print(f"[INFO] Saved new best model to {best_path}")


if __name__ == "__main__":
    main()
