# acp/training/train.py
"""
Training entry point for Action Chunk Prediction (ACP)
======================================================

This script handles:
- Config parsing
- Dataset & DataLoader
- Model / Optimizer / Scheduler setup
- Loss computation
- Training loop with logging & checkpoint saving
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from acp.data.sequence_dataset import SequenceDataset
from acp.data.collate import collate_fn
from acp.models.action_chunk_predictor import ActionChunkPredictor
from acp.training.losses import aggregate_losses
from acp.training.optim import build_optim_and_sched
from acp.utils.logger import get_logger
from acp.utils.seed import set_seed
from acp.utils.checkpoint import save as save_ckpt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/model_acp.yaml",
                    help="YAML config file for model + training")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", type=str, default="outputs")
    return ap.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg, logger):
    train_path = cfg["data"]["train"]
    val_path = cfg["data"].get("val", None)

    train_set = SequenceDataset(train_path)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    logger.info(f"Loaded train dataset: {len(train_set)} samples")

    val_loader = None
    if val_path and os.path.exists(val_path):
        val_set = SequenceDataset(val_path)
        val_loader = DataLoader(
            val_set,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
        )
        logger.info(f"Loaded val dataset: {len(val_set)} samples")

    return train_loader, val_loader


def build_model(cfg, device):
    model = ActionChunkPredictor(
        d_model=cfg["model"]["d_model"],
        use_vq=cfg["model"]["use_vq"],
        codebook_size=cfg["model"].get("codebook_size", 128),
        depth=cfg["encoder"]["depth"],
        heads=cfg["encoder"]["heads"],
        dropout=cfg["encoder"]["dropout"],
    ).to(device)
    return model


def train_one_epoch(model, loader, optim, sched, device, cfg, logger, epoch):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        seq = batch["seq"].to(device)
        target_b = batch["boundary"].to(device)

        outputs = model(seq)
        targets = {"boundary": target_b}
        logs = aggregate_losses(outputs, targets, cfg["loss"])

        loss = logs["loss_total"]
        optim.zero_grad()
        loss.backward()
        optim.step()
        if sched:
            sched.step()

        total_loss += float(loss.item())
        if (step + 1) % cfg["training"].get("log_interval", 20) == 0:
            logger.info(f"Epoch {epoch} Step {step+1}/{len(loader)} Loss={loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    return avg_loss


def validate(model, loader, device, cfg, logger):
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            target_b = batch["boundary"].to(device)
            outputs = model(seq)
            targets = {"boundary": target_b}
            logs = aggregate_losses(outputs, targets, cfg["loss"])
            total_loss += float(logs["loss_total"].item())
    avg_loss = total_loss / len(loader)
    logger.info(f"Validation Loss={avg_loss:.4f}")
    return avg_loss


def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.makedirs(args.save_dir, exist_ok=True)
    logger = get_logger("acp.train", log_file=os.path.join(args.save_dir, "train.log"))

    set_seed(cfg.get("seed", 42))

    # Build data
    train_loader, val_loader = build_dataloaders(cfg, logger)

    # Build model
    device = torch.device(args.device)
    model = build_model(cfg, device)

    # Build optimizer & scheduler
    optim, sched = build_optim_and_sched(model, cfg)

    # Training loop
    best_val = float("inf")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optim, sched, device, cfg, logger, epoch)
        logger.info(f"[Epoch {epoch}] Train Loss={train_loss:.4f}")

        val_loss = validate(model, val_loader, device, cfg, logger)
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            save_ckpt(os.path.join(args.save_dir, "checkpoint_best.pth"),
                      model=model.state_dict(), optim=optim.state_dict(),
                      epoch=epoch, best=best_val)
            logger.info(f"Saved new best checkpoint at epoch {epoch} (val loss={val_loss:.4f})")

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
