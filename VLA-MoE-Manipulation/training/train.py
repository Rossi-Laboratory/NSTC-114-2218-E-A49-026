import os, sys, argparse, yaml, time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import SyntheticManipulationDataset
from models.vla_moe import VLAMoE
from training.trainer import Trainer
from utils.seed import set_seed

def load_yaml(p):
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = load_yaml(cfg['model']['config'])

    set_seed(cfg.get('seed', 42))

    device = torch.device(cfg.get('device', 'cpu'))
    os.makedirs(cfg['output_dir'], exist_ok=True)

    # dataset
    ds = SyntheticManipulationDataset(
        num_samples=cfg['dataset']['num_samples'],
        image_size=cfg['dataset']['image_size'],
        vocab_size=cfg['dataset']['vocab_size'],
        max_text_len=cfg['dataset']['max_text_len'],
        proprio_dim=cfg['dataset']['proprio_dim'],
        tactile_dim=cfg['dataset']['tactile_dim'],
        action_dim=cfg['dataset']['action_dim'],
    )
    dl = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])

    # model
    model = VLAMoE(model_cfg).to(device)
    trainer = Trainer(
        model=model,
        device=device,
        lr=cfg['learning_rate'],
        output_dir=cfg['output_dir'],
        log_interval=cfg['log_interval'],
        save_interval=cfg['save_interval'],
    )

    for epoch in range(1, cfg['epochs'] + 1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for step, batch in enumerate(pbar, 1):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            loss, metrics = trainer.train_step(batch)
            pbar.set_postfix({"loss": f"{loss:.4f}", **{k: f"{v:.3f}" for k, v in metrics.items()}})

        if epoch % cfg['save_interval'] == 0:
            trainer.save_ckpt(os.path.join(cfg['output_dir'], 'checkpoints', f"epoch_{epoch}.pt"))
            trainer.save_ckpt(os.path.join(cfg['output_dir'], 'checkpoints', "latest.pt"))

if __name__ == "__main__":
    main()
