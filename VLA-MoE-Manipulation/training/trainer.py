import os, torch
import torch.nn as nn
import torch.optim as optim
from utils.logger import log
from utils.checkpoint import save_checkpoint
from training.loss_functions import action_mse_loss, load_balance_loss

class Trainer:
    def __init__(self, model, device, lr=1e-3, output_dir="outputs", log_interval=10, save_interval=1):
        self.model = model
        self.device = device
        self.opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    def train_step(self, batch):
        self.model.train()
        image = batch['image'].to(self.device)
        text_ids = batch['text_ids'].to(self.device)
        proprio = batch['proprio'].to(self.device)
        tactile = batch['tactile'].to(self.device)
        target = batch['action'].to(self.device)

        pred, weights = self.model(image, text_ids, proprio, tactile)

        mse = action_mse_loss(pred, target)
        lb = load_balance_loss(weights)
        loss = mse + 0.01 * lb

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        metrics = {
            "mse": mse.item(),
            "lb": lb.item(),
        }
        return loss.item(), metrics

    def save_ckpt(self, path):
        save_checkpoint(path, self.model, self.opt)
