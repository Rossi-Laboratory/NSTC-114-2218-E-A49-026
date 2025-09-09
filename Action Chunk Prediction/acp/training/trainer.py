# acp/training/trainer.py
import torch, os
from torch.utils.data import DataLoader
from ..data.sequence_dataset import SequenceDataset
from ..data.collate import collate_fn
from ..models.action_chunk_predictor import ActionChunkPredictor

def save_ckpt(path, model, optim, epoch, best):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch, "best": best}, path)

def train_loop(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SequenceDataset(cfg["data"]["train"])
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    model = ActionChunkPredictor(
        d_model=cfg["model"]["d_model"],
        use_vq=cfg["model"]["use_vq"],
        codebook_size=cfg["model"].get("codebook_size",128),
        depth=cfg["encoder"]["depth"],
        heads=cfg["encoder"]["heads"],
        dropout=cfg["encoder"]["dropout"],
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    best = 1e9
    for epoch in range(1, cfg["training"]["epochs"]+1):
        model.train(); total=0.0
        for batch in dl:
            seq = batch["seq"].to(device)
            target_b = batch["boundary"].to(device).float()
            out = model(seq)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(out["boundary_logits"], target_b)
            loss = bce + cfg["training"].get("lambda_vq",1.0) * out["vq_loss"]
            optim.zero_grad(); loss.backward(); optim.step()
            total += float(loss.item())
        avg = total/len(dl)
        print(f"[Epoch {epoch}] loss={avg:.4f}")
        if avg<best:
            best=avg
            save_ckpt("outputs/checkpoint_best.pth", model, optim, epoch, best)
    print("[Done] best loss:", best)
