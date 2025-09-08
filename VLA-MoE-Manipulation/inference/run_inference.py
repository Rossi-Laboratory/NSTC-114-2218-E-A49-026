import os, sys, argparse, yaml, torch
from PIL import Image
import numpy as np

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.vla_moe import VLAMoE
from data.dataset import tokenize
from utils.checkpoint import load_checkpoint
from utils.seed import set_seed

def load_yaml(p):
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/inference.yaml')
    parser.add_argument('--image', type=str, default=None, help='optional path to an RGB image')
    parser.add_argument('--command', type=str, default='pick up the red block')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = load_yaml(cfg['model']['config'])
    device = torch.device(cfg.get('device', 'cpu'))
    set_seed(123)

    # Inputs
    H = W = cfg['dataset']['image_size']
    if args.image and os.path.isfile(args.image):
        img = Image.open(args.image).convert('RGB').resize((W, H))
        image = torch.from_numpy(np.array(img)).float().permute(2,0,1) / 255.0
    else:
        image = torch.rand(3, H, W)  # random if not provided

    text_ids = tokenize(args.command, cfg['dataset']['vocab_size'], cfg['dataset']['max_text_len'])
    proprio = torch.zeros(cfg['dataset']['proprio_dim'])
    tactile = torch.zeros(cfg['dataset']['tactile_dim'])

    image = image.unsqueeze(0).to(device)
    text_ids = text_ids.unsqueeze(0).to(device)
    proprio = proprio.unsqueeze(0).to(device)
    tactile = tactile.unsqueeze(0).to(device)

    # Model
    model = VLAMoE(model_cfg).to(device)
    ckpt_path = cfg['checkpoint_path']
    if os.path.isfile(ckpt_path):
        load_checkpoint(ckpt_path, model)
        print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint not found at {ckpt_path}; using randomly initialized weights.")

    model.eval()
    with torch.no_grad():
        action, weights = model(image, text_ids, proprio, tactile)

    print("Command:", args.command)
    print("Predicted action (7D: dx, dy, dz, droll, dpitch, dyaw, gripper):\n", action.cpu().numpy())
    print("Expert mixture weights:", weights.cpu().numpy())

if __name__ == "__main__":
    main()
