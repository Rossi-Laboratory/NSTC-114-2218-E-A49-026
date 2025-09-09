# acp/training/train.py
import argparse, yaml
from .trainer import train_loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/model_acp.yaml')
    args = ap.parse_args()
    with open(args.config,'r',encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    train_loop(cfg)

if __name__ == '__main__':
    main()
