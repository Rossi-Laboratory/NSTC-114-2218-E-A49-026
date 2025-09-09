#!/usr/bin/env python
# scripts/preprocess_sequences.py
import argparse, json, torch, os
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, required=True, help='demo sequence json')
    ap.add_argument('--output', type=str, required=True, help='output .pt')
    args = ap.parse_args()
    with open(args.input,'r',encoding='utf-8') as f:
        seq = json.load(f)  # expects {"actions": [[...], ...]}
    import torch
    actions = torch.tensor(seq['actions'], dtype=torch.float32).unsqueeze(0)  # [1,T,D]
    T, D = actions.shape[1], actions.shape[2]
    # naive boundary target: every k steps
    k = max(5, T//5)
    boundary = torch.zeros(1, T)
    boundary[0, k-1::k] = 1.0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({'seqs': actions, 'boundaries': boundary}, args.output)
    print(f'[INFO] saved {args.output}')
if __name__ == '__main__':
    main()
