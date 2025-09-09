# acp/inference/run_inference.py
import argparse, json, torch, yaml
from ..models.action_chunk_predictor import ActionChunkPredictor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/model_acp.yaml')
    ap.add_argument('--seq_path', type=str, default='data/processed/train_sequences.pt')
    ap.add_argument('--save', type=str, default='outputs/last_infer.json')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data = torch.load(args.seq_path)
    seq = data['seqs'][0:1]  # [1,T,D]
    model = ActionChunkPredictor(d_model=cfg['model']['d_model']).eval()
    with torch.no_grad():
        out = model(seq)
    res = {'boundary_logits': out['boundary_logits'].sigmoid().cpu().tolist(),
           'codes': (out['codes'].cpu().tolist() if out['codes'] is not None else None)}
    import os; os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[INFO] saved {args.save}")

if __name__ == '__main__':
    main()
