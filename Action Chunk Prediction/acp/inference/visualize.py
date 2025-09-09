# acp/inference/visualize.py
import argparse, json
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, required=True)
    ap.add_argument('--save', type=str, default='outputs/vis_chunks.png')
    args = ap.parse_args()
    with open(args.input,'r',encoding='utf-8') as f:
        data = json.load(f)
    probs = data['boundary_logits'][0]  # first sequence
    plt.figure()
    plt.plot(probs)
    plt.title('Boundary Probability')
    plt.xlabel('t'); plt.ylabel('p(boundary)')
    import os; os.makedirs(os.path.dirname(args.save), exist_ok=True)
    plt.savefig(args.save, dpi=200)
    print(f"[INFO] saved {args.save}")

if __name__ == '__main__':
    main()
