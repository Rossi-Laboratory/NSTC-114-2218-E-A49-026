# Manipulation Example
- `sequences/demo_seq_01.json` : toy action sequence (T=30, D=7)
- Run:
  ```bash
  python scripts/preprocess_sequences.py --input examples/manipulation/sequences/demo_seq_01.json --output data/processed/train_sequences.pt
  bash scripts/train_acp.sh
  bash scripts/infer_acp.sh
  bash scripts/eval_acp.sh
  ```
