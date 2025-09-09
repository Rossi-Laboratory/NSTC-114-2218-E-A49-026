#!/usr/bin/env bash
set -e
python -m acp.inference.run_inference --config configs/model_acp.yaml --seq_path data/processed/train_sequences.pt --save outputs/last_infer.json
