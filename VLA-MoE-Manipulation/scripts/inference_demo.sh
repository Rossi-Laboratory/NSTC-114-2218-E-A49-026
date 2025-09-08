#!/usr/bin/env bash
set -e
python inference/run_inference.py --config configs/inference.yaml --command "pick up the red block"
