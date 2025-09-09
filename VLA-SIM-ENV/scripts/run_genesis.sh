#!/usr/bin/env bash
set -euo pipefail
python -m src.init_genesis --config configs/genesis_config.yaml --scene configs/scenes/industrial.yaml
