#!/usr/bin/env bash
set -euo pipefail
python -m src.init_omniverse --config configs/omniverse_config.yaml --scene configs/scenes/multi_object.yaml
