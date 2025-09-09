#!/usr/bin/env bash
set -euo pipefail
python -m src.init_omniverse --config configs/omniverse_config.yaml --scene configs/scenes/multi_object.yaml --dry-run
python -m src.init_genesis   --config configs/genesis_config.yaml   --scene configs/scenes/household.yaml   --dry-run
python -m src.init_genesis   --config configs/genesis_config.yaml   --scene configs/scenes/industrial.yaml  --dry-run
