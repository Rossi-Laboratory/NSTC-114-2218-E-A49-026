# Usage

Dry-run (no external simulator required):
```bash
python -m src.init_omniverse --config configs/omniverse_config.yaml --scene configs/scenes/multi_object.yaml --dry-run
python -m src.init_genesis   --config configs/genesis_config.yaml   --scene configs/scenes/household.yaml   --dry-run
```

Run with simulators:
```bash
bash scripts/run_omniverse.sh
bash scripts/run_genesis.sh
```

Multi-scene smoke test:
```bash
bash scripts/test_multiscene.sh
```
