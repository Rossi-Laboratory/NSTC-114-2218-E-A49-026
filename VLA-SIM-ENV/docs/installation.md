# Installation

## Conda (recommended)
```bash
conda env create -f environment.yaml
conda activate vla-sim-env
```

## PIP
```bash
pip install -r requirements.txt
```

## Omniverse / Isaac Sim
- Ensure access to Isaac Sim Python environment or add its site-packages to `PYTHONPATH`.
- Asset paths should be updated in `configs/scenes/*.yaml` to valid USD/URDF.

## Genesis
- Install the `genesis` Python package or follow the official binding instructions.
- Verify import: `python -c "import genesis"`.
