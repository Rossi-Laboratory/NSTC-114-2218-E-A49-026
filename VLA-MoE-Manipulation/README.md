# VLA-MoE-Manipulation

Vision–Language–Action (VLA) model with a Mixture-of-Experts (MoE) architecture tailored for **robotic manipulation**.
This repository includes a minimal yet runnable skeleton for training and inference with synthetic data, so you can verify the pipeline end-to-end before wiring in your real datasets.

> Inputs: RGB image(s), language command, proprioception/tactile states  
> Outputs: 6-DoF end-effector pose deltas + gripper action (7D)

## Highlights
- **MoE Experts** for grasping, placement, tool-use, and force/tactile strategies
- **Multi-head Latent Attention** for cross-modal fusion (vision, language, proprioception, tactile)
- **Config-driven** training & inference (`configs/*.yaml`)
- **Synthetic dataset** to validate the code path out-of-the-box

---

## Installation

```bash
# (Recommended) Create a fresh virtualenv or conda env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> Tested with Python 3.9–3.11.

## Quick Start (Synthetic Demo)

```bash
# Train on synthetic data
python training/train.py --config configs/train.yaml

# Run a toy inference demo
python inference/run_inference.py --config configs/inference.yaml \
  --command "pick up the red block and place it in the box"
```

Artifacts:
- checkpoints in `outputs/checkpoints/`
- logs/metrics in console prints

## Project Structure

```
VLA-MoE-Manipulation/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── train.yaml
│   ├── inference.yaml
│   └── model_moe.yaml
├── data/
│   ├── raw/              # (your data here)
│   ├── processed/        # (your data here)
│   ├── dataset.py
│   └── __init__.py
├── models/
│   ├── __init__.py
│   ├── moe_gate.py
│   ├── multihead_latent_attn.py
│   ├── vla_moe.py
│   └── experts/
│       ├── __init__.py
│       ├── grasping_expert.py
│       ├── placement_expert.py
│       ├── tooluse_expert.py
│       └── tactile_expert.py
├── training/
│   ├── __init__.py
│   ├── train.py
│   ├── trainer.py
│   └── loss_functions.py
├── inference/
│   ├── __init__.py
│   ├── run_inference.py
│   ├── eval_manipulation.py
│   └── visualization.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── checkpoint.py
│   ├── metrics.py
│   └── seed.py
└── scripts/
    ├── preprocess_data.py
    ├── train_moe.sh
    └── inference_demo.sh
```

## Notes
- The code uses simple token hashing for text and a tiny CNN for vision to remain self-contained.
- Replace the `SyntheticManipulationDataset` with your dataloader in `data/dataset.py`.
- Extend experts or add new ones by creating files under `models/experts/` and wiring them in `models/vla_moe.py`/`models/moe_gate.py`.

## License
Apache-2.0 (see `LICENSE` in a production repository). This demo is provided "as is" with no warranty.
