# VLA-MoE Manipulation

**Mixture-of-Experts Vision-Language-Action Model for Robotic Manipulation**

> National Science and Technology Council (NSTC) Project  
> Project ID: **NSTC-114-2218-E-A49-026/**

---

## 📌 Overview
This repository contains the implementation of **VLA-MoE (Vision-Language-Action with Mixture-of-Experts)** for robotic manipulation.  
The goal is to design a **Mixture-of-Experts (MoE)** architecture that supports **multi-modal perception** (vision, language, proprioception, tactile) and **efficient action reasoning**.

**Key features**
- **MoE-based VLA architecture** with learnable routing (top-k) and expert mixing
- **Multi-Head Latent Attention** for compact cross-modal fusion
- **Expert modules** for *grasping*, *placement*, *tool-use*, and *tactile* behaviors
- **Training pipeline** with action regression + MoE balance/entropy regularizers
- **Evaluation & visualization** utilities for actions and expert weights
- **Config-driven** (`configs/*.yaml`) and **scriptable** (`scripts/*.sh`)

---

## 📂 Repository Structure
```
VLA-MoE-Manipulation/
├─ configs/                  # YAML configs for model & training
├─ data/                     # Dataset & tokenization helpers
├─ inference/                # Inference API, eval, visualization
├─ models/                   # Backbones, MoE gate, experts
│  └─ experts/               # Grasping / Placement / Tooluse / Tactile
├─ scripts/                  # Train / inference / preprocessing scripts
├─ training/                 # Trainer, losses, loops
├─ utils/                    # Checkpoint, logging, metrics
├─ requirements.txt
├─ setup.py
├─ environment.yaml
└─ README.md
```

---

## ⚙️ Installation (Linux / Omniverse friendly)
We recommend a Python **virtual environment** with pip.

```bash
python3 -m venv vla-moe-env
source vla-moe-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# (optional) enable console entry points
pip install -e .
```

> If you are on Omniverse / Linux without conda, the above is sufficient.

---

## 🗂️ Data Preparation
Prepare a raw JSON list where each item contains paths/labels for one step:

```json
[
  {
    "image": "data/raw/img_0001.png",
    "command": "pick up the red block",
    "proprio": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0],
    "tactile": [0.01, 0.02, 0.03, 0.00],
    "action":  [0.00, 0.00, -0.05, 0.00, 0.00, 0.00, 1.00]
  }
]
```

Then preprocess to a compact `.pt` file:
```bash
python scripts/preprocess_data.py   --input data/raw/train.json   --output data/processed/train.pt   --image_size 128 --vocab_size 512 --max_text_len 16
```

---

## 🚀 Training
Use the provided script (creates logs and checkpoints under `outputs/`).

```bash
bash scripts/train_moe.sh
```

Resume from a checkpoint:
```bash
bash scripts/train_moe.sh --resume outputs/checkpoint_best.pth
```

> The training entry point is also available as a console command when installed in editable mode:  
> `vlamoe-train --config configs/train.yaml --model_cfg configs/model_moe.yaml`

---

## 🔍 Inference & Visualization
Quick demo (downloads a sample image if none is given):

```bash
bash scripts/inference_demo.sh "place the block on the box" data/raw/sample.png
```

Programmatic usage:
```python
from inference import InferenceSession
sess = InferenceSession("configs/inference.yaml", "configs/model_moe.yaml", device="cpu")
action, weights = sess.predict("pick up the red block", image_path="data/raw/sample.png")
print(action, weights)
```

Visualization:
```python
from inference import visualization
visualization.visualize_action("data/raw/sample.png", action, weights,
                               command="pick up the red block",
                               save_path="outputs/demo/vis_demo.png",
                               show=False)
```

> The action is a 7D vector `[dx, dy, dz, droll, dpitch, dyaw, gripper]`.  
> Expert weights indicate the mixture contribution of each expert head.

---

## 📊 Evaluation
We provide utilities for batch metrics such as **success rate**, **MSE/RMSE/MAE**, **normalized error**, and **expert usage**.

```python
from inference.eval_manipulation import evaluate_batch, aggregate_results
metrics = evaluate_batch(pred, target, weights, thresh=0.25)
print(metrics)
```

---

## ⚙️ Configuration
- `configs/model_moe.yaml` — model dims, expert list/order, MoE top-k, latent attention heads.
- `configs/train.yaml` — dataset settings, batch size, epochs, lr/scheduler, loss weights.
- `configs/inference.yaml` — device, checkpoint path, and dataset defaults for pre/post-processing.

---

## 💾 Checkpoints
- Checkpoints are stored under `outputs/` as `checkpoint_epoch*.pth` and `checkpoint_best.pth`.
- Use `--resume <path>` to continue training or set `inference.yaml: checkpoint_path` for inference.

---

## 🧪 Tested Environment
- Python 3.9+ on Linux (Ubuntu)  
- PyTorch 2.2+ (CUDA optional)  
- Works in Omniverse environments using `pip` + venv

---

## 🪪 License
This project is licensed under the **Apache License 2.0**. See `LICENSE` if provided.

---

## 🙌 Acknowledgement
This work is supported by the **National Science and Technology Council (NSTC)**, Taiwan.  
**Project ID: NSTC-114-2218-E-A49-026/**
