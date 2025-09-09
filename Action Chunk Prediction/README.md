# Action Chunk Prediction (ACP)

**Goal**: Improve a robot’s ability to learn **multi-step behaviors** by predicting **action chunks** instead of step-by-step actions.  
**Objective**: Design an **Action Chunk Prediction** framework. We **reference FAST** (chunk boundary & aggregation) and **Vector-Quantized (VQ) Action Chunking** to build the initial architecture and codebase.

> **Important**: This project **uses your existing `VLA-MoE-Manipulation`** repository for low-level action inference. **Do not** re-implement VLA here. All interactions go through `acp/integration/vla_moe_bridge.py`.

---

## 🔧 What ACP Does
- Encodes an action sequence with a temporal backbone (Transformer or TCN).  
- Predicts **chunk boundaries** (FAST-style boundary detection).  
- Aggregates segments into **chunk embeddings** (segment-mean pooling).  
- Optionally **quantizes** chunk embeddings into discrete **VQ codes** (codebook).  
- (Optional) **CL3** hierarchical chunking: **atomic → micro-chunk → macro-chunk**.

ACP focuses on **efficient sequence learning** and **long-horizon behavior** by operating at chunk level.

---

## 📦 Repository Structure
```
Action-Chunk-Prediction/
├─ configs/                 # Model / training / inference configurations
├─ acp/
│  ├─ integration/          # Bridge to VLA-MoE-Manipulation (no re-implementation)
│  ├─ models/               # Temporal encoder, boundary detector, VQ, FAST aggregator
│  ├─ data/                 # Dataset loader / collate
│  ├─ training/             # Trainer, train loop
│  ├─ inference/            # Inference, decode, visualization
│  └─ utils/                # Logger, checkpoint, seed, config
├─ scripts/                 # Preprocess / train / infer / eval / submodule helper
├─ docs/                    # DESIGN / DATA_FORMAT / EVALUATION / INTEGRATION
├─ examples/manipulation/   # Manipulation demo sequences and commands
├─ data/                    # Default data folders (raw/processed/cache)
└─ third_party/
   └─ VLA-MoE-Manipulation/ # Your existing VLA repo (git submodule or manual clone)
```

---

## 🚀 Quickstart (Linux / Omniverse-friendly)

### 0) (Once) Place your VLA repo
```bash
# Option A: manual
mkdir -p third_party
git clone <YOUR_VLA_MOE_REPO_URL> third_party/VLA-MoE-Manipulation

# Option B: helper script (prints instructions)
bash scripts/prepare_submodule.sh
```

### 1) Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2) Manipulation Example (included in this repo)
We provide a tiny toy sequence under `examples/manipulation/sequences/demo_seq_01.json` (7D actions over time).  
You can convert it into a learnable tensor dataset, train ACP, then infer and visualize boundaries.

```bash
# Convert JSON -> torch tensors (.pt)
python scripts/preprocess_sequences.py   --input examples/manipulation/sequences/demo_seq_01.json   --output data/processed/train_sequences.pt

# Train ACP (boundary + optional VQ)
bash scripts/train_acp.sh

# Run inference (predict chunk boundaries + VQ indices) and visualize
bash scripts/infer_acp.sh
bash scripts/eval_acp.sh  # creates outputs/vis_chunks.png
```

> **Note**: For real use, export multi-step trajectories from your **VLA-MoE-Manipulation** inference and convert them to the same JSON format. ACP only handles chunking and sequence modeling.

---

## 🗂️ Data Format

**Input JSON** (`examples/manipulation/sequences/demo_seq_01.json`):
```json
{
  "actions": [
    [0.0, 0.0, 0.01, 0, 0, 0, 0],
    [0.0, 0.0, 0.01, 0, 0, 0, 0],
    "... continued ..."
  ]
}
```
- `actions`: `T × 7` continuous low-level actions (e.g., `[dx, dy, dz, droll, dpitch, dyaw, gripper]`).  
- ACP’s `scripts/preprocess_sequences.py` will create a toy boundary label (periodic, for demo) and save a `.pt` file containing:
  - `seqs`: `[N, T, D]` float tensor
  - `boundaries`: `[N, T]` binary tensor

**Run preprocessing**
```bash
python scripts/preprocess_sequences.py   --input examples/manipulation/sequences/demo_seq_01.json   --output data/processed/train_sequences.pt
```

---

## ⚙️ Configuration
- `configs/model_acp.yaml`: model size, use_vq, codebook size, boundary threshold  
- `configs/train.yaml`: batch size, epochs, learning rate, loss weights  
- `configs/inference.yaml`: checkpoint path, decoding parameters  

Example (abridged):
```yaml
model:
  name: ActionChunkPredictor
  d_model: 256
  use_vq: true
  codebook_size: 128
  boundary_threshold: 0.5
encoder:
  type: transformer
  depth: 4
  heads: 4
  dropout: 0.1
training:
  batch_size: 16
  epochs: 10
  lr: 3.0e-4
```

---

## 🧠 Architecture (ACP core)
1. **Temporal Encoder** (Transformer): encodes `[B, T, D]` sequences → contextualized features.  
2. **Boundary Detector** (FAST-style head): predicts logits `[B, T]` for chunk **end** positions.  
3. **FAST Aggregation**: segment-mean pooling → chunk embeddings `[B, N, D]`.  
4. **VQ (optional)**: discretize embeddings into code indices (straight-through estimator).  
5. **(Optional) CL3**: see `acp/models/hierarchical_chunker.py` for 3-level boundaries and VQ.

We **do not** implement a VLA model here. When you need to **decode or simulate** long action sequences, use `acp/integration/vla_moe_bridge.py` to call your existing **VLA-MoE-Manipulation**.

---

## 📊 Evaluation
- **Boundary quality**: visualize probability curves; compute BCE during training.  
- **Reconstruction** (if applied): measure MSE/RMSE/MAE vs. original actions after decoding.  
- **Chunk statistics**: code usage, average chunk length, top-k distribution.

Run the toy visualization:
```bash
bash scripts/infer_acp.sh
bash scripts/eval_acp.sh  # generates outputs/vis_chunks.png
```

---

## 🧩 Integrating with VLA-MoE-Manipulation
- Place your existing **VLA-MoE-Manipulation** under `third_party/`.  
- Use the bridge `acp/integration/vla_moe_bridge.py` to import its `InferenceSession`.  
- All low-level action inference must come from that repo. **Do not** re-implement VLA here.

---

## 🪪 License
This repository is released under the **Apache License 2.0**. See `LICENSE`.

