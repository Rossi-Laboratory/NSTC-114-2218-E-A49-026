#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# Inference Demo Script for VLA-MoE Manipulation
# ------------------------------------------------------------------------------
# This script demonstrates how to run inference with the trained (or random-init)
# VLAMoE model. It accepts optional arguments:
#   1) command string (defaults to "pick up the red block and place it in the box")
#   2) image path (defaults to downloading a toy sample)
# Example:
#   bash scripts/inference_demo.sh "insert the peg into the hole" data/raw/sample.jpg
# ------------------------------------------------------------------------------

set -e

# --------- CONFIG ---------
INFER_CFG="configs/inference.yaml"
MODEL_CFG="configs/model_moe.yaml"
OUT_DIR="outputs/demo"
mkdir -p "${OUT_DIR}"

# --------- ARGS ---------
COMMAND=${1:-"pick up the red block and place it in the box"}
IMAGE_PATH=${2:-"${OUT_DIR}/toy_image.jpg"}

# --------- ENV CHECK ---------
echo "[INFO] Checking python environment..."
if ! command -v python &> /dev/null; then
  echo "[ERROR] Python not found. Please activate your venv first."
  exit 1
fi

# --------- SAMPLE IMAGE ---------
if [ ! -f "${IMAGE_PATH}" ]; then
  echo "[INFO] No input image found. Downloading a toy sample..."
  curl -L -o "${IMAGE_PATH}" https://picsum.photos/256/256
fi

# --------- RUN INFERENCE ---------
echo "[INFO] Running inference..."
python inference/run_inference.py \
  --config "${INFER_CFG}" \
  --model_cfg "${MODEL_CFG}" \
  --command "${COMMAND}" \
  --image "${IMAGE_PATH}" \
  --save_path "${OUT_DIR}/prediction.txt"

# --------- OPTIONAL VISUALIZATION ---------
if [ -f "${OUT_DIR}/prediction.txt" ]; then
  echo "[INFO] Inference result saved to ${OUT_DIR}/prediction.txt"
  echo "[INFO] Running visualization..."
  python - <<'PYCODE'
import os
from inference import InferenceSession, visualization

infer_cfg = "configs/inference.yaml"
model_cfg = "configs/model_moe.yaml"
sess = InferenceSession(infer_cfg, model_cfg, device="cpu")

command = """'"${COMMAND}"'"""
image_path = """'"${IMAGE_PATH}"'"""
action, weights = sess.predict(command, image_path=image_path)

save_img = os.path.join("outputs/demo", "vis_demo.png")
visualization.visualize_action(
    image_path,
    action,
    weights,
    command=command,
    save_path=save_img,
    show=False
)
print(f"[INFO] Visualization saved to {save_img}")
PYCODE
fi

echo "[INFO] Demo finished."
