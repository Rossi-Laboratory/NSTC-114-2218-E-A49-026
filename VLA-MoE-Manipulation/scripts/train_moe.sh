#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# Train MoE-based VLA Manipulation Model
# ------------------------------------------------------------------------------
# This script launches training for VLAMoE with the given configs.
# It also checks GPU availability, manages logs, and supports resume.
#
# Usage:
#   bash scripts/train_moe.sh
#   bash scripts/train_moe.sh --resume outputs/checkpoint_last.pth
#
# All logs will be saved under outputs/logs/YYYYMMDD_HHMM/.
# ------------------------------------------------------------------------------

set -e

# --------- CONFIG ---------
TRAIN_CFG="configs/train.yaml"
MODEL_CFG="configs/model_moe.yaml"
OUT_DIR="outputs"
LOG_DIR="${OUT_DIR}/logs/$(date +%Y%m%d_%H%M)"
mkdir -p "${LOG_DIR}"

# --------- ARGUMENTS ---------
RESUME_PATH=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --resume)
      RESUME_PATH="$2"
      shift 2
      ;;
    *)
      echo "[WARN] Unknown argument: $1"
      shift
      ;;
  esac
done

# --------- ENV CHECK ---------
echo "[INFO] Checking environment..."
if command -v nvidia-smi &> /dev/null; then
  echo "[INFO] Detected GPU(s):"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "[WARN] No NVIDIA GPU detected, training will run on CPU."
fi

if ! command -v python &> /dev/null; then
  echo "[ERROR] Python not found. Please activate your environment."
  exit 1
fi

# --------- TRAINING ---------
echo "[INFO] Starting training..."
CMD="python training/train.py --config ${TRAIN_CFG} --model_cfg ${MODEL_CFG} --out_dir ${OUT_DIR}"
if [ -n "${RESUME_PATH}" ]; then
  CMD="${CMD} --resume ${RESUME_PATH}"
fi

echo "[INFO] Running: ${CMD}"
${CMD} 2>&1 | tee "${LOG_DIR}/train.log"

echo "[INFO] Training finished. Logs saved in ${LOG_DIR}"
