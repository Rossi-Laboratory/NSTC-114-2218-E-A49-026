#!/usr/bin/env bash
set -e
mkdir -p third_party
cd third_party
if [ ! -d "VLA-MoE-Manipulation" ]; then
  echo "[INFO] Please clone your existing VLA-MoE-Manipulation here:"
  echo "       git clone <YOUR_VLA_REPO_URL> VLA-MoE-Manipulation"
else
  echo "[INFO] VLA-MoE-Manipulation already exists. You can 'git pull' inside it."
fi
