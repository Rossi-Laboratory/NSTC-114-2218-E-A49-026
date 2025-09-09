#!/usr/bin/env bash
set -e
python -m acp.inference.visualize --input outputs/last_infer.json --save outputs/vis_chunks.png
