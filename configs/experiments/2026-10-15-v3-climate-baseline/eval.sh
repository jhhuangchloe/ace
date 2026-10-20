#!/bin/bash
# Evaluate a trained V3 on the 22 real TC windows vs the linear / vortex-shift baselines.
#   GPU=3 bash configs/experiments/2026-10-15-v3-climate-baseline/eval.sh [best|latest]
set -euo pipefail
V3_DIR="${V3_DIR:-$HOME/Projects/v3}"
TAG="${TAG:-v3-climate-9yr}"
OUT_DIR="${OUT_DIR:-/data/pmd/runs/$TAG}"
WHICH="${1:-best}"
cd "$V3_DIR"
CUDA_VISIBLE_DEVICES="${GPU:-3}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    conda run --no-capture-output -n "${ENV_NAME:-v3}" python eval_tc.py \
    --ckpt "$OUT_DIR/$WHICH.pkl" --windows assets_for_eval/tc_windows.npz \
    --baselines assets_for_eval/res_phase.json --out "$OUT_DIR/res_tc_$WHICH.json"
