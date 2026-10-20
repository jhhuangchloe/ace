#!/bin/bash
# Shortcut: train the climate-adapted V3 (separate JAX env + repo at $V3_DIR, branch climate-adapt).
# Two coarse 100 km endpoints in -> 9 fine 25 km frames out; a candidate temporal baseline for the
# two-block diffusion model. See $V3_DIR/README_CLIMATE.md.
#
# Run in tmux:   bash configs/experiments/2026-10-15-v3-climate-baseline/run.sh
# Knobs (env):   GPUS=2,3  N_ITER=100000  TAG=v3-climate-9yr  OUT_DIR=/data/pmd/runs/$TAG  EXTRA="--lr 1e-4"
# Resumes from $OUT_DIR/latest.pkl. Evaluate on the TC windows with:
#   bash configs/experiments/2026-10-15-v3-climate-baseline/eval.sh
set -euo pipefail
V3_DIR="${V3_DIR:-$HOME/Projects/v3}"
exec bash "$V3_DIR/run_climate.sh" "$@"
