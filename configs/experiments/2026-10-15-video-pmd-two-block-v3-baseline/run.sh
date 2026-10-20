#!/bin/bash
# Local training of the two-block video PMD with the learned V3 temporal baseline.
# Same data / schedule / launcher as the vortex-shift and linear-control runs; only the
# config differs. Run in tmux:
#   GPUS=2,3,4,6,7 bash configs/experiments/2026-10-15-video-pmd-two-block-v3-baseline/run.sh
# (global batch_size 5 in the config -> use 1 or 5 GPUs, or edit batch_size; see the
# vortex-shift run.sh for the other knobs: ENV_NAME, NCCL_SHM_DISABLE, ...)
# Control = linear baseline, same schedule:
#   VARIANT=control bash configs/experiments/2026-10-10-video-pmd-two-block-vortex-shift/run.sh
#
# Before the first launch (V3 must be trained on the same 9 years as the diffusion model):
#   1. train V3:   GPUS=2,3 bash ~/Projects/v3/run_climate.sh   # 100k steps, ~17 h (data-loading bound)
#   2. export:     conda run -n v3 python scripts/v3_baseline/export_params_npz.py \
#                      /data/pmd/runs/v3-climate-9yr/best.pkl /data/pmd/v3_baseline/params_9yr.npz
#   3. convert:    PYTHONPATH=. python scripts/v3_baseline/convert_checkpoint.py \
#                      /data/pmd/v3_baseline/params_9yr.npz /data/pmd/v3_baseline/v3_baseline_9yr.pt
#   4. re-estimate r_sigma_data_by_channel and the OU lengths (scripts/v3_baseline/r_statistics.py,
#      fit_r_kernel.py) and put them in the two yaml files.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V3_WEIGHTS="${V3_WEIGHTS:-/data/pmd/v3_baseline/v3_baseline_9yr.pt}"
if [ ! -f "$V3_WEIGHTS" ]; then
    echo "Missing $V3_WEIGHTS -- convert the V3 checkpoint first (scripts/v3_baseline/)." >&2
    exit 1
fi
CONFIG="${CONFIG:-$HERE/video_train.yaml}" \
    exec bash "$HERE/../2026-10-10-video-pmd-two-block-vortex-shift/run.sh" "$@"
