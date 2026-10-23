#!/bin/bash
# Compare the vortex-shift two-block model against the original two-block
# method (linear-interpolation baseline; /data/pmd/checkpoints/two-block-flat)
# on one GPU: ensemble inference over a few short windows of summer 2022 (global,
# 25km, divide_generation), then
# fair-CRPS / RMSE / MAE / spread-skill scoring against truth.
#
#   GPU=4 bash configs/experiments/2026-10-19-video-pmd-vortex-vs-two-block-eval/run_eval.sh
#   SMOKE=1 GPU=4 bash .../run_eval.sh        # 1 window x 1 day x 2 members, to time it first
#
# Knobs: GPU (default 4), N_ENS (4), DAYS (2 days per window), WINDOWS (space-
# separated start dates), CKPT_NAME (latest.ckpt; best.ckpt to compare best),
# OUT (results dir), MIN_FREE_GB (60), ENV_NAME (fme).
# Inference is resumable: rerun after an interruption and finished batches are skipped.
set -euo pipefail

GPU="${GPU:-4}"
ENV_NAME="${ENV_NAME:-fme}"
N_ENS="${N_ENS:-4}"
DAYS="${DAYS:-2}"
# Local data only covers 2013-2015 and 2022-06-01..2022-09-30 (2023 was never copied).
WINDOWS="${WINDOWS:-2022-06-10 2022-07-20 2022-08-28 2022-09-20}"
CKPT_NAME="${CKPT_NAME:-latest.ckpt}"
MIN_FREE_GB="${MIN_FREE_GB:-60}"
OUT="${OUT:-/data/pmd/runs/eval-vortex-vs-two-block}"
if [ "${SMOKE:-0}" = "1" ]; then
    WINDOWS="2022-08-28"; DAYS=1; N_ENS=2; OUT="${OUT}-smoke"
fi

RUNS=/data/pmd/runs
TRUTH=/data/pmd/local_data/2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
# label -> training config (model/test_data blocks) and run dir (checkpoint)
declare -A TRAIN_CFG=(
    [vortex]="$REPO_ROOT/configs/experiments/2026-10-10-video-pmd-two-block-vortex-shift/video_train.yaml"
    [two-block]="$REPO_ROOT/configs/experiments/2026-10-10-video-pmd-two-block-vortex-shift/video_train_control.yaml"
)
declare -A RUN_DIR=(
    [vortex]="$RUNS/video-pmd-two-block-vortex-shift"
    [two-block]="/data/pmd/checkpoints/two-block-flat"
)
MODELS=(vortex two-block)  # first = model under test

# Host kernel module (580.95.05) is older than the installed userspace libs
# (580.173.02), which breaks NVML; put the matching on-disk libs first.
KMOD_VER=$(awk '/NVRM version/ {for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/) {print $i; exit}}' /proc/driver/nvidia/version)
LIBDIR=/usr/lib/x86_64-linux-gnu
if [ -e "$LIBDIR/libnvidia-ml.so.$KMOD_VER" ] && [ "$(readlink -f "$LIBDIR/libnvidia-ml.so.1")" != "$(readlink -f "$LIBDIR/libnvidia-ml.so.$KMOD_VER")" ]; then
    SHIM="/tmp/nvshim-$KMOD_VER"
    mkdir -p "$SHIM"
    ln -sf "$LIBDIR/libnvidia-ml.so.$KMOD_VER" "$SHIM/libnvidia-ml.so.1"
    ln -sf "$LIBDIR/libcuda.so.$KMOD_VER" "$SHIM/libcuda.so.1"
    export LD_LIBRARY_PATH="$SHIM${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

for m in "${MODELS[@]}"; do
    [ -f "${RUN_DIR[$m]}/checkpoints/$CKPT_NAME" ] || { echo "Missing ${RUN_DIR[$m]}/checkpoints/$CKPT_NAME" >&2; exit 1; }
done

# Refuse to start unless the GPU has enough free memory.
CHECK_PY='
import sys, warnings
warnings.filterwarnings("ignore")
import torch
free = torch.cuda.mem_get_info(int(sys.argv[1]))[0] / 2**30
print(f"GPU {sys.argv[1]}: {free:.1f} GB free")
sys.exit(0 if free >= float(sys.argv[2]) else "Not enough free GPU memory; refusing to start.")
'
conda run --no-capture-output -n "$ENV_NAME" python -c "$CHECK_PY" "$GPU" "$MIN_FREE_GB"

mkdir -p "$OUT"
cd "$REPO_ROOT"
for m in "${MODELS[@]}"; do
    for start in $WINDOWS; do
        zarr="$OUT/$m/$start.zarr"
        cfg="$OUT/$m/$start.yaml"
        mkdir -p "$OUT/$m"
        conda run -n "$ENV_NAME" python "$SCRIPT_DIR/make_inference_config.py" \
            "${TRAIN_CFG[$m]}" "${RUN_DIR[$m]}/checkpoints/$CKPT_NAME" "$zarr" \
            "$OUT/$m/logs-$start" "$start" "$DAYS" "$N_ENS" "$cfg"
        echo "=== $m $start (days=$DAYS n_ens=$N_ENS) $(date +%T) ==="
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
            conda run --no-capture-output -n "$ENV_NAME" \
            python -m fme.downscaling.video_inference "$cfg" 2>&1 | tee -a "$OUT/$m/inference.log"
    done
done

conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT_DIR/score.py" \
    --root "$OUT" --models "${MODELS[@]}" --truth "$TRUTH" 2>&1 | tee "$OUT/scores.txt"
echo "Results in $OUT (scores.txt, scores_summary.csv, scores_per_window.csv)"
