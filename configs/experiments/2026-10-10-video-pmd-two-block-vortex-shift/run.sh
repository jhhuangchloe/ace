#!/bin/bash
# Local (non-Beaker) training of the two-block video PMD with the
# translation-aware (vortex-shift) baseline, DDP on 2 GPUs of this machine.
#
# One-time data copy (~190 GB, ~35 min at ~90 MB/s; resumable):
#   conda run -n fme python configs/experiments/2026-10-10-video-pmd-two-block-vortex-shift/prepare_data.py
#
# Run (vortex variant on GPUs 2,3,6,7):
#   bash configs/experiments/2026-10-10-video-pmd-two-block-vortex-shift/run.sh
# Control (same data/steps, linear baseline) -- run after, or on other GPUs:
#   VARIANT=control bash configs/experiments/2026-10-10-video-pmd-two-block-vortex-shift/run.sh
#   VARIANT=control GPUS=3,5 bash .../run.sh
# Other knobs: GPUS="2,3,4,6,7" (comma list; the configs use global batch_size 5, so
# use 1 or 5 GPUs, or edit batch_size; NCCL_SHM_DISABLE=1 helped when GPUs span
# both CPU sockets), ENV_NAME=fme, CONFIG=<path> to override the config.
set -euo pipefail

# GPUs spanning both CPU sockets hit a NCCL shared-memory error here without this.
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

VARIANT="${VARIANT:-vortex}"
GPUS="${GPUS:-2,3,4,6,7}"
ENV_NAME="${ENV_NAME:-fme}"
DATA_DIR="/data/pmd/local_data"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
case "$VARIANT" in
    vortex) DEFAULT_CONFIG="$SCRIPT_DIR/video_train.yaml" ;;
    control) DEFAULT_CONFIG="$SCRIPT_DIR/video_train_control.yaml" ;;
    *) echo "VARIANT must be 'vortex' or 'control', got '$VARIANT'" >&2; exit 1 ;;
esac
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
N_GPUS=$(awk -F, '{print NF}' <<<"$GPUS")

BATCH=$(awk '/^  batch_size:/ {print $2; exit}' "$CONFIG")
if [ $((BATCH % N_GPUS)) -ne 0 ]; then
    echo "batch_size $BATCH in $CONFIG is not divisible by $N_GPUS GPUs (GPUS=$GPUS)." >&2
    exit 1
fi

for store in 25km 100km; do
    if [ ! -d "$DATA_DIR/2026-07-14-X-SHiELD-AMIP-FME-3h-$store.zarr" ]; then
        echo "Missing $DATA_DIR/...-$store.zarr -- run prepare_data.py first." >&2
        exit 1
    fi
done

# Refuse to start on a GPU that something else is already using.
for g in ${GPUS//,/ }; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used" -gt 2000 ]; then
        echo "GPU $g already has ${used} MiB in use; refusing to start. Set GPUS=... or free it." >&2
        exit 1
    fi
done

EXPERIMENT_DIR=$(awk '/^experiment_dir:/ {print $2}' "$CONFIG")
mkdir -p "$EXPERIMENT_DIR"
echo "variant=$VARIANT gpus=$GPUS config=$CONFIG experiment_dir=$EXPERIMENT_DIR"

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -n "$ENV_NAME" \
    torchrun --standalone --nproc_per_node "$N_GPUS" \
    -m fme.downscaling.video_train "$CONFIG" 2>&1 | tee -a "$EXPERIMENT_DIR/train_stdout.log"
