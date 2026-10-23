#!/bin/bash
# TMP: resume the vortex-shift run on GPUs 2,3,4,5 (global batch 4, 1 sample/GPU)
# using video_train_tmp_gpus2-5.yaml. Resumes from <experiment_dir>/checkpoints/latest.ckpt.
# Differs from run.sh: nvidia-smi is broken on this host (driver/library mismatch), so the
# free-memory check uses torch.cuda.mem_get_info instead.
#
#   bash configs/experiments/2026-10-10-video-pmd-two-block-vortex-shift/run_tmp_gpus2-5.sh
# Knobs: MIN_FREE_GB (default 45; ~38 GB peak per GPU at 1 sample/GPU), ENV_NAME, GPUS, CONFIG.
set -euo pipefail

export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

# Host has a kernel module (580.95.05) older than the installed userspace libs (580.173.02),
# which breaks NVML (and so NCCL). The old 580.95.05 libs are still on disk: put them first.
KMOD_VER=$(awk '/NVRM version/ {for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/) {print $i; exit}}' /proc/driver/nvidia/version)
LIBDIR=/usr/lib/x86_64-linux-gnu
if [ -e "$LIBDIR/libnvidia-ml.so.$KMOD_VER" ] && [ "$(readlink -f "$LIBDIR/libnvidia-ml.so.1")" != "$(readlink -f "$LIBDIR/libnvidia-ml.so.$KMOD_VER")" ]; then
    SHIM="/tmp/nvshim-$KMOD_VER"
    mkdir -p "$SHIM"
    ln -sf "$LIBDIR/libnvidia-ml.so.$KMOD_VER" "$SHIM/libnvidia-ml.so.1"
    ln -sf "$LIBDIR/libcuda.so.$KMOD_VER" "$SHIM/libcuda.so.1"
    export LD_LIBRARY_PATH="$SHIM${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "Using NVIDIA libs $KMOD_VER from $SHIM (matching the loaded kernel module)"
fi

GPUS="${GPUS:-2,3,4,5}"
ENV_NAME="${ENV_NAME:-fme}"
MIN_FREE_GB="${MIN_FREE_GB:-45}"
DATA_DIR="/data/pmd/local_data"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
CONFIG="${CONFIG:-$SCRIPT_DIR/video_train_tmp_gpus2-5.yaml}"
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

# Refuse to start unless every requested GPU has enough free memory.
CHECK_PY='
import sys, warnings
warnings.filterwarnings("ignore")
import torch
gpus = [int(g) for g in sys.argv[1].split(",")]
need = float(sys.argv[2])
bad = False
for g in gpus:
    free = torch.cuda.mem_get_info(g)[0] / 2**30
    ok = free >= need
    print(f"GPU {g}: {free:.1f} GB free", "OK" if ok else f"< {need:.0f} GB NEEDED")
    bad |= not ok
if bad:
    sys.exit("Not enough free memory on some GPUs; refusing to start.")
'
conda run --no-capture-output -n "$ENV_NAME" python -c "$CHECK_PY" "$GPUS" "$MIN_FREE_GB"

EXPERIMENT_DIR=$(awk '/^experiment_dir:/ {print $2}' "$CONFIG")
mkdir -p "$EXPERIMENT_DIR"
echo "gpus=$GPUS config=$CONFIG experiment_dir=$EXPERIMENT_DIR"
ls -l "$EXPERIMENT_DIR/checkpoints/latest.ckpt" 2>/dev/null || echo "no latest.ckpt in $EXPERIMENT_DIR: starting from scratch"

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -n "$ENV_NAME" \
    torchrun --standalone --nproc_per_node "$N_GPUS" \
    -m fme.downscaling.video_train "$CONFIG" 2>&1 | tee -a "$EXPERIMENT_DIR/train_stdout.log"
