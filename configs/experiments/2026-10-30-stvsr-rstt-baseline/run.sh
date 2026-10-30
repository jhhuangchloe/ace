#!/bin/bash
# Local DDP training of the RSTT-S space-time SR baseline (see pretrain.yaml's header).
#
#   bash configs/experiments/2026-10-30-stvsr-rstt-baseline/run.sh                    # 1. pretrain
#   STAGE=finetune bash configs/experiments/2026-10-30-stvsr-rstt-baseline/run.sh     # 2. fine-tune
#   STAGE=scratch GPUS=3,4 bash configs/experiments/2026-10-30-stvsr-rstt-baseline/run.sh  # control
#
# Knobs: STAGE (pretrain|finetune|scratch), GPUS="1,3" (comma list; configs use global batch_size
# 16, so 1, 2, 4, 8 or 16 GPUs), ENV_NAME=fme, CONFIG=<path> to override the config.
# The data must already be local: ../2026-10-10-video-pmd-two-block-vortex-shift/prepare_data.py.
set -euo pipefail

# GPUs spanning both CPU sockets hit a NCCL shared-memory error here without this.
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

STAGE="${STAGE:-pretrain}"
GPUS="${GPUS:-1,3}"
ENV_NAME="${ENV_NAME:-fme}"
DATA_DIR="/data/pmd/local_data"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
case "$STAGE" in
    pretrain|finetune|scratch) DEFAULT_CONFIG="$SCRIPT_DIR/$STAGE.yaml" ;;
    *) echo "STAGE must be pretrain, finetune or scratch, got '$STAGE'" >&2; exit 1 ;;
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

INIT=$(awk '/^init_weights_from:/ {print $2}' "$CONFIG")
EXPERIMENT_DIR=$(awk '/^experiment_dir:/ {print $2}' "$CONFIG")
if [ -n "$INIT" ] && [ ! -f "$INIT" ] && [ ! -f "$EXPERIMENT_DIR/checkpoints/latest.ckpt" ]; then
    echo "init_weights_from $INIT does not exist -- run the pretraining stage first." >&2
    exit 1
fi

# Refuse to start on a GPU that something else is already using.
for g in ${GPUS//,/ }; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used" -gt 2000 ]; then
        echo "GPU $g already has ${used} MiB in use; refusing to start. Set GPUS=... or free it." >&2
        exit 1
    fi
done

mkdir -p "$EXPERIMENT_DIR"
echo "stage=$STAGE gpus=$GPUS config=$CONFIG experiment_dir=$EXPERIMENT_DIR"

cd "$REPO_ROOT"
# PYTHONPATH: run this checkout's fme even if the env's editable install points at another one.
CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 PYTHONPATH="$REPO_ROOT" \
    conda run --no-capture-output -n "$ENV_NAME" \
    torchrun --standalone --nproc_per_node "$N_GPUS" \
    -m fme.downscaling.video_train "$CONFIG" 2>&1 | tee -a "$EXPERIMENT_DIR/train_stdout.log"
