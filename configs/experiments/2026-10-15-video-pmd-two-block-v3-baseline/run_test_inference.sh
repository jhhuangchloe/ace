#!/bin/bash
# Full test-set inference (2023-01-01 .. 2024-01-04, global, 4 members) for a finished V3 two-block run.
# Waits until that run's TRAINING is finished, then runs inference on the GPUs the training used.
#
#   tmux new -s infer
#   MODEL=flat   bash configs/experiments/2026-10-15-video-pmd-two-block-v3-baseline/run_test_inference.sh
#   MODEL=vortex GPUS=2,3 bash .../run_test_inference.sh     # hybrid is already trained: starts right away
#   DRY_RUN=1 MODEL=flat bash .../run_test_inference.sh      # print what it sees / would do, then exit
#
# MODEL=flat    V3 + flat noise, no vortex   (train run .../video-pmd-two-block-v3-baseline-detail)
# MODEL=vortex  V3 + vortex blend + OU       (train run .../video-pmd-two-block-v3-vortex-detail-ou)
#
# "Training finished" = no training process of that run alive AND its latest.ckpt startEpoch >= max_epochs.
# If the process is gone but training is not finished (crash / restart) it keeps waiting (it does not give up),
# so restart training freely. Inference is resumable: rerun this script after any interruption and finished
# batches are skipped (each launch is marked "=== relaunch" in inference.log).
#
# GPUs: GPUS if set, else the CUDA_VISIBLE_DEVICES of the training process seen while it is alive, else
# 0,1,5,7 for flat. Global batch 16 (config) is split over the GPUs: it must divide 368 clips, so use 1, 2
# or 4 GPUs (16 / N GPUs per rank; ~35 GB at 4/GPU). 3 GPUs do not work.
#
# Knobs (env): MODEL (flat), GPUS, MIN_FREE_GB (default 50 at 4 clips/GPU, 75 at 8;
# waits until every GPU has that free), POLL (60),
# ENV_NAME (fme), DRY_RUN.
set -uo pipefail

MODEL="${MODEL:-flat}"
ENV_NAME="${ENV_NAME:-fme}"
MIN_FREE_GB="${MIN_FREE_GB:-}"   # default scales with clips/GPU, see below
POLL="${POLL:-60}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
case "$MODEL" in
    flat)
        CONFIG="$HERE/test_inference_flat_detail.yaml"
        TRAIN_DIR=/data/pmd/runs/video-pmd-two-block-v3-baseline-detail
        # [x] bracket trick: the regex does not match this script's own command line
        TRAIN_PATTERN='fme.downscaling.video_trai[n].*video_train_flat_bs8_4gpu[s]'
        DEFAULT_GPUS="0,1,5,7" ;;
    vortex)
        CONFIG="$HERE/test_inference_vortex_detail_ou.yaml"
        TRAIN_DIR=/data/pmd/runs/video-pmd-two-block-v3-vortex-detail-ou
        TRAIN_PATTERN='fme.downscaling.video_trai[n].*video_train_ou_vorte[x]'
        DEFAULT_GPUS="" ;;
    *) echo "MODEL must be flat or vortex, got '$MODEL'" >&2; exit 1 ;;
esac
OUT_ZARR=$(awk '/^output_path:/ {print $2}' "$CONFIG")
OUT_DIR=$(dirname "$OUT_ZARR")
LOG="$OUT_DIR/inference.log"
mkdir -p "$OUT_DIR"
slog() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT_DIR/watch.log"; }

# Host kernel module is older than the installed userspace NVIDIA libs: use the matching on-disk ones
# (nvidia-smi for the watcher, NCCL/torch for the inference).
KMOD_VER=$(awk '/NVRM version/ {for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/) {print $i; exit}}' /proc/driver/nvidia/version 2>/dev/null)
LIBDIR=/usr/lib/x86_64-linux-gnu
if [ -n "$KMOD_VER" ] && [ -e "$LIBDIR/libnvidia-ml.so.$KMOD_VER" ]; then
    SHIM="/tmp/nvshim-$KMOD_VER"
    mkdir -p "$SHIM"
    ln -sf "$LIBDIR/libnvidia-ml.so.$KMOD_VER" "$SHIM/libnvidia-ml.so.1"
    export LD_LIBRARY_PATH="$SHIM${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

train_pid() { pgrep -f "$TRAIN_PATTERN" | head -1; }
train_gpus() {
    local p; p=$(train_pid)
    [ -n "$p" ] && tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'
}
max_epochs() { awk '/^max_epochs:/ {print $2; exit}' "$TRAIN_DIR/config.yaml" 2>/dev/null; }
ckpt_start_epoch() {
    local ck="$TRAIN_DIR/checkpoints/latest.ckpt"
    [ -f "$ck" ] || { echo -1; return; }
    conda run -n "$ENV_NAME" python -c "
import sys, torch
print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['startEpoch']))" "$ck" 2>/dev/null | grep -E '^[0-9]+$' | tail -1
}
train_finished() {
    local need have; need=$(max_epochs); have=$(ckpt_start_epoch)
    [ -n "$need" ] && [ -n "$have" ] && [ "$have" -ge "$need" ] 2>/dev/null
}
free_ok() {  # every GPU in $1 (comma list) has >= MIN_FREE_GB free
    local g used total
    for g in ${1//,/ }; do
        read -r used total < <(nvidia-smi -i "$g" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ',')
        [ -n "${used:-}" ] || return 1
        [ $(( (total - used) / 1024 )) -ge "$MIN_FREE_GB" ] || return 1
    done
}

slog "MODEL=$MODEL: waiting for training in $TRAIN_DIR (max_epochs=$(max_epochs)), then inference with $(basename "$CONFIG")"
LAST_GPUS=""
while true; do
    pid=$(train_pid)
    if [ -n "$pid" ]; then
        g=$(train_gpus); [ -n "$g" ] && LAST_GPUS="$g"
        if [ "${DRY_RUN:-0}" = "1" ]; then
            slog "DRY RUN: training alive (pid $pid) on GPUs ${LAST_GPUS:-?}; latest.ckpt startEpoch=$(ckpt_start_epoch)/$(max_epochs)"
            break
        fi
    elif train_finished; then
        slog "training finished (startEpoch >= max_epochs)."
        break
    else
        if [ "${DRY_RUN:-0}" = "1" ]; then slog "DRY RUN: training not running and not finished"; break; fi
        slog "training process gone but not finished (startEpoch=$(ckpt_start_epoch)/$(max_epochs)); waiting for a restart..."
    fi
    sleep "$POLL"
done

GPUS="${GPUS:-${LAST_GPUS:-$DEFAULT_GPUS}}"
if [ -z "$GPUS" ]; then echo "Could not determine GPUs; set GPUS=a,b,c,d" >&2; exit 1; fi
N_GPUS=$(awk -F, '{print NF}' <<<"$GPUS")
BATCH=$(awk '/^  batch_size:/ {print $2; exit}' "$CONFIG")
if [ $((BATCH % N_GPUS)) -ne 0 ] || [ $((368 % BATCH)) -ne 0 ]; then
    echo "batch_size $BATCH (368 clips) does not work with $N_GPUS GPUs (GPUS=$GPUS): use 1, 2 or 4 GPUs." >&2
    exit 1
fi
[ -n "$MIN_FREE_GB" ] || MIN_FREE_GB=$(( BATCH / N_GPUS <= 4 ? 50 : (BATCH / N_GPUS <= 8 ? 75 : 150) ))
CKPT=$(awk '/^checkpoint_path:/ {print $2}' "$CONFIG")
[ -f "$CKPT" ] || { echo "Missing checkpoint $CKPT" >&2; exit 1; }
slog "inference: GPUS=$GPUS ($N_GPUS ranks, $((BATCH / N_GPUS)) clips/GPU) ckpt=$CKPT out=$OUT_ZARR"
if [ "${DRY_RUN:-0}" = "1" ]; then
    slog "DRY RUN: would wait for >= ${MIN_FREE_GB} GB free on GPUs $GPUS, then run torchrun video_inference"
    exit 0
fi

until free_ok "$GPUS"; do
    slog "waiting for >= ${MIN_FREE_GB} GB free on every GPU in $GPUS ..."
    sleep "$POLL"
done

cd "$REPO_ROOT"
echo "=== relaunch $N_GPUS GPUs batch $BATCH $(date) ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$GPUS" NCCL_SHM_DISABLE=1 PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -n "$ENV_NAME" \
    torchrun --standalone --nproc_per_node "$N_GPUS" -m fme.downscaling.video_inference "$CONFIG" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
if grep -q "Completed inference" <(awk '/=== relaunch/ {buf=""} {buf = buf $0 "\n"} END {printf "%s", buf}' "$LOG"); then
    slog "inference completed: $OUT_ZARR"
else
    slog "inference exited (status $status) WITHOUT 'Completed inference'; rerun this script to resume."
    exit 1
fi
