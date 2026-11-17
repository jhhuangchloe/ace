#!/bin/bash
# Cascade (temporal infill -> spatial SR) on the S2022 storm window, run locally on one GPU, then added to the
# comparison in /data/pmd/runs/eval-tc-v3-final (re-scores everything).
#   Stage 1: 5ch per-channel-OU 1-degree 24h->3h infill (/data/pmd/checkpoints/5ch-kernel), local 100 km store, 1 member
#   Stage 2: HiRO SR (/data/pmd/checkpoints/hiro-v6, EMA) on member 0 of stage 1, 4 members, regional S2022 patch
#
#   GPU=7 bash configs/experiments/2026-10-19-video-pmd-vortex-vs-two-block-eval/run_cascade_s2022.sh
# Knobs: GPU (7), N_ENS (4), MIN_FREE_GB (25), OUT (/data/pmd/runs/eval-tc-v3-final), ENV_NAME (fme).
# Resumable (both stages skip finished work). Stage 1 and 2 outputs: $OUT/cascade-work/.
set -euo pipefail

GPU="${GPU:-7}"
ENV_NAME="${ENV_NAME:-fme}"
N_ENS="${N_ENS:-4}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"
OUT="${OUT:-/data/pmd/runs/eval-tc-v3-final}"
START=2022-09-23; STOP=2022-09-29   # same frames as the other S2022 zarrs (09-23 00Z .. 09-29 21Z)
WORK="$OUT/cascade-work"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
REF="$OUT/rstt/S2022.zarr"
for f in /data/pmd/checkpoints/5ch-kernel/checkpoints/latest.ckpt /data/pmd/checkpoints/hiro-v6/checkpoints/ema_ckpt.tar "$REF"; do
    [ -e "$f" ] || { echo "Missing $f" >&2; exit 1; }
done

KMOD_VER=$(awk '/NVRM version/ {for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/) {print $i; exit}}' /proc/driver/nvidia/version 2>/dev/null || true)
LIBDIR=/usr/lib/x86_64-linux-gnu
if [ -n "$KMOD_VER" ] && [ -e "$LIBDIR/libnvidia-ml.so.$KMOD_VER" ] && [ "$(readlink -f "$LIBDIR/libnvidia-ml.so.1")" != "$(readlink -f "$LIBDIR/libnvidia-ml.so.$KMOD_VER")" ]; then
    SHIM="/tmp/nvshim-$KMOD_VER"; mkdir -p "$SHIM"
    ln -sf "$LIBDIR/libnvidia-ml.so.$KMOD_VER" "$SHIM/libnvidia-ml.so.1"
    ln -sf "$LIBDIR/libcuda.so.$KMOD_VER" "$SHIM/libcuda.so.1"
    export LD_LIBRARY_PATH="$SHIM${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

CHECK_PY='
import sys, warnings
warnings.filterwarnings("ignore")
import torch
free = torch.cuda.mem_get_info(int(sys.argv[1]))[0] / 2**30
print(f"GPU {sys.argv[1]}: {free:.1f} GB free")
sys.exit(0 if free >= float(sys.argv[2]) else "Not enough free GPU memory; refusing to start.")
'
conda run --no-capture-output -n "$ENV_NAME" python -c "$CHECK_PY" "$GPU" "$MIN_FREE_GB"

mkdir -p "$WORK"; cd "$REPO_ROOT"
# the same storm-centred patch as run_tc_eval_final.sh: lat 3..47, lon 106..178
conda run -n "$ENV_NAME" python "$SCRIPT_DIR/make_cascade_configs.py" "$WORK" "$START" "$STOP" "$N_ENS" 3 47 106 178

echo "=== stage 1: temporal infill $(date +%T) ==="
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$ENV_NAME" \
    python -m fme.downscaling.video_inference "$WORK/stage1.yaml" 2>&1 | tee -a "$WORK/stage1.log"
[ -d "$WORK/stage1-infill.zarr" ] || { echo "stage 1 produced no output; see $WORK/stage1.log" >&2; exit 1; }

echo "=== stage 2: SR $(date +%T) ==="
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$ENV_NAME" \
    python -m fme.downscaling.inference "$WORK/stage2.yaml" 2>&1 | tee -a "$WORK/stage2.log"
[ -d "$WORK/stage2/S2022.zarr" ] || { echo "stage 2 produced no output; see $WORK/stage2.log" >&2; exit 1; }

mkdir -p "$OUT/cascade"
conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT_DIR/cascade_to_zarr.py" \
    "$WORK/stage2/S2022.zarr" "$REF" "$OUT/cascade/S2022.zarr"
cp -f "$OUT/rstt/S2022.yaml" "$OUT/cascade/S2022.yaml"  # the plots read the storm-patch config from here

echo "=== scoring all models incl. cascade ==="
SKIP_INFERENCE=1 OUT="$OUT" bash "$SCRIPT_DIR/run_tc_eval_final.sh"
