#!/bin/bash
# Tropical-cyclone evaluation of the vortex-shift model vs the original two-block
# model on known storms (tracks 789 intense / 795 moderate / 829 mild, 2023, from
# scripts/tropical_cyclones/two_block_family_analysis.py). Storm-centred regional
# ensemble inference (one 44x72 coarse patch per storm), then tc_score.py: SLP error,
# max-wind error, centre-position error, radial structure scorecard.
#
#   GPU=4 bash configs/experiments/2026-10-19-video-pmd-vortex-vs-two-block-eval/run_tc_eval.sh
#   STORMS="T789 T795 T829" GPU=4 bash .../run_tc_eval.sh   # the three 2023 storms
#
# S2022 (default) uses only the 2022 data already local; the 2023 storms T789/T795/T829 need
# prepare_data.py --val-stop 2024-01-04 to have finished.
# Knobs: GPU (4), N_ENS (4), STORMS (default S2022; also T789 T795 T829), CKPT_NAME (latest.ckpt),
# BATCH (clips per batch, 2), MIN_FREE_GB (25), OUT, ENV_NAME. Inference is resumable.
set -euo pipefail

GPU="${GPU:-4}"
ENV_NAME="${ENV_NAME:-fme}"
N_ENS="${N_ENS:-4}"
STORMS="${STORMS:-S2022}"  # 2022 typhoon (data available locally); T789/T795/T829 need the 2023 data
CKPT_NAME="${CKPT_NAME:-latest.ckpt}"
BATCH="${BATCH:-2}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"
OUT="${OUT:-/data/pmd/runs/eval-tc-vortex-vs-two-block}"

# storm -> "start stop lat0 lon0". The data subset's stop date is INCLUSIVE of that whole day
# (frames run to stop 21:00), so stop = the last track point's day. The 44x72 degree box
# [lat0, lat0+44] x [lon0, lon0+72] contains the whole track with margin.
declare -A STORM=(
    [S2022]="2022-09-23 2022-09-28 3 106"  # tracks_extra.json (find_tc.py): 917 hPa typhoon, peak 09-26
    [T789]="2023-05-24 2023-05-31 0 111"  # last track point 06-01 00:00 (one frame) is skipped
    [T795]="2023-06-30 2023-07-09 0 97"
    [T829]="2023-11-03 2023-11-11 0 96"
)

RUNS=/data/pmd/runs
TRUTH=/data/pmd/local_data/2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
declare -A TRAIN_CFG=(
    [vortex]="$REPO_ROOT/configs/experiments/2026-10-10-video-pmd-two-block-vortex-shift/video_train.yaml"
    [two-block]="$REPO_ROOT/configs/experiments/2026-10-10-video-pmd-two-block-vortex-shift/video_train_control.yaml"
)
declare -A RUN_DIR=(
    [vortex]="$RUNS/video-pmd-two-block-vortex-shift"
    [two-block]="/data/pmd/checkpoints/two-block-flat"
)
MODELS=(vortex two-block)  # first = model under test

# Host kernel module is older than the installed userspace NVIDIA libs; use the matching on-disk ones.
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

# The data must be fully copied (an un-copied range reads as all-NaN) -- check every variable
# of both stores over the selected storms' windows (prepare_data.py fetches variable by variable).
WINDOWS=""
for s in $STORMS; do read -r start stop _ <<<"${STORM[$s]}"; WINDOWS="$WINDOWS $start,$stop"; done
CHECK_DATA='
import sys, numpy as np, xarray as xr
wins = [w.split(",") for w in sys.argv[3:]]
bad = []
for store in sys.argv[1:3]:
    ds = xr.open_zarr(store)
    for v in ["eastward_wind_at_ten_meters", "northward_wind_at_ten_meters", "PRMSL", "PRATEsfc", "air_temperature_at_two_meters"]:
        for lo, hi in wins:
            x = ds[v].sel(time=slice(lo, hi)).isel(latitude=slice(None, None, 40), longitude=slice(None, None, 40)).isel(time=slice(None, None, 4)).values
            if np.isnan(x).any():
                bad.append(store.rsplit("/", 1)[-1] + " " + v + " " + lo + ".." + hi)
if bad:
    sys.exit("Data not fully copied yet (NaN in): " + "; ".join(bad))
print("data check OK")
'
conda run --no-capture-output -n "$ENV_NAME" python -c "$CHECK_DATA" "$TRUTH" "${TRUTH/25km/100km}" $WINDOWS

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
for s in $STORMS; do
    read -r start stop lat0 lon0 <<<"${STORM[$s]}"
    days=$(( ( $(date -d "$stop" +%s) - $(date -d "$start" +%s) ) / 86400 + 1 ))
    for m in "${MODELS[@]}"; do
        mkdir -p "$OUT/$m"
        cfg="$OUT/$m/$s.yaml"
        conda run -n "$ENV_NAME" python "$SCRIPT_DIR/make_inference_config.py" \
            "${TRAIN_CFG[$m]}" "${RUN_DIR[$m]}/checkpoints/$CKPT_NAME" "$OUT/$m/$s.zarr" \
            "$OUT/$m/logs-$s" "$start" "$days" "$N_ENS" "$cfg" "$lat0" "$lon0" "$BATCH"
        echo "=== $m $s ($start..$stop, $days days, n_ens=$N_ENS) $(date +%T) ==="
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
            conda run --no-capture-output -n "$ENV_NAME" \
            python -m fme.downscaling.video_inference "$cfg" 2>&1 | tee -a "$OUT/$m/inference.log"
    done
done

conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT_DIR/tc_score.py" \
    --root "$OUT" --models "${MODELS[@]}" --truth "$TRUTH" --storms $STORMS --tracks-json "$SCRIPT_DIR/tracks_extra.json" 2>&1 | tee "$OUT/tc_scores.txt"
echo "Results in $OUT (tc_scores.txt, tc_scores_per_member_frame.csv)"
