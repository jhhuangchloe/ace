#!/bin/bash
# Small-subset (S2022 typhoon, 2022-09-23..28, one storm-centred 44x72 coarse patch, 4 members) comparison of
# the final models:
#   v3-vortex  PMD + V3 baseline + vortex ("residual") correction + per-channel OU   (RUN: new inference here)
#   v3-flat    PMD + V3 baseline + flat noise, no vortex                              (RUN: new inference here)
#   v3-ou      PMD + V3 baseline + OU            (existing, units-fixed run  eval-tc-vortex-vs-two-block/v3-ou-fineunits)
#   two-block  PMD + linear interpolation        (existing, post-hoc units-corrected)
#   rstt       RSTT / STVSR video diffusion      (existing)
#   cascade    cascaded infill -> SR             (only if CASCADE_ZARR=<its S2022.zarr on the same patch/dates> is given)
# V3 pretrained baseline and one-shot linear interpolation (nearest / bicubic) are drawn from the data inside the plots.
#
#   GPU=7 bash configs/experiments/2026-10-19-video-pmd-vortex-vs-two-block-eval/run_tc_eval_final.sh
#   SKIP_INFERENCE=1 ...   only link / score / plot      CASCADE_ZARR=/path/S2022.zarr ...   add the cascade
# Knobs: GPU (7), N_ENS (4), BATCH (2), MIN_FREE_GB (25), OUT, ENV_NAME, CKPT_NAME (best.ckpt: same as the full-year runs),
# GIFS=1 also writes the storm GIFs (slower).  Inference is resumable: rerun after an interruption.
set -euo pipefail

GPU="${GPU:-7}"
ENV_NAME="${ENV_NAME:-fme}"
N_ENS="${N_ENS:-4}"
BATCH="${BATCH:-2}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"
CKPT_NAME="${CKPT_NAME:-best.ckpt}"
OUT="${OUT:-/data/pmd/runs/eval-tc-v3-final}"
STORM_NAME=S2022
START=2022-09-23; STOP=2022-09-28; LAT0=3; LON0=106   # same window/patch as run_tc_eval.sh
DAYS=$(( ( $(date -d "$STOP" +%s) - $(date -d "$START" +%s) ) / 86400 + 1 ))

RUNS=/data/pmd/runs
TRUTH=/data/pmd/local_data/2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
V3DIR="$REPO_ROOT/configs/experiments/2026-10-15-video-pmd-two-block-v3-baseline"
declare -A TRAIN_CFG=(
    [v3-vortex]="$V3DIR/video_train_ou_vortex_bs8_4gpus.yaml"
    [v3-flat]="$V3DIR/video_train_flat_bs8_4gpus.yaml"
)
declare -A RUN_DIR=(
    [v3-vortex]="$RUNS/video-pmd-two-block-v3-vortex-detail-ou"
    [v3-flat]="$RUNS/video-pmd-two-block-v3-baseline-detail"
)
NEW_MODELS=(v3-vortex v3-flat)

# Host kernel module older than the installed userspace NVIDIA libs: use the matching on-disk ones.
KMOD_VER=$(awk '/NVRM version/ {for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/) {print $i; exit}}' /proc/driver/nvidia/version 2>/dev/null || true)
LIBDIR=/usr/lib/x86_64-linux-gnu
if [ -n "$KMOD_VER" ] && [ -e "$LIBDIR/libnvidia-ml.so.$KMOD_VER" ] && [ "$(readlink -f "$LIBDIR/libnvidia-ml.so.1")" != "$(readlink -f "$LIBDIR/libnvidia-ml.so.$KMOD_VER")" ]; then
    SHIM="/tmp/nvshim-$KMOD_VER"
    mkdir -p "$SHIM"
    ln -sf "$LIBDIR/libnvidia-ml.so.$KMOD_VER" "$SHIM/libnvidia-ml.so.1"
    ln -sf "$LIBDIR/libcuda.so.$KMOD_VER" "$SHIM/libcuda.so.1"
    export LD_LIBRARY_PATH="$SHIM${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUT"
cd "$REPO_ROOT"

if [ "${SKIP_INFERENCE:-0}" != "1" ]; then
    for m in "${NEW_MODELS[@]}"; do
        [ -f "${RUN_DIR[$m]}/checkpoints/$CKPT_NAME" ] || { echo "Missing ${RUN_DIR[$m]}/checkpoints/$CKPT_NAME" >&2; exit 1; }
    done
    CHECK_PY='
import sys, warnings
warnings.filterwarnings("ignore")
import torch
free = torch.cuda.mem_get_info(int(sys.argv[1]))[0] / 2**30
print(f"GPU {sys.argv[1]}: {free:.1f} GB free")
sys.exit(0 if free >= float(sys.argv[2]) else "Not enough free GPU memory; refusing to start.")
'
    conda run --no-capture-output -n "$ENV_NAME" python -c "$CHECK_PY" "$GPU" "$MIN_FREE_GB"

    for m in "${NEW_MODELS[@]}"; do
        mkdir -p "$OUT/$m"
        cfg="$OUT/$m/$STORM_NAME.yaml"
        conda run -n "$ENV_NAME" python "$SCRIPT_DIR/make_inference_config.py" \
            "${TRAIN_CFG[$m]}" "${RUN_DIR[$m]}/checkpoints/$CKPT_NAME" "$OUT/$m/$STORM_NAME.zarr" \
            "$OUT/$m/logs-$STORM_NAME" "$START" "$DAYS" "$N_ENS" "$cfg" "$LAT0" "$LON0" "$BATCH"
        # exact fix for the coarse-block units offset (scripts/video_pmd_eval/COARSE_BLOCK_UNITS.md), as in the full-year configs
        conda run -n "$ENV_NAME" python -I -c "
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); c['model']['coarse_block_in_fine_units'] = True
yaml.safe_dump(c, open(sys.argv[1], 'w'), sort_keys=False)" "$cfg"
        echo "=== $m $STORM_NAME ($START..$STOP, $DAYS days, n_ens=$N_ENS, ckpt=$CKPT_NAME) $(date +%T) ==="
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
            conda run --no-capture-output -n "$ENV_NAME" \
            python -m fme.downscaling.video_inference "$cfg" 2>&1 | tee -a "$OUT/$m/inference.log"
    done
fi

# Link the models that already have S2022 output (same window, patch and ensemble size).
link() {  # $1 = model name, $2 = dir holding S2022.zarr
    if [ -d "$2/$STORM_NAME.zarr" ]; then
        mkdir -p "$OUT/$1"; ln -sfn "$2/$STORM_NAME.zarr" "$OUT/$1/$STORM_NAME.zarr"
        [ -f "$2/$STORM_NAME.yaml" ] && ln -sfn "$2/$STORM_NAME.yaml" "$OUT/$1/$STORM_NAME.yaml"  # the plots read the run's config
    else echo "WARNING: no $2/$STORM_NAME.zarr; $1 skipped" >&2; fi
}
link v3-ou "$RUNS/eval-tc-vortex-vs-two-block/v3-ou-fineunits"
link two-block "$RUNS/eval-tc-vortex-vs-two-block/corrected/two-block"
link rstt "$RUNS/eval-tc-vortex-vs-two-block/rstt"
MODELS=(v3-vortex v3-flat v3-ou two-block rstt)
if [ -n "${CASCADE_ZARR:-}" ]; then
    mkdir -p "$OUT/cascade"; ln -sfn "$CASCADE_ZARR" "$OUT/cascade/$STORM_NAME.zarr"
    [ -f "${CASCADE_ZARR%.zarr}.yaml" ] && ln -sfn "${CASCADE_ZARR%.zarr}.yaml" "$OUT/cascade/$STORM_NAME.yaml"
    MODELS+=(cascade)
else
    echo "NOTE: no CASCADE_ZARR given; the cascade is not in this comparison."
fi
MODELS=($(for m in "${MODELS[@]}"; do [ -d "$OUT/$m/$STORM_NAME.zarr" ] && echo "$m"; done))
echo "models: ${MODELS[*]}"

TRACKS="$SCRIPT_DIR/tracks_extra.json"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT_DIR/tc_score.py" \
    --root "$OUT" --models "${MODELS[@]}" --truth "$TRUTH" --storms "$STORM_NAME" --tracks-json "$TRACKS" 2>&1 | tee "$OUT/tc_scores.txt"
conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT_DIR/tc_mse_crps.py" \
    "$OUT" "$STORM_NAME" "$TRACKS" "$TRUTH" "${MODELS[@]}" 2>&1 | tee "$OUT/tc_mse_crps.txt"
conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT_DIR/plot_tc_minslp.py" \
    "$OUT" "$STORM_NAME" "$TRACKS" "$TRUTH" "$OUT/minslp_$STORM_NAME.png" "${MODELS[@]}" 2>&1 | tail -3
if [ "${GIFS:-0}" = "1" ]; then
    for f in prmsl wind; do
        MODELS="${MODELS[*]}" conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT_DIR/plot_tc_gif.py" \
            "$OUT" "$STORM_NAME" "$TRACKS" "$TRUTH" "$OUT/tc_${STORM_NAME}_$f.gif" "$f" 2>&1 | tail -2
    done
fi
echo "Results in $OUT: tc_scores.txt, tc_mse_crps.txt, minslp_$STORM_NAME.png"
