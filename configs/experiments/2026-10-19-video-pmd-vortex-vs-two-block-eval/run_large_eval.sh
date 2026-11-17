#!/bin/bash
# Larger small-subset comparison, sized to finish in ~6 hours on ONE GPU (default GPUS=7).
# Regional windows (44x72-coarse-cell = 176x288 25 km patches, DAYS=3 -> 32 frames each) over the 2023 test year: tropical
# cyclones plus non-TC regions/seasons, processed ONE WINDOW AT A TIME in the order of WINDOW_TABLE, so whatever has finished
# is always scoreable. A time-budget guard (TIME_BUDGET_MIN, default 330 min) stops starting new windows when the next one
# would not fit, then scores what is done.
#
# Models per window (4 ensemble members, same checkpoints/settings as run_tc_eval_final.sh; best.ckpt everywhere):
#   v3-vortex  v3-flat  v3-ou  two-block  rstt   diffusion inference            (GPU)
#   cascade                                      stage 1 infill + stage 2 SR    (GPU)
#   v3-raw  linear-bicubic  linear-nearest       deterministic baselines        (CPU, in the background)
# then score_windows.py: patch-wide MSE / fair CRPS per channel + day-block bootstrap intervals.
#
#   GPUS=7 bash configs/experiments/2026-10-19-video-pmd-vortex-vs-two-block-eval/run_large_eval.sh
#   TIME_BUDGET_MIN=240 ...   shorter budget    DAYS=2 ...   shorter windows    PHASES="score" ...   only re-score
#   MODELS="v3-flat v3-ou" ...   subset of the diffusion models    WINDOWS="TC789 IndiaMon" ...   subset / order of windows
# Knobs: GPUS (space or comma separated; default "7"; several GPUs split the models of each window), N_ENS (4),
# CKPT_NAME (best.ckpt), MIN_FREE_GB (25), OUT (/data/pmd/runs/eval-large), ENV_NAME (fme), DAYS (3),
# PHASES (default "diffusion cascade baselines score"), TIME_BUDGET_MIN (330).
#
# Measured on S2022 (7 days = 6 clips, 4 members, one GPU): v3-vortex 22 min, v3-flat 21, v3-ou 19, two-block 15, rstt 0.5,
# cascade ~23 (stage 1 ~8 + SR ~15)  =>  ~17 GPU-min per clip over all models + ~6 min fixed (model loads) per window,
# i.e. ~57 min per 3-clip window: about 5 windows in the default budget (about 3 x the frames of the single S2022 window).
# Each finished (window, model) leaves a .done marker: rerunning after an interruption or a failed job skips finished work.
# A failed job is reported at the end (failed_jobs.txt) and does not stop the others.
set -uo pipefail

GPUS="${GPUS:-7}"; GPUS="${GPUS//,/ }"
ENV_NAME="${ENV_NAME:-fme}"
N_ENS="${N_ENS:-4}"
CKPT_NAME="${CKPT_NAME:-best.ckpt}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"
OUT="${OUT:-/data/pmd/runs/eval-large}"
PHASES="${PHASES:-diffusion cascade baselines score}"
MODELS="${MODELS:-v3-vortex v3-flat v3-ou two-block rstt}"
DAYS="${DAYS:-3}"            # clips per window; make_inference_config: (DAYS+1)*8 frames (S2022 used 6 -> 56)
TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-330}"

# name  start  lat0  lon0     (44 x 72 degree box [lat0, lat0+44] x [lon0, lon0+72]; track windows: run_tc_eval.sh)
WINDOW_TABLE="
TC789       2023-05-24    0 111
IndiaMon    2023-07-20    0  60
NA-winter   2023-01-10   30 280
EqPacific   2023-04-12  -22 180
TC829       2023-11-03    0  96
MaritimeC   2023-02-15  -30 100
AtlMDR      2023-08-15    0 288
TC795       2023-06-30    0  97
SouthOcean  2023-10-05  -60   0
NPac-winter 2023-12-10   20 140
"
WINDOWS="${WINDOWS:-$(awk 'NF {print $1}' <<<"$WINDOW_TABLE" | tr '\n' ' ')}"

RUNS=/data/pmd/runs
TRUTH=/data/pmd/local_data/2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
V3DIR="$REPO_ROOT/configs/experiments/2026-10-15-video-pmd-two-block-v3-baseline"
declare -A TRAIN_CFG=(
    [v3-vortex]="$V3DIR/video_train_ou_vortex_bs8_4gpus.yaml"
    [v3-flat]="$V3DIR/video_train_flat_bs8_4gpus.yaml"
    [v3-ou]="$RUNS/video-pmd-two-block-v3-baseline-ou-detail/config.yaml"
    [two-block]="$REPO_ROOT/configs/experiments/2026-10-10-video-pmd-two-block-vortex-shift/video_train_control.yaml"
    [rstt]="$RUNS/stvsr-rstt-finetune/config.yaml"
)
declare -A RUN_DIR=(
    [v3-vortex]="$RUNS/video-pmd-two-block-v3-vortex-detail-ou"
    [v3-flat]="$RUNS/video-pmd-two-block-v3-baseline-detail"
    [v3-ou]="$RUNS/video-pmd-two-block-v3-baseline-ou-detail"
    [two-block]="/data/pmd/checkpoints/two-block-flat"
    [rstt]="$RUNS/stvsr-rstt-finetune"
)
# two-block-family models get the exact coarse-block units fix (scripts/video_pmd_eval/COARSE_BLOCK_UNITS.md)
UNITS_FIX=" v3-vortex v3-flat v3-ou two-block "

win_field() {  # $1 window, $2 field index (2 start, 3 lat0, 4 lon0)
    awk -v w="$1" -v i="$2" '$1 == w {print $i}' <<<"$WINDOW_TABLE"
}

KMOD_VER=$(awk '/NVRM version/ {for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/) {print $i; exit}}' /proc/driver/nvidia/version 2>/dev/null || true)
LIBDIR=/usr/lib/x86_64-linux-gnu
if [ -n "$KMOD_VER" ] && [ -e "$LIBDIR/libnvidia-ml.so.$KMOD_VER" ] && [ "$(readlink -f "$LIBDIR/libnvidia-ml.so.1")" != "$(readlink -f "$LIBDIR/libnvidia-ml.so.$KMOD_VER")" ]; then
    SHIM="/tmp/nvshim-$KMOD_VER"; mkdir -p "$SHIM"
    ln -sf "$LIBDIR/libnvidia-ml.so.$KMOD_VER" "$SHIM/libnvidia-ml.so.1"
    ln -sf "$LIBDIR/libcuda.so.$KMOD_VER" "$SHIM/libcuda.so.1"
    export LD_LIBRARY_PATH="$SHIM${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

CHECK_FREE='
import sys, warnings
warnings.filterwarnings("ignore")
import torch
free = torch.cuda.mem_get_info(int(sys.argv[1]))[0] / 2**30
sys.exit(0 if free >= float(sys.argv[2]) else f"GPU {sys.argv[1]}: only {free:.1f} GB free")
'
gpu_ok() { conda run -n "$ENV_NAME" python -c "$CHECK_FREE" "$1" "$MIN_FREE_GB" >/dev/null 2>&1; }

mkdir -p "$OUT"
cd "$REPO_ROOT"
FAILED_LOG="$OUT/failed_jobs.txt"; : > "$FAILED_LOG"

# ---- data check: every window's stores must be fully copied (an un-copied range reads as NaN) ----
CHECK_DATA='
import sys, numpy as np, xarray as xr
bad = []
for store in sys.argv[1:3]:
    ds = xr.open_zarr(store)
    for w in sys.argv[3:]:
        lo, hi = w.split(",")
        for v in ["eastward_wind_at_ten_meters", "northward_wind_at_ten_meters", "PRMSL", "PRATEsfc", "air_temperature_at_two_meters"]:
            x = ds[v].sel(time=slice(lo, hi)).isel(latitude=slice(None, None, 40), longitude=slice(None, None, 40)).isel(time=slice(None, None, 4)).values
            if x.size == 0 or np.isnan(x).any():
                bad.append(store.rsplit("/", 1)[-1] + " " + v + " " + w)
if bad:
    sys.exit("Data missing / not fully copied: " + "; ".join(bad))
print("data check OK")
'
RANGES=""
for w in $WINDOWS; do
    s=$(win_field "$w" 2); [ -n "$s" ] || { echo "unknown window $w" >&2; exit 1; }
    RANGES="$RANGES $s,$(date -d "$s + $DAYS days" +%F)"
done
conda run --no-capture-output -n "$ENV_NAME" python -c "$CHECK_DATA" "$TRUTH" "${TRUTH/25km/100km}" $RANGES || exit 1

# ---- one (window, model) diffusion inference on one GPU ----
run_diffusion_job() {  # gpu window model
    local gpu=$1 w=$2 m=$3 d="$OUT/$3" start lat0 lon0 cfg
    [ -f "$d/$w.done" ] && { echo "[gpu $gpu] $m $w already done"; return 0; }
    start=$(win_field "$w" 2); lat0=$(win_field "$w" 3); lon0=$(win_field "$w" 4)
    mkdir -p "$d"; cfg="$d/$w.yaml"
    [ -f "${RUN_DIR[$m]}/checkpoints/$CKPT_NAME" ] || { echo "$m $w: missing ${RUN_DIR[$m]}/checkpoints/$CKPT_NAME" | tee -a "$FAILED_LOG"; return 1; }
    conda run -n "$ENV_NAME" python "$SCRIPT_DIR/make_inference_config.py" \
        "${TRAIN_CFG[$m]}" "${RUN_DIR[$m]}/checkpoints/$CKPT_NAME" "$d/$w.zarr" "$d/logs-$w" \
        "$start" "$DAYS" "$N_ENS" "$cfg" "$lat0" "$lon0" 2 >/dev/null || { echo "$m $w: config generation failed" | tee -a "$FAILED_LOG"; return 1; }
    if [[ "$UNITS_FIX" == *" $m "* ]]; then
        conda run -n "$ENV_NAME" python -I -c "
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); c['model']['coarse_block_in_fine_units'] = True
yaml.safe_dump(c, open(sys.argv[1], 'w'), sort_keys=False)" "$cfg"
    fi
    until gpu_ok "$gpu"; do echo "[gpu $gpu] waiting for >= ${MIN_FREE_GB} GB free..."; sleep 120; done
    echo "=== [gpu $gpu] $m $w ($start, lat0=$lat0 lon0=$lon0) $(date +%T) ==="
    if CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$ENV_NAME" \
            python -m fme.downscaling.video_inference "$cfg" >> "$d/inference-$w.log" 2>&1 \
        && grep -q "Completed inference" "$d/inference-$w.log"; then
        touch "$d/$w.done"
    else
        echo "$m $w: inference failed (see $d/inference-$w.log)" | tee -a "$FAILED_LOG"; return 1
    fi
}

run_cascade_job() {  # gpu window
    local gpu=$1 w=$2 start lat0 lon0 work="$OUT/cascade-work/$2" stop ref
    [ -f "$OUT/cascade/$w.done" ] && { echo "[gpu $gpu] cascade $w already done"; return 0; }
    start=$(win_field "$w" 2); lat0=$(win_field "$w" 3); lon0=$(win_field "$w" 4)
    stop=$(date -d "$start + $DAYS days" +%F)
    mkdir -p "$work" "$OUT/cascade"
    ref="$OUT/v3-flat/$w.zarr"; [ -d "$ref" ] || ref="$OUT/rstt/$w.zarr"
    [ -d "$ref" ] || { echo "cascade $w: no reference zarr (run the diffusion phase first)" | tee -a "$FAILED_LOG"; return 1; }
    conda run -n "$ENV_NAME" python "$SCRIPT_DIR/make_cascade_configs.py" "$work" "$start" "$stop" "$N_ENS" \
        "$lat0" "$(( lat0 + 44 ))" "$lon0" "$(( lon0 + 72 ))" >/dev/null || { echo "cascade $w: config generation failed" | tee -a "$FAILED_LOG"; return 1; }
    # the SR output is named after the output config entry ("S2022" in make_cascade_configs.py)
    until gpu_ok "$gpu"; do echo "[gpu $gpu] waiting for >= ${MIN_FREE_GB} GB free..."; sleep 120; done
    echo "=== [gpu $gpu] cascade $w $(date +%T) ==="
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$ENV_NAME" \
        python -m fme.downscaling.video_inference "$work/stage1.yaml" >> "$work/stage1.log" 2>&1 \
      && CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$ENV_NAME" \
        python -m fme.downscaling.inference "$work/stage2.yaml" >> "$work/stage2.log" 2>&1 \
      && conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT_DIR/cascade_to_zarr.py" \
        "$work/stage2/S2022.zarr" "$ref" "$OUT/cascade/$w.zarr" >> "$work/convert.log" 2>&1 \
      && touch "$OUT/cascade/$w.done" \
      || { echo "cascade $w: failed (see $work/stage*.log, convert.log)" | tee -a "$FAILED_LOG"; return 1; }
}

# round-robin over the GPUs: worker i takes jobs i, i+N, ...
run_phase() {  # $1 = function, remaining = job argument strings ("window model" / "window")
    local fn=$1; shift
    local -a gpus=($GPUS) jobs=("$@")
    local i g
    for ((g = 0; g < ${#gpus[@]}; g++)); do
        (
            for ((i = g; i < ${#jobs[@]}; i += ${#gpus[@]})); do
                $fn "${gpus[$g]}" ${jobs[$i]} || true
            done
        ) &
    done
    wait
}

has_phase() { [[ " $PHASES " == *" $1 "* ]]; }

run_baselines_job() {  # window (CPU only)
    local w=$1 ref="$OUT/v3-flat/$1.yaml"
    [ -f "$ref" ] || ref="$OUT/rstt/$w.yaml"
    [ -f "$ref" ] || { echo "baselines $w: no reference yaml" | tee -a "$FAILED_LOG"; return 1; }
    [ -d "$OUT/v3-raw/$w.zarr" ] && [ -d "$OUT/linear-bicubic/$w.zarr" ] && [ -d "$OUT/linear-nearest/$w.zarr" ] && return 0
    echo "=== baselines $w $(date +%T) ==="
    PYTHONPATH="$SCRIPT_DIR" conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT_DIR/baseline_zarr.py" \
        "$OUT" "$w" "$ref" "$TRUTH" linear-bicubic linear-nearest v3-raw >> "$OUT/baselines-$w.log" 2>&1 \
        || echo "baselines $w: failed (see $OUT/baselines-$w.log)" | tee -a "$FAILED_LOG"
}

if has_phase cascade; then
    # one shared copy of the HiRO checkpoint with use_apex_gn off (apex is not installed); made once, before the jobs
    export HIRO_NOAPEX_CKPT="$OUT/cascade-work/hiro-v6-noapex.ckpt"
    mkdir -p "$OUT/cascade-work/_prep"
    first=${WINDOWS%% *}
    conda run -n "$ENV_NAME" python "$SCRIPT_DIR/make_cascade_configs.py" "$OUT/cascade-work/_prep" \
        "$(win_field "$first" 2)" "$(date -d "$(win_field "$first" 2) + $DAYS days" +%F)" "$N_ENS" 0 44 0 72 >/dev/null
fi

T0=$(date +%s); BUDGET=$(( TIME_BUDGET_MIN * 60 )); n_done=0
for w in $WINDOWS; do
    if (( n_done > 0 )); then
        elapsed=$(( $(date +%s) - T0 )); avg=$(( elapsed / n_done ))
        if (( elapsed + avg * 115 / 100 > BUDGET )); then
            echo "time budget: $(( elapsed / 60 )) min used, ~$(( avg / 60 )) min per window, budget ${TIME_BUDGET_MIN} min: not starting $w or later windows."
            break
        fi
    fi
    echo "##### window $w ($(( n_done + 1 ))): $(( ($(date +%s) - T0) / 60 )) min elapsed #####"
    if has_phase diffusion; then
        jobs=(); for m in $MODELS; do jobs+=("$w $m"); done
        run_phase run_diffusion_job "${jobs[@]}"
    fi
    if has_phase baselines; then run_baselines_job "$w" & fi        # CPU, overlaps with the cascade on the GPU
    if has_phase cascade; then run_phase run_cascade_job "$w"; fi
    n_done=$(( n_done + 1 ))
done
wait
echo "windows processed: $n_done; total $(( ($(date +%s) - T0) / 60 )) min"

if has_phase score; then
    ens=(); for m in $MODELS cascade; do [ -d "$OUT/$m" ] && ens+=("$m"); done
    echo "scoring: ens=${ens[*]}"
    conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT_DIR/score_windows.py" "$OUT" "$TRUTH" \
        --ens "${ens[@]}" --det v3-raw linear-bicubic linear-nearest --ref v3-ou
fi

if [ -s "$FAILED_LOG" ]; then echo; echo "FAILED JOBS (rerun the script to retry them):"; cat "$FAILED_LOG"; else echo "no failed jobs"; fi
echo "Results: $OUT/large_eval_tables.txt (+ large_eval_per_frame.csv)"
