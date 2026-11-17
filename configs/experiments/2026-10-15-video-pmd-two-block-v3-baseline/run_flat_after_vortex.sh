#!/bin/bash
# Chain two trainings in one tmux window:
#   1. watch the running V3 + vortex + OU training (the hybrid) until it has finished its max_epochs,
#   2. then start the V3 + flat-noise training (NO vortex correction) for its 60 epochs on the same GPUs.
#
#   tmux new -s chain
#   bash configs/experiments/2026-10-15-video-pmd-two-block-v3-baseline/run_flat_after_vortex.sh
#   DRY_RUN=1 bash .../run_flat_after_vortex.sh      # print what it sees and would do, then exit
#
# "Finished" means: no training process of the watched run is alive AND its latest.ckpt says
# startEpoch >= max_epochs. If the process is gone but the run is not finished (crash, or you are
# restarting it on other GPUs), it waits RESTART_GRACE seconds for the run to come back and gives up
# without starting the flat run if it does not.
#
# Knobs (env):
#   WATCH_DIR      experiment dir of the watched run   (default .../video-pmd-two-block-v3-vortex-detail-ou)
#   WATCH_PATTERN  regex that matches the watched run's command line (default: its vortex config name)
#   GPUS           GPUs for the flat run (default: the ones the watched run uses; 4 -> batch 8, 3 -> batch 6)
#   CONFIG         flat config override (default picked from the number of GPUs)
#   MIN_FREE_GB    GPU free-memory needed before launching (default 70; polls until every GPU has it)
#   POLL           seconds between checks (default 60);  RESTART_GRACE seconds (default 1800)
#   RESERVE_GB_PER_SAMPLE  passed to the launcher (default 40 = reserve 80 GB per GPU at 2 samples/GPU; 0 = off)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
LAUNCHER="$REPO_ROOT/configs/experiments/2026-10-10-video-pmd-two-block-vortex-shift/run_tmp_gpus2-5.sh"
WATCH_DIR="${WATCH_DIR:-/data/pmd/runs/video-pmd-two-block-v3-vortex-detail-ou}"
# [x] bracket trick: the regex does not match this script's own command line
WATCH_PATTERN="${WATCH_PATTERN:-fme.downscaling.video_trai[n].*video_train_ou_vorte[x]}"
MIN_FREE_GB="${MIN_FREE_GB:-70}"
POLL="${POLL:-60}"
RESTART_GRACE="${RESTART_GRACE:-1800}"
ENV_NAME="${ENV_NAME:-fme}"
LOG="${LOG:-/data/pmd/runs/chain_flat_after_vortex.log}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Host kernel module is older than the installed userspace NVIDIA libs: use the matching on-disk ones
# so nvidia-smi works (no CUDA context is created by this watcher).
KMOD_VER=$(awk '/NVRM version/ {for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/) {print $i; exit}}' /proc/driver/nvidia/version 2>/dev/null)
LIBDIR=/usr/lib/x86_64-linux-gnu
if [ -n "$KMOD_VER" ] && [ -e "$LIBDIR/libnvidia-ml.so.$KMOD_VER" ]; then
    SHIM="/tmp/nvshim-$KMOD_VER"
    mkdir -p "$SHIM"
    ln -sf "$LIBDIR/libnvidia-ml.so.$KMOD_VER" "$SHIM/libnvidia-ml.so.1"
    export LD_LIBRARY_PATH="$SHIM${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

watched_pid() { pgrep -f "$WATCH_PATTERN" | head -1; }

watched_gpus() {  # CUDA_VISIBLE_DEVICES of the watched run's processes
    local p; p=$(watched_pid)
    [ -n "$p" ] && tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'
}

max_epochs() { awk '/^max_epochs:/ {print $2; exit}' "$WATCH_DIR/config.yaml" 2>/dev/null; }

ckpt_start_epoch() {  # epoch counter stored in latest.ckpt (== max_epochs once training is complete)
    local ck="$WATCH_DIR/checkpoints/latest.ckpt"
    [ -f "$ck" ] || { echo -1; return; }
    conda run -n "$ENV_NAME" python -c "
import sys, torch
print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['startEpoch']))" "$ck" 2>/dev/null | grep -E '^[0-9]+$' | tail -1
}

finished() {
    local need have; need=$(max_epochs); have=$(ckpt_start_epoch)
    [ -n "$need" ] && [ -n "$have" ] && [ "$have" -ge "$need" ] 2>/dev/null
}

pick_config() {  # $1 = number of GPUs
    case "$1" in
        4) echo "$HERE/video_train_flat_bs8_4gpus.yaml" ;;
        3) echo "$HERE/video_train_flat_gpus017.yaml" ;;
        *) echo "" ;;
    esac
}

free_ok() {  # every GPU in $1 (comma list) has >= MIN_FREE_GB free
    local g used total
    for g in ${1//,/ }; do
        read -r used total < <(nvidia-smi -i "$g" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ',')
        [ -n "${used:-}" ] || return 1
        [ $(( (total - used) / 1024 )) -ge "$MIN_FREE_GB" ] || return 1
    done
}

log "watching $WATCH_DIR (pattern: $WATCH_PATTERN, max_epochs=$(max_epochs))"
LAST_GPUS=""
gone_since=""
while true; do
    pid=$(watched_pid)
    if [ -n "$pid" ]; then
        gone_since=""
        g=$(watched_gpus); [ -n "$g" ] && LAST_GPUS="$g"
        if [ "${DRY_RUN:-0}" = "1" ]; then
            log "DRY RUN: watched run is alive (pid $pid) on GPUs ${LAST_GPUS:-?}; latest.ckpt startEpoch=$(ckpt_start_epoch)/$(max_epochs)"
            break
        fi
        sleep "$POLL"
        continue
    fi
    # no process of the watched run
    if finished; then
        log "watched run is finished (startEpoch >= max_epochs)."
        break
    fi
    [ "${DRY_RUN:-0}" = "1" ] && { log "DRY RUN: watched run is not running and not finished"; break; }
    [ -n "$gone_since" ] || gone_since=$(date +%s)
    if [ $(( $(date +%s) - gone_since )) -ge "$RESTART_GRACE" ]; then
        log "watched run is not running and NOT finished (startEpoch=$(ckpt_start_epoch), max_epochs=$(max_epochs)) for ${RESTART_GRACE}s: not starting the flat run."
        exit 1
    fi
    log "watched run not running, not finished; waiting for it to come back (up to ${RESTART_GRACE}s)..."
    sleep 30
done

# ---- choose GPUs / config for the flat run ----
GPUS="${GPUS:-$LAST_GPUS}"
if [ -z "$GPUS" ]; then
    echo "Could not determine GPUs (the watched run was never seen). Set GPUS=0,1,5,7" >&2
    [ "${DRY_RUN:-0}" = "1" ] || exit 1
fi
N_GPUS=$(awk -F, '{print NF}' <<<"${GPUS:-x}")
CONFIG="${CONFIG:-$(pick_config "$N_GPUS")}"
if [ -z "$CONFIG" ] || [ ! -f "$CONFIG" ]; then
    echo "No flat config for $N_GPUS GPUs (have 3 and 4). Set CONFIG=<path>." >&2
    exit 1
fi
# the flat run must not contain any vortex correction, and must be the V3 + flat-noise setup
if grep -qE '^[[:space:]]+vortex_shift_baseline:[[:space:]]*true' "$CONFIG"; then
    echo "$CONFIG enables vortex_shift_baseline -- refusing (the flat run must have no vortex correction)." >&2
    exit 1
fi
grep -qE '^[[:space:]]+r_kernel:[[:space:]]*brownian_bridge' "$CONFIG" || { echo "$CONFIG is not the flat (brownian_bridge) kernel." >&2; exit 1; }
FLAT_DIR=$(awk '/^experiment_dir:/ {print $2}' "$CONFIG")
log "flat run: GPUS=$GPUS config=$(basename "$CONFIG") max_epochs=$(awk '/^max_epochs:/ {print $2}' "$CONFIG") experiment_dir=$FLAT_DIR $( [ -f "$FLAT_DIR/checkpoints/latest.ckpt" ] && echo '(resumes from its latest.ckpt)' || echo '(fresh start)')"
if [ "${DRY_RUN:-0}" = "1" ]; then
    log "DRY RUN: would wait for >= ${MIN_FREE_GB} GB free on GPUs $GPUS, then run: GPUS=$GPUS MIN_FREE_GB=$MIN_FREE_GB CONFIG=$CONFIG bash $LAUNCHER"
    exit 0
fi

# ---- wait until the freed GPUs have room (neighbours may have grabbed some) ----
until free_ok "$GPUS"; do
    log "waiting for >= ${MIN_FREE_GB} GB free on every GPU in $GPUS ..."
    sleep "$POLL"
done

log "starting the flat run."
cd "$REPO_ROOT"
GPUS="$GPUS" MIN_FREE_GB="$MIN_FREE_GB" CONFIG="$CONFIG" bash "$LAUNCHER"
status=$?
log "flat run exited with status $status."
exit $status
