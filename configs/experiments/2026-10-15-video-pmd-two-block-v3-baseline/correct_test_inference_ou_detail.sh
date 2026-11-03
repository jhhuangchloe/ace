#!/bin/bash
# Remove the coarse-block units offset (+0.58 K T2m in the tropics, see
# scripts/video_pmd_eval/COARSE_BLOCK_UNITS.md) from the finished v3-ou test-set inference
# (test_inference_ou_detail.yaml, generated without coarse_block_in_fine_units), in place:
# writes and verifies a corrected copy, then swaps it in and keeps the original as
# <name>-uncorrected.zarr. Run only after the inference has finished; refuses otherwise.
#
#   bash configs/experiments/2026-10-15-video-pmd-two-block-v3-baseline/correct_test_inference_ou_detail.sh
#
# Knobs: ENV_NAME (fme), MIN_FREE_GB (250; the copy is ~200 GB until the backup is deleted).
set -euo pipefail

ENV_NAME="${ENV_NAME:-fme}"
MIN_FREE_GB="${MIN_FREE_GB:-250}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
CONFIG="$SCRIPT_DIR/test_inference_ou_detail.yaml"
ZARR=$(awk '/^output_path:/ {print $2}' "$CONFIG")
LOG="$(dirname "$ZARR")/inference.log"

if pgrep -f "video_inference.*$(basename "$CONFIG")" >/dev/null; then
    echo "Inference with $(basename "$CONFIG") is still running; wait for it to finish." >&2
    exit 1
fi
# Rank 0 logs each batch; "Completed inference" follows a barrier, so both ranks are done.
last_run=$(awk '/=== relaunch/ {buf=""} {buf = buf $0 "\n"} END {printf "%s", buf}' "$LOG")
if ! grep -q "Completed inference" <<<"$last_run"; then
    echo "$LOG has no 'Completed inference' after the last launch; the run is not finished." >&2
    grep -E "batch [0-9]+/[0-9]+ written" <<<"$last_run" | tail -1 >&2 || true
    exit 1
fi
if [ -d "${ZARR%.zarr}-uncorrected.zarr" ]; then
    echo "${ZARR%.zarr}-uncorrected.zarr exists: already corrected? Nothing done." >&2
    exit 1
fi
free_gb=$(df -BG --output=avail "$(dirname "$ZARR")" | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    echo "Only ${free_gb} GB free next to $ZARR; need ${MIN_FREE_GB} GB." >&2
    exit 1
fi

cd "$REPO_ROOT"
echo "Correcting $ZARR ($(date +%T))"
conda run --no-capture-output -n "$ENV_NAME" python \
    scripts/video_pmd_eval/correct_coarse_block_units.py "$ZARR" "$CONFIG" --replace
echo "Done ($(date +%T)). Score the corrected store; delete ${ZARR%.zarr}-uncorrected.zarr when happy."
