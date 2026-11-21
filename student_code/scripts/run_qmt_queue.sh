#!/usr/bin/env bash
# Usage: bash student_code/scripts/run_qmt_queue.sh <cuda-ordinal> <variant> [--epochs N] <seed> [seed ...]
set -euo pipefail

CUDA_ORDINAL=${1:?CUDA ordinal required}
VARIANT=${2:?variant required}
shift 2
[[ "$CUDA_ORDINAL" =~ ^[0-9]+$ ]] || { echo "CUDA ordinal must be numeric" >&2; exit 2; }
EPOCHS=40
if [[ "${1:-}" == "--epochs" ]]; then
  EPOCHS=${2:?epoch count required after --epochs}
  shift 2
fi
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "Epoch count must be a positive integer" >&2; exit 2; }
(( $# > 0 )) || { echo "At least one seed is required" >&2; exit 2; }
case "$VARIANT" in
  full) FLAGS=() ;;
  cross) FLAGS=(--no-qmt-multiscale --no-qmt-hard-negative) ;;
  gate-only) FLAGS=(--no-qmt-cross-attn --no-qmt-multiscale --no-qmt-hard-negative --qmt-feature-gate) ;;
  cross-gate) FLAGS=(--no-qmt-multiscale --no-qmt-hard-negative --qmt-feature-gate) ;;
  cross-multiscale) FLAGS=(--no-qmt-hard-negative) ;;
  off-multiscale) FLAGS=(--no-qmt-multiscale) ;;
  cross-neg-query) FLAGS=(--no-qmt-multiscale --no-qmt-hard-negative --qmt-negative-query) ;;
  cross-trcc) FLAGS=(--no-qmt-multiscale --no-qmt-hard-negative --qmt-trcc) ;;
  cross-bridge) FLAGS=(--no-qmt-multiscale --no-qmt-hard-negative --qmt-bridge) ;;
  cross-iou-rank) FLAGS=(--no-qmt-multiscale --no-qmt-hard-negative --qmt-iou-rank) ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=${QVH_PYTHON:-$(conda run -n QVHighlights python -c 'import sys; print(sys.executable)' | tail -n 1)}
DEVICE_INFO=$(CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES="$CUDA_ORDINAL" \
  "$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))') || {
  echo "CUDA ordinal ${CUDA_ORDINAL} cannot be initialized with FASTEST_FIRST" >&2
  exit 3
}
OUTPUT_ROOT=outputs/QMT
LOG_ROOT=outputs/logs/QMT
if (( EPOCHS != 40 )); then
  OUTPUT_ROOT="outputs/QMT_epoch${EPOCHS}"
  LOG_ROOT="outputs/logs/QMT_epoch${EPOCHS}"
fi
mkdir -p "${OUTPUT_ROOT}/${VARIANT}" "$LOG_ROOT"
LOCK="${OUTPUT_ROOT}/${VARIANT}/.queue.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "Another QMT ${VARIANT} queue is already running" >&2; exit 4; }
for SEED in "$@"; do
  RUN="${OUTPUT_ROOT}/${VARIANT}/seed${SEED}"
  LOG="${LOG_ROOT}/${VARIANT}_seed${SEED}.log"
  if [[ -f "$RUN/model_best.pt" && -f "$RUN/summary.json" ]]; then
    echo "[$(date '+%F %T')] skip completed ${VARIANT} seed${SEED}" | tee -a "$LOG"
    continue
  fi
  echo "[$(date '+%F %T')] start ${VARIANT} seed${SEED} CUDA ordinal ${CUDA_ORDINAL} (${DEVICE_INFO})" | tee -a "$LOG"
  CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES="$CUDA_ORDINAL" \
    "$PY" student_code/qmt/train.py \
      --feature-root data/features \
      --train-annotations data/course/train.jsonl \
      --dev-annotations data/course/dev.jsonl \
      --output-dir "$RUN" --seed "$SEED" --epochs "$EPOCHS" \
      "${FLAGS[@]}" >> "$LOG" 2>&1
done
