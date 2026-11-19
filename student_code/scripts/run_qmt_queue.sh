#!/usr/bin/env bash
# Usage: bash student_code/scripts/run_qmt_queue.sh <cuda-ordinal> <full|cross|cross-multiscale|off-multiscale|cross-neg-query> <seed> [seed ...]
set -euo pipefail

CUDA_ORDINAL=${1:?CUDA ordinal required}
VARIANT=${2:?variant required}
shift 2
[[ "$CUDA_ORDINAL" =~ ^[0-9]+$ ]] || { echo "CUDA ordinal must be numeric" >&2; exit 2; }
case "$VARIANT" in
  full) FLAGS=() ;;
  cross) FLAGS=(--no-qmt-multiscale --no-qmt-hard-negative) ;;
  cross-multiscale) FLAGS=(--no-qmt-hard-negative) ;;
  off-multiscale) FLAGS=(--no-qmt-multiscale) ;;
  cross-neg-query) FLAGS=(--no-qmt-multiscale --no-qmt-hard-negative --qmt-negative-query) ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac

cd /data14T/xz/repos/QVHighlights
PY=/data14T/xz/miniconda3/envs/QVHighlights/bin/python
DEVICE_INFO=$(CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES="$CUDA_ORDINAL" \
  "$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))') || {
  echo "CUDA ordinal ${CUDA_ORDINAL} cannot be initialized with FASTEST_FIRST" >&2
  exit 3
}
mkdir -p "outputs/QMT/${VARIANT}" outputs/logs/QMT
LOCK="outputs/QMT/${VARIANT}/.queue.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "Another QMT ${VARIANT} queue is already running" >&2; exit 4; }
for SEED in "$@"; do
  RUN="outputs/QMT/${VARIANT}/seed${SEED}"
  LOG="outputs/logs/QMT/${VARIANT}_seed${SEED}.log"
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
      --output-dir "$RUN" --seed "$SEED" "${FLAGS[@]}" >> "$LOG" 2>&1
done
