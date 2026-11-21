#!/usr/bin/env bash
# Usage: bash student_code/scripts/run_a0_queue.sh <cuda-ordinal> [--epochs N] <seed> [seed ...]
set -euo pipefail

CUDA_ORDINAL=${1:?CUDA ordinal required}
shift
[[ "$CUDA_ORDINAL" =~ ^[0-9]+$ ]] || { echo "CUDA ordinal must be numeric" >&2; exit 2; }
EPOCHS=40
if [[ "${1:-}" == "--epochs" ]]; then
  EPOCHS=${2:?epoch count required after --epochs}
  shift 2
fi
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "Epoch count must be a positive integer" >&2; exit 2; }
(( $# > 0 )) || { echo "At least one seed is required" >&2; exit 2; }

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=${QVH_PYTHON:-$(conda run -n QVHighlights python -c 'import sys; print(sys.executable)' | tail -n 1)}
DEVICE_INFO=$(CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES="$CUDA_ORDINAL" \
  "$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))') || {
  echo "CUDA ordinal ${CUDA_ORDINAL} cannot be initialized with FASTEST_FIRST" >&2
  exit 3
}
OUTPUT_ROOT=outputs/A0
LOG_ROOT=outputs/logs/A0
if (( EPOCHS != 40 )); then
  OUTPUT_ROOT="outputs/A0_epoch${EPOCHS}"
  LOG_ROOT="outputs/logs/A0_epoch${EPOCHS}"
fi
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
LOCK="${OUTPUT_ROOT}/.queue.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "Another A0 ${EPOCHS}-epoch queue is already running" >&2; exit 4; }
for SEED in "$@"; do
  RUN="${OUTPUT_ROOT}/seed${SEED}"
  LOG="${LOG_ROOT}/seed${SEED}.log"
  if [[ -f "$RUN/model_best.pt" && -f "$RUN/run_summary.json" ]]; then
    echo "[$(date '+%F %T')] skip completed A0 epoch${EPOCHS} seed${SEED}" | tee -a "$LOG"
    continue
  fi
  echo "[$(date '+%F %T')] start A0 epoch${EPOCHS} seed${SEED} CUDA ordinal ${CUDA_ORDINAL} (${DEVICE_INFO})" | tee -a "$LOG"
  CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES="$CUDA_ORDINAL" \
    "$PY" scripts/train_locator.py \
      --feature-root data/features \
      --data-dir data/course \
      --output-dir "$RUN" --seed "$SEED" --epochs "$EPOCHS" \
      --batch-size 32 --learning-rate 0.0001 --eval-every 5 >> "$LOG" 2>&1
done
