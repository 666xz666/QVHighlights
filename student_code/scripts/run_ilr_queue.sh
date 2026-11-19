#!/usr/bin/env bash
# Usage: bash student_code/scripts/run_ilr_queue.sh <cuda-ordinal> <full|off> <seed> [seed ...]
set -euo pipefail

CUDA_ORDINAL=${1:?CUDA ordinal required}
VARIANT=${2:?variant must be full or off}
shift 2
[[ "$CUDA_ORDINAL" =~ ^[0-9]+$ ]] || { echo "CUDA ordinal must be numeric" >&2; exit 2; }
case "$VARIANT" in
  full) FLAGS=() ;;
  off) FLAGS=(--disable-ilr) ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac

cd /data14T/xz/repos/QVHighlights
PY=/data14T/xz/miniconda3/envs/QVHighlights/bin/python
CONFIG=student_code/configs/vlm_qlora_ilr.json
DEVICE_INFO=$(CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES="$CUDA_ORDINAL" \
  "$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))') || {
  echo "CUDA ordinal ${CUDA_ORDINAL} cannot be initialized with FASTEST_FIRST" >&2
  exit 3
}
mkdir -p "outputs/ILR/${VARIANT}" outputs/logs/ILR
LOCK="outputs/ILR/${VARIANT}/.queue.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "Another ILR ${VARIANT} queue is already running" >&2; exit 4; }
for SEED in "$@"; do
  RUN="outputs/ILR/${VARIANT}/seed${SEED}"
  LOG="outputs/logs/ILR/${VARIANT}_seed${SEED}.log"
  echo "[$(date '+%F %T')] start ${VARIANT} seed${SEED} CUDA ordinal ${CUDA_ORDINAL} (${DEVICE_INFO})" | tee -a "$LOG"
  if [[ ! -f "$RUN/checkpoint-936/adapter_config.json" ]]; then
    RESUME=()
    for SAVED_STEP in 624 312; do
      if [[ -f "$RUN/checkpoint-${SAVED_STEP}/trainer_state.pt" ]]; then
        RESUME=(--resume "$RUN/checkpoint-${SAVED_STEP}")
        break
      fi
    done
    CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES="$CUDA_ORDINAL" \
      "$PY" student_code/ilr/train.py \
        --config "$CONFIG" --train-data data/reference_full/sft_pairwise.jsonl \
        --cache-only --cache-dir data/reference_frame_cache \
        --output-dir "$RUN" --seed "$SEED" "${FLAGS[@]}" "${RESUME[@]}" >> "$LOG" 2>&1
  fi
  for STEP in 312 624 936; do
    PRED="$RUN/dev_step${STEP}.jsonl"
    SCORE="$RUN/dev_step${STEP}_score.json"
    [[ -f "$SCORE" ]] && continue
    CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES="$CUDA_ORDINAL" \
      "$PY" -m vlm.rerank \
        --annotations data/course/dev_queries.jsonl \
        --candidates data/reference_candidates/dev.jsonl \
        --config "$CONFIG" --adapter "$RUN/checkpoint-${STEP}" \
        --cache-only --cache-dir data/reference_frame_cache \
        --output "$PRED" >> "$LOG" 2>&1
    "$PY" scripts/evaluate.py \
      --ground-truth data/course/dev.jsonl --predictions "$PRED" \
      --reference-candidates data/reference_candidates/dev.jsonl \
      --output "$SCORE" >> "$LOG" 2>&1
  done
done
