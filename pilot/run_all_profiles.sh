#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

ARTIFACT_ROOT="$ROOT_DIR/pilot/artifacts"
MASTER_LOG_FILE="$ARTIFACT_ROOT/run_all_profiles.log"

ALL_PROFILES=(
  "sa_sst2"
  "sa_imdb"
  "nli_snli"
  "nli_multinli"
  "gen_sst2"
  "gen_imdb"
  "gen_snli"
  "gen_multinli"
  "gen_agnews"
)

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRAD_ACC="${GRAD_ACC:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
KEEP_EXISTING="${KEEP_EXISTING:-0}"
PROFILE_LIST="${PROFILE_LIST:-}"

mkdir -p "$ARTIFACT_ROOT"
exec > >(tee -a "$MASTER_LOG_FILE") 2>&1

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

resolve_profiles() {
  if [[ -n "$PROFILE_LIST" ]]; then
    local old_ifs="$IFS"
    IFS=','
    read -r -a parsed <<< "$PROFILE_LIST"
    IFS="$old_ifs"
    printf '%s\n' "${parsed[@]}"
    return
  fi
  printf '%s\n' "${ALL_PROFILES[@]}"
}

log "Starting all-profile Chrysalis run"
log "Training config: device=$DEVICE batch_size=$BATCH_SIZE eval_batch_size=$EVAL_BATCH_SIZE grad_acc=$GRAD_ACC num_workers=$NUM_WORKERS"
log "Master log: $MASTER_LOG_FILE"

mapfile -t profiles < <(resolve_profiles)

for profile in "${profiles[@]}"; do
  log "Running profile=$profile"
  PROFILE="$profile" \
  DEVICE="$DEVICE" \
  BATCH_SIZE="$BATCH_SIZE" \
  EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
  GRAD_ACC="$GRAD_ACC" \
  NUM_WORKERS="$NUM_WORKERS" \
  KEEP_EXISTING="$KEEP_EXISTING" \
    ./pilot/run_fullscale.sh
  log "Completed profile=$profile"
done

log "All requested profiles completed"
