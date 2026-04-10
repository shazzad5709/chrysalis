#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

ARTIFACT_ROOT="$ROOT_DIR/pilot/artifacts"
MASTER_LOG_FILE="$ARTIFACT_ROOT/run_quantize_v4_all_profiles.log"

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

SOURCE_VERSION="${SOURCE_VERSION:-v3_distilled}"
TARGET_VERSION="${TARGET_VERSION:-v4_quantized}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:-256}"
VERIFICATION_SIZE="${VERIFICATION_SIZE:-8}"
SEED="${SEED:-42}"
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

log "Starting all-profile v4 quantization run"
log "Quantization config: source_version=$SOURCE_VERSION target_version=$TARGET_VERSION calibration_size=$CALIBRATION_SIZE verification_size=$VERIFICATION_SIZE seed=$SEED"
log "Master log: $MASTER_LOG_FILE"

mapfile -t profiles < <(resolve_profiles)

for profile in "${profiles[@]}"; do
  log "Running quantization for profile=$profile"
  PROFILE="$profile" \
  SOURCE_VERSION="$SOURCE_VERSION" \
  TARGET_VERSION="$TARGET_VERSION" \
  CALIBRATION_SIZE="$CALIBRATION_SIZE" \
  VERIFICATION_SIZE="$VERIFICATION_SIZE" \
  SEED="$SEED" \
  KEEP_EXISTING="$KEEP_EXISTING" \
    ./pilot/run_quantize_v4.sh
  log "Completed quantization for profile=$profile"
done

log "All requested v4 quantization profiles completed"
