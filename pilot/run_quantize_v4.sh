#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE="${PROFILE:-sa_sst2}"
SOURCE_VERSION="${SOURCE_VERSION:-v3_distilled}"
TARGET_VERSION="${TARGET_VERSION:-v4_quantized}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:-256}"
VERIFICATION_SIZE="${VERIFICATION_SIZE:-8}"
SEED="${SEED:-42}"
KEEP_EXISTING="${KEEP_EXISTING:-0}"
ARTIFACT_ROOT="$ROOT_DIR/pilot/artifacts"
LOG_DIR="$ARTIFACT_ROOT/$PROFILE"
LOG_FILE="$LOG_DIR/v4_quantization.log"
export PYTHONUNBUFFERED=1

mkdir -p "$ARTIFACT_ROOT" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_python() {
  uv run python -u "$@"
}

FORCE_FLAG=()
if [[ "$KEEP_EXISTING" != "1" ]]; then
  FORCE_FLAG+=(--force)
fi

log "Starting v4 INT8 quantization"
log "Requested profile: $PROFILE"
log "Source version: $SOURCE_VERSION"
log "Target version: $TARGET_VERSION"
log "Calibration size: $CALIBRATION_SIZE"
log "Verification size: $VERIFICATION_SIZE"

run_python pilot/quantize_v4.py \
  --profile "$PROFILE" \
  --source-version "$SOURCE_VERSION" \
  --target-version "$TARGET_VERSION" \
  --calibration-size "$CALIBRATION_SIZE" \
  --verification-size "$VERIFICATION_SIZE" \
  --seed "$SEED" \
  "${FORCE_FLAG[@]}"

log "v4 quantization completed"
log "Use loader spec: pilot/quantized_model_loader.py:load_model_bundle"
log "Example snapshot command:"
log "uv run python pilot/pipeline.py --stage snapshot --profile $PROFILE --model-version $TARGET_VERSION --model-loader pilot/quantized_model_loader.py:load_model_bundle"
