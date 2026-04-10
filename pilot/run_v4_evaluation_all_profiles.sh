#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

ARTIFACT_ROOT="$ROOT_DIR/pilot/artifacts"
MASTER_LOG_FILE="$ARTIFACT_ROOT/run_v4_evaluation_all_profiles.log"

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

OLD_VERSION="${OLD_VERSION:-v3_distilled}"
NEW_VERSION="${NEW_VERSION:-v4_quantized}"
MODEL_LOADER="${MODEL_LOADER:-pilot/quantized_model_loader.py:load_model_bundle}"
KEEP_EXISTING="${KEEP_EXISTING:-0}"
PROFILE_LIST="${PROFILE_LIST:-}"

mkdir -p "$ARTIFACT_ROOT"
exec > >(tee -a "$MASTER_LOG_FILE") 2>&1

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_python() {
  uv run python -u "$@"
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

log "Starting all-profile v4 evaluation run"
log "Evaluation config: old_version=$OLD_VERSION new_version=$NEW_VERSION model_loader=$MODEL_LOADER"
log "Master log: $MASTER_LOG_FILE"

mapfile -t profiles < <(resolve_profiles)

for profile in "${profiles[@]}"; do
  PROFILE_ARTIFACT_ROOT="$ARTIFACT_ROOT/$profile"
  CORPUS_DIR="$PROFILE_ARTIFACT_ROOT/corpus"
  SNAPSHOT_DIR="$PROFILE_ARTIFACT_ROOT/snapshots"
  REPORT_DIR="$PROFILE_ARTIFACT_ROOT/regression_reports"

  log "Running v4 evaluation for profile=$profile"

  if [[ ! -d "$CORPUS_DIR" ]]; then
    log "Skipping profile=$profile because corpus dir is missing: $CORPUS_DIR"
    continue
  fi

  if [[ "$KEEP_EXISTING" != "1" ]]; then
    log "Removing stale v4 snapshot/report artifacts for profile=$profile"
    rm -rf "$SNAPSHOT_DIR/$NEW_VERSION"
    rm -f "$REPORT_DIR"/regression_report_"${OLD_VERSION}"_to_"${NEW_VERSION}".csv
    rm -f "$REPORT_DIR"/fairness_regression_report_"${OLD_VERSION}"_to_"${NEW_VERSION}".csv
  else
    log "KEEP_EXISTING=1 set; existing v4 snapshot/report artifacts will be reused when possible for profile=$profile"
  fi

  log "Running snapshot stage for profile=$profile version=$NEW_VERSION"
  run_python pilot/pipeline.py \
    --stage snapshot \
    --profile "$profile" \
    --model-version "$NEW_VERSION" \
    --model-loader "$MODEL_LOADER" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --corpus-dir "$CORPUS_DIR"

  log "Running diff stage for profile=$profile transition=${OLD_VERSION}->${NEW_VERSION}"
  run_python pilot/pipeline.py \
    --stage diff \
    --profile "$profile" \
    --transition "${OLD_VERSION}->${NEW_VERSION}" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --corpus-dir "$CORPUS_DIR" \
    --report-dir "$REPORT_DIR"

  log "Completed v4 evaluation for profile=$profile"
done

log "All requested v4 evaluation profiles completed"
