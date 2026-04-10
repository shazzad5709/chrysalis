#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

ARTIFACT_ROOT="$ROOT_DIR/pilot/artifacts"
MASTER_LOG_FILE="$ARTIFACT_ROOT/run_chrysalis_demo_all_profiles.log"

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

V1_VERSION="${V1_VERSION:-v1_base}"
V2_VERSION="${V2_VERSION:-v2_retrain}"
V3_VERSION="${V3_VERSION:-v3_distilled}"
V4_VERSION="${V4_VERSION:-v4_quantized}"
DEFAULT_MODEL_LOADER="${DEFAULT_MODEL_LOADER:-pilot/model_loader.py:load_model_bundle}"
V4_MODEL_LOADER="${V4_MODEL_LOADER:-pilot/quantized_model_loader.py:load_model_bundle}"
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

run_profile_demo() {
  local profile="$1"
  local profile_artifact_root="$ARTIFACT_ROOT/$profile"
  local corpus_dir="$profile_artifact_root/corpus"
  local snapshot_dir="$profile_artifact_root/snapshots"
  local report_dir="$profile_artifact_root/regression_reports"
  local manual_validation_dir="$profile_artifact_root/manual_validation"
  local profile_log_file="$profile_artifact_root/chrysalis_demo.log"

  mkdir -p "$profile_artifact_root" "$corpus_dir" "$snapshot_dir" "$report_dir" "$manual_validation_dir"

  {
    log "Starting Chrysalis demonstration for profile=$profile"
    log "Artifact root: $profile_artifact_root"
    log "Model versions: $V1_VERSION, $V2_VERSION, $V3_VERSION, $V4_VERSION"

    if [[ "$KEEP_EXISTING" != "1" ]]; then
      log "Removing stale corpus/snapshot/report artifacts for profile=$profile"
      rm -rf "$corpus_dir" "$manual_validation_dir" "$snapshot_dir/$V1_VERSION" "$snapshot_dir/$V2_VERSION" "$snapshot_dir/$V3_VERSION" "$snapshot_dir/$V4_VERSION"
      rm -f "$report_dir"/regression_report_*.csv "$report_dir"/fairness_regression_report_*.csv
      mkdir -p "$corpus_dir" "$manual_validation_dir" "$snapshot_dir" "$report_dir"
    else
      log "KEEP_EXISTING=1 set; existing artifacts will be reused when possible for profile=$profile"
    fi

    log "Running corpus stage for profile=$profile"
    run_python pilot/pipeline.py \
      --stage corpus \
      --profile "$profile" \
      --corpus-dir "$corpus_dir" \
      --manual-validation-dir "$manual_validation_dir"

    log "Running snapshot stage for profile=$profile version=$V1_VERSION"
    run_python pilot/pipeline.py \
      --stage snapshot \
      --profile "$profile" \
      --model-version "$V1_VERSION" \
      --model-loader "$DEFAULT_MODEL_LOADER" \
      --snapshot-dir "$snapshot_dir" \
      --corpus-dir "$corpus_dir"

    log "Running snapshot stage for profile=$profile version=$V2_VERSION"
    run_python pilot/pipeline.py \
      --stage snapshot \
      --profile "$profile" \
      --model-version "$V2_VERSION" \
      --model-loader "$DEFAULT_MODEL_LOADER" \
      --snapshot-dir "$snapshot_dir" \
      --corpus-dir "$corpus_dir"

    log "Running snapshot stage for profile=$profile version=$V3_VERSION"
    run_python pilot/pipeline.py \
      --stage snapshot \
      --profile "$profile" \
      --model-version "$V3_VERSION" \
      --model-loader "$DEFAULT_MODEL_LOADER" \
      --snapshot-dir "$snapshot_dir" \
      --corpus-dir "$corpus_dir"

    log "Running snapshot stage for profile=$profile version=$V4_VERSION"
    run_python pilot/pipeline.py \
      --stage snapshot \
      --profile "$profile" \
      --model-version "$V4_VERSION" \
      --model-loader "$V4_MODEL_LOADER" \
      --snapshot-dir "$snapshot_dir" \
      --corpus-dir "$corpus_dir"

    log "Running diff stage for profile=$profile transition=${V1_VERSION}->${V2_VERSION}"
    run_python pilot/pipeline.py \
      --stage diff \
      --profile "$profile" \
      --transition "${V1_VERSION}->${V2_VERSION}" \
      --snapshot-dir "$snapshot_dir" \
      --corpus-dir "$corpus_dir" \
      --report-dir "$report_dir"

    log "Running diff stage for profile=$profile transition=${V2_VERSION}->${V3_VERSION}"
    run_python pilot/pipeline.py \
      --stage diff \
      --profile "$profile" \
      --transition "${V2_VERSION}->${V3_VERSION}" \
      --snapshot-dir "$snapshot_dir" \
      --corpus-dir "$corpus_dir" \
      --report-dir "$report_dir"

    log "Running diff stage for profile=$profile transition=${V3_VERSION}->${V4_VERSION}"
    run_python pilot/pipeline.py \
      --stage diff \
      --profile "$profile" \
      --transition "${V3_VERSION}->${V4_VERSION}" \
      --snapshot-dir "$snapshot_dir" \
      --corpus-dir "$corpus_dir" \
      --report-dir "$report_dir"

    log "Completed Chrysalis demonstration for profile=$profile"
    log "Detailed profile log: $profile_log_file"
  } 2>&1 | tee -a "$profile_log_file"
}

log "Starting all-profile Chrysalis demonstration run"
log "Master log: $MASTER_LOG_FILE"
log "Configured versions: v1=$V1_VERSION v2=$V2_VERSION v3=$V3_VERSION v4=$V4_VERSION"
log "Loaders: default=$DEFAULT_MODEL_LOADER v4=$V4_MODEL_LOADER"

mapfile -t profiles < <(resolve_profiles)

for profile in "${profiles[@]}"; do
  run_profile_demo "$profile"
done

log "All requested Chrysalis demonstration profiles completed"
