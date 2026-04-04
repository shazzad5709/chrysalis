#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ARTIFACT_ROOT="$ROOT_DIR/pilot/artifacts"
KEEP_EXISTING="${KEEP_EXISTING:-0}"
PROFILE="${PROFILE:-sa_sst2}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRAD_ACC="${GRAD_ACC:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROFILE_ARTIFACT_ROOT="$ARTIFACT_ROOT/$PROFILE"
CORPUS_DIR="$PROFILE_ARTIFACT_ROOT/corpus"
SNAPSHOT_DIR="$PROFILE_ARTIFACT_ROOT/snapshots"
REPORT_DIR="$PROFILE_ARTIFACT_ROOT/regression_reports"
MANUAL_VALIDATION_DIR="$PROFILE_ARTIFACT_ROOT/manual_validation"
LOG_FILE="$PROFILE_ARTIFACT_ROOT/fullscale_pilot_run.log"
export PYTHONUNBUFFERED=1

mkdir -p "$ARTIFACT_ROOT" "$PROFILE_ARTIFACT_ROOT" "$CORPUS_DIR" "$SNAPSHOT_DIR" "$REPORT_DIR" "$MANUAL_VALIDATION_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_python() {
  uv run python -u "$@"
}

log "Starting full-scale Chrysalis pilot"
log "Pipeline profile: $PROFILE"
log "Repository root: $ROOT_DIR"
log "Training config: device=$DEVICE batch_size=$BATCH_SIZE eval_batch_size=$EVAL_BATCH_SIZE grad_acc=$GRAD_ACC num_workers=$NUM_WORKERS"

if [[ "$KEEP_EXISTING" != "1" ]]; then
  log "Removing stale artifacts"
  rm -rf "$SNAPSHOT_DIR/v1_base" "$SNAPSHOT_DIR/v2_retrain" "$SNAPSHOT_DIR/v3_distilled"
  rm -f "$REPORT_DIR"/regression_report_*.csv "$REPORT_DIR"/fairness_regression_report_*.csv
else
  log "KEEP_EXISTING=1 set; existing snapshot and report artifacts will be reused when possible"
fi

log "Running full-spec training for v1_base, v2_retrain, and v3_distilled"
run_python pilot/pipeline.py \
  --stage train \
  --profile "$PROFILE" \
  --full-spec-train \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACC" \
  --num-workers "$NUM_WORKERS"

log "Running corpus stage with full available source splits"
run_python pilot/pipeline.py \
  --stage corpus \
  --profile "$PROFILE" \
  --corpus-dir "$CORPUS_DIR" \
  --manual-validation-dir "$MANUAL_VALIDATION_DIR"

for version in v1_base v2_retrain v3_distilled; do
  log "Running snapshot stage for $version"
  run_python pilot/pipeline.py \
    --stage snapshot \
    --profile "$PROFILE" \
    --model-version "$version" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --corpus-dir "$CORPUS_DIR"
done

for transition in "v1_base->v2_retrain" "v2_retrain->v3_distilled"; do
  log "Running diff stage for $transition"
  run_python pilot/pipeline.py \
    --stage diff \
    --profile "$PROFILE" \
    --transition "$transition" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --corpus-dir "$CORPUS_DIR" \
    --report-dir "$REPORT_DIR"
done

log "Running full test suite"
uv run pytest tests/ -v --tb=short

log "Full-scale Chrysalis pilot completed"
log "Detailed log written to $LOG_FILE"
