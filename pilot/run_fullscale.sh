#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ARTIFACT_ROOT="$ROOT_DIR/pilot/artifacts"
CORPUS_DIR="$ARTIFACT_ROOT/corpus"
SNAPSHOT_DIR="$ARTIFACT_ROOT/snapshots"
REPORT_DIR="$ARTIFACT_ROOT/regression_reports"
MANUAL_VALIDATION_DIR="$ARTIFACT_ROOT/manual_validation"
LOG_FILE="$ARTIFACT_ROOT/fullscale_pilot_run.log"

KEEP_EXISTING="${KEEP_EXISTING:-0}"
export PYTHONUNBUFFERED=1

mkdir -p "$ARTIFACT_ROOT" "$CORPUS_DIR" "$SNAPSHOT_DIR" "$REPORT_DIR" "$MANUAL_VALIDATION_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_python() {
  uv run python -u "$@"
}

log "Starting full-scale Chrysalis pilot"
log "Dataset policy: SA=SST-2+IMDb, NLI=SNLI+MultiNLI, Generic=SA+NLI+AG News"
log "Repository root: $ROOT_DIR"

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
  --full-spec-train \
  --device cuda

log "Running corpus stage with full available source splits"
run_python pilot/pipeline.py \
  --stage corpus \
  --corpus-dir "$CORPUS_DIR" \
  --manual-validation-dir "$MANUAL_VALIDATION_DIR"

for version in v1_base v2_retrain v3_distilled; do
  log "Running snapshot stage for $version"
  run_python pilot/pipeline.py \
    --stage snapshot \
    --model-version "$version" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --corpus-dir "$CORPUS_DIR"
done

for transition in "v1_base->v2_retrain" "v2_retrain->v3_distilled"; do
  log "Running diff stage for $transition"
  run_python pilot/pipeline.py \
    --stage diff \
    --transition "$transition" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --corpus-dir "$CORPUS_DIR" \
    --report-dir "$REPORT_DIR"
done

log "Running full test suite"
uv run pytest tests/ -v --tb=short

log "Full-scale Chrysalis pilot completed"
log "Detailed log written to $LOG_FILE"
