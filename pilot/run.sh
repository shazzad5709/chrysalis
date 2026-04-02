#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ARTIFACT_ROOT="$ROOT_DIR/pilot/artifacts"
CORPUS_DIR="$ARTIFACT_ROOT/corpus"
SNAPSHOT_DIR="$ARTIFACT_ROOT/snapshots"
REPORT_DIR="$ARTIFACT_ROOT/regression_reports"
MANUAL_VALIDATION_DIR="$ARTIFACT_ROOT/manual_validation"
LOG_FILE="$ARTIFACT_ROOT/session_8b_run.log"

KEEP_EXISTING="${KEEP_EXISTING:-0}"

mkdir -p "$ARTIFACT_ROOT" "$CORPUS_DIR" "$SNAPSHOT_DIR" "$REPORT_DIR" "$MANUAL_VALIDATION_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_python() {
  uv run python "$@"
}

log "Starting Session 8B pipeline"
log "Repository root: $ROOT_DIR"
log "Artifacts root: $ARTIFACT_ROOT"

if [[ "$KEEP_EXISTING" != "1" ]]; then
  log "Removing stale snapshots and regression reports"
  rm -rf "$SNAPSHOT_DIR/v1_base" "$SNAPSHOT_DIR/v2_retrain" "$SNAPSHOT_DIR/v3_distilled"
  rm -f "$REPORT_DIR"/regression_report_*.csv "$REPORT_DIR"/fairness_regression_report_*.csv
else
  log "KEEP_EXISTING=1 set; existing snapshots and reports will be reused where possible"
fi

log "Running corpus stage"
run_python pilot/pipeline.py \
  --stage corpus \
  --corpus-dir "$CORPUS_DIR" \
  --manual-validation-dir "$MANUAL_VALIDATION_DIR"

log "Verifying frozen corpus counts"
python3 - <<'PY'
import csv
from pathlib import Path

corpus_dir = Path("pilot/artifacts/corpus")
counts = {}
for path in sorted(corpus_dir.glob("*_corpus.csv")):
    with path.open(newline="", encoding="utf-8") as handle:
        counts[path.name] = sum(1 for _ in csv.DictReader(handle))

print("Corpus counts:")
for name, count in counts.items():
    print(f"  {name}: {count}")

failures = {name: count for name, count in counts.items() if count < 200}
if failures:
    print("Corpus count check failed:")
    for name, count in failures.items():
        print(f"  {name}: {count} < 200")
    raise SystemExit(1)
PY

log "Checking manual validation artifacts"
python3 - <<'PY'
from pathlib import Path

required = [
    "manual_validation_artifacts_CHR-GEN-005.csv",
    "manual_validation_artifacts_CHR-GEN-018-A.csv",
    "manual_validation_artifacts_CHR-GEN-018-B.csv",
    "manual_validation_artifacts_CHR-GEN-019.csv",
    "manual_validation_artifacts_CHR-NLI-004.csv",
    "manual_validation_artifacts_CHR-NLI-005.csv",
    "manual_validation_artifacts_CHR-NLI-006.csv",
    "manual_validation_artifacts_CHR-SA-001.csv",
    "manual_validation_artifacts_CHR-SA-007.csv",
    "manual_validation_artifacts_CHR-SA-008.csv",
    "manual_validation_artifacts_CHR-SA-010.csv",
]
manual_dir = Path("pilot/artifacts/manual_validation")
missing = [name for name in required if not (manual_dir / name).exists()]
if missing:
    print("Missing manual validation artifacts:")
    for name in missing:
        print(f"  {name}")
    raise SystemExit(1)
print("All manual validation artifacts present.")
PY

for version in v1_base v2_retrain v3_distilled; do
  log "Running snapshot stage for $version"
  run_python pilot/pipeline.py \
    --stage snapshot \
    --model-version "$version" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --corpus-dir "$CORPUS_DIR"
done

log "Verifying snapshot outputs"
python3 - <<'PY'
from pathlib import Path

versions = ["v1_base", "v2_retrain", "v3_distilled"]
expected = {
    "CHR-GEN-005_snapshot.csv",
    "CHR-GEN-018_snapshot.csv",
    "CHR-GEN-019_snapshot.csv",
    "CHR-NLI-004_snapshot.csv",
    "CHR-NLI-005_snapshot.csv",
    "CHR-NLI-006_snapshot.csv",
    "CHR-SA-001_snapshot.csv",
    "CHR-SA-007_snapshot.csv",
    "CHR-SA-008_snapshot.csv",
    "CHR-SA-010_snapshot.csv",
}
for version in versions:
    version_dir = Path("pilot/artifacts/snapshots") / version
    files = {path.name for path in version_dir.glob("*_snapshot.csv")}
    missing = sorted(expected - files)
    extra = sorted(files - expected)
    print(f"{version}: {len(files)} snapshot files")
    if missing:
        print(f"  missing: {missing}")
        raise SystemExit(1)
    if extra:
        print(f"  extra: {extra}")
PY

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

log "Verifying regression reports"
python3 - <<'PY'
import csv
from pathlib import Path

report_dir = Path("pilot/artifacts/regression_reports")
standard_reports = sorted(report_dir.glob("regression_report_*.csv"))
fairness_reports = sorted(report_dir.glob("fairness_regression_report_*.csv"))

if len(standard_reports) != 2:
    print(f"Expected 2 standard reports, found {len(standard_reports)}")
    raise SystemExit(1)

flagged = []
release_blocked = []
fairness_in_standard = False

for path in standard_reports:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"{path.name}: {len(rows)} rows")
    for row in rows:
        if row["mr_id"] == "CHR-NLI-005":
            fairness_in_standard = True
        if row["behavioral_regression_flag"].lower() == "true":
            flagged.append((path.name, row["mr_id"], row["matched_pass_rate_delta"]))
        if row["release_blocked"].lower() == "true":
            release_blocked.append((path.name, row["mr_id"]))

for path in fairness_reports:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"{path.name}: {len(rows)} rows")

print(f"Flagged regressions: {len(flagged)}")
for name, mr_id, delta in flagged:
    print(f"  {name}: {mr_id} delta={delta}")

print(f"Release blocked rows: {len(release_blocked)}")
for name, mr_id in release_blocked:
    print(f"  {name}: {mr_id}")

if fairness_in_standard:
    print("CHR-NLI-005 incorrectly appeared in a standard regression report.")
    raise SystemExit(1)

if not flagged:
    print("No behavioral regressions were flagged.")

print("Report verification complete.")
PY

log "Session 8B pipeline completed"
log "See detailed log at $LOG_FILE"
