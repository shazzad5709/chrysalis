#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m alteron.ci.runner \
  --config "${SCRIPT_DIR}/study_ci.yml" \
  --profile pr-fast \
  --candidate-model-dir "${SCRIPT_DIR}/study_model/candidate" \
  --candidate-version candidate \
  --baseline-snapshot-dir "${SCRIPT_DIR}/study_input/snapshots/stable" \
  --baseline-version stable \
  --corpus-dir "${SCRIPT_DIR}/study_input/corpus" \
  --output-dir "${SCRIPT_DIR}/study_output" \
  --model-loader "./user-study/study_loader.py:load_model" \
  --force
EXIT_CODE=$?

echo
if [[ "${EXIT_CODE}" -eq 0 ]]; then
  echo "Study run completed with a passing CI result."
elif [[ "${EXIT_CODE}" -eq 1 ]]; then
  echo "Study run completed with a blocking CI result."
  echo "For this bundled study scenario, that result is expected."
  echo "Inspect:"
  echo "  user-study/study_output/ci_summary.json"
  echo "  user-study/study_output/regression_reports/regression_report_stable_to_candidate.csv"
else
  echo "Study run exited with unexpected code ${EXIT_CODE}."
fi

exit "${EXIT_CODE}"
