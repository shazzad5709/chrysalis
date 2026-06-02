#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "The study setup will:"
echo "  1. install uv if it is missing,"
echo "  2. create a local .venv, and"
echo "  3. install the study runtime dependency from requirements.txt."
echo
read -r -p "Do you want to continue? [y/N] " STUDY_CONSENT
case "${STUDY_CONSENT}" in
  y|Y|yes|YES)
    ;;
  *)
    echo "Setup cancelled. No installation was performed."
    exit 0
    ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Failed to install uv automatically." >&2
  echo "Please install uv manually and rerun ./user-study/setup_study.sh" >&2
  exit 1
fi

echo "Using uv: $(uv --version)"
echo "Provisioning Python 3.10+ and creating .venv ..."
uv venv --python 3.10 .venv
echo "Installing study runtime dependencies ..."
uv pip install --python .venv/bin/python -r requirements.txt

echo
echo "Setup complete."
echo "Next, run: ./user-study/run_study.sh"
