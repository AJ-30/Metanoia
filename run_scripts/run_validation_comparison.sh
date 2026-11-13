#!/usr/bin/env bash

# Run the missing-component evaluator for every component on one split and
# write the comparison table under evaluator/outputs/<split>_comparison/.
#
# This is a batch driver, not a training script. It calls the normal evaluator
# entry point three times (tv, showcase, dining_set) and rebuilds the table
# after each component, so an interrupted run still leaves useful results.
#
# Usage:
#   ./run_scripts/run_validation_comparison.sh --skip-existing
#   ./run_scripts/run_validation_comparison.sh --split test --skip-existing
#   ./run_scripts/run_validation_comparison.sh --table-only
#
# On macOS, caffeinate keeps the machine awake for the multi-hour validation
# pass. The script falls back to a plain Python launch on other systems.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -x "./venv/bin/python" ]]; then
  echo "Missing ./venv/bin/python. Activate or create the project venv first." >&2
  exit 1
fi

PYTHON_ARGS=("evaluator/run_validation_comparison.py" "$@")

if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu ./venv/bin/python "${PYTHON_ARGS[@]}"
fi

exec ./venv/bin/python "${PYTHON_ARGS[@]}"
