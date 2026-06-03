#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENAR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ENAR_ROOT}/env/bin/python}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/test_load_sd_v1_5.py" "$@"
