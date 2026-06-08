#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENAR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${ENAR_DIR}/env/bin/python"
RUN_SCRIPT="${SCRIPT_DIR}/run_small_dataset_simple_test.py"
LOG_DIR="${ENAR_DIR}/outputs/simple_test/background_logs"

mkdir -p "${LOG_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/simple_test_${timestamp}.log"
pid_file="${LOG_DIR}/simple_test_${timestamp}.pid"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Runner not found: ${RUN_SCRIPT}" >&2
  exit 1
fi

cd "${ENAR_DIR}/.."
nohup "${PYTHON_BIN}" "${RUN_SCRIPT}" "$@" >"${log_file}" 2>&1 &
pid="$!"
echo "${pid}" >"${pid_file}"

echo "Started simple test in background."
echo "PID: ${pid}"
echo "PID file: ${pid_file}"
echo "Log: ${log_file}"
echo
echo "Watch:"
echo "  tail -f ${log_file}"
echo "Stop:"
echo "  kill ${pid}"
