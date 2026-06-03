#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARAM_FILE="${1:-${ENAR_ROOT}/pre_model/LLM/llava_1_5_7b_modelscope.params.env}"
PYTHON_BIN="${PYTHON_BIN:-${ENAR_ROOT}/env/bin/python}"

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Parameter file not found: ${PARAM_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${PARAM_FILE}"
set +a

: "${MODELSCOPE_MODEL_ID:=swift/llava-1.5-7b-hf}"
: "${MODELSCOPE_REVISION:=master}"
: "${MODELSCOPE_DISABLE_SSL_VERIFY:=0}"
: "${MODEL_LOCAL_DIR:=llava-1.5-7b-hf}"

TARGET_DIR="${ENAR_ROOT}/pre_model/LLM/${MODEL_LOCAL_DIR}"
mkdir -p "${TARGET_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN=/path/to/python or create the EnAR/env environment first." >&2
  exit 1
fi

export MODELSCOPE_MODEL_ID
export MODELSCOPE_REVISION
export MODELSCOPE_DISABLE_SSL_VERIFY
export TARGET_DIR

"${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print(
        "Missing dependency: requests\n"
        "Install it with:\n"
        "  /home/qianustb/EnAR/env/bin/python -m pip install -U requests",
        file=sys.stderr,
    )
    sys.exit(2)

model_id = os.environ["MODELSCOPE_MODEL_ID"]
revision = os.environ.get("MODELSCOPE_REVISION", "master")
target_dir = Path(os.environ["TARGET_DIR"]).resolve()
disable_ssl_verify = os.environ.get("MODELSCOPE_DISABLE_SSL_VERIFY", "0") == "1"

if disable_ssl_verify:
    import urllib3

    original_request = requests.sessions.Session.request

    def request_without_ssl_verify(self, method, url, **kwargs):
        kwargs.setdefault("verify", False)
        return original_request(self, method, url, **kwargs)

    requests.sessions.Session.request = request_without_ssl_verify
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    print("Warning: ModelScope SSL certificate verification is disabled.")

try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    print(
        "Missing dependency: modelscope\n"
        "Install it with:\n"
        "  /home/qianustb/EnAR/env/bin/python -m pip install -U modelscope",
        file=sys.stderr,
    )
    sys.exit(2)

print(f"ModelScope model: {model_id}@{revision}")
print(f"Target: {target_dir}")

snapshot_path = snapshot_download(
    model_id,
    revision=revision,
    local_dir=str(target_dir),
)

manifest = {
    "backend": "modelscope",
    "modelscope_model_id": model_id,
    "revision": revision,
    "snapshot_path": str(Path(snapshot_path).resolve()),
    "target_dir": str(target_dir),
    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
}

manifest_path = target_dir / "download_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
print(f"Done. Manifest written to {manifest_path}")
PY
