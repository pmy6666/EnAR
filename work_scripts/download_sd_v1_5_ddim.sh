#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARAM_FILE="${1:-${ENAR_ROOT}/pre_model/DDIM/stable_diffusion_v1_5.params.env}"
PYTHON_BIN="${PYTHON_BIN:-${ENAR_ROOT}/env/bin/python}"

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Parameter file not found: ${PARAM_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${PARAM_FILE}"
set +a

: "${MODEL_REPO_ID:?MODEL_REPO_ID is required}"
: "${MODEL_REVISION:=main}"
: "${DOWNLOAD_BACKEND:=modelscope}"
: "${MODELSCOPE_MODEL_ID:=AI-ModelScope/stable-diffusion-v1-5}"
: "${MODELSCOPE_REVISION:=master}"
: "${MODELSCOPE_DISABLE_SSL_VERIFY:=0}"
: "${MODEL_LOCAL_DIR:=stable-diffusion-v1-5}"
: "${HF_ENDPOINT:=https://huggingface.co}"
: "${HF_DOWNLOAD_MAX_WORKERS:=8}"
: "${HF_XET_HIGH_PERFORMANCE:=1}"
: "${HF_ALLOW_PATTERNS:=}"
: "${HF_IGNORE_PATTERNS:=}"
: "${HF_FILE_LIST:=}"

TARGET_DIR="${ENAR_ROOT}/pre_model/DDIM/${MODEL_LOCAL_DIR}"
mkdir -p "${TARGET_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN=/path/to/python or create the EnAR/env environment first." >&2
  exit 1
fi

export MODEL_REPO_ID
export MODEL_REVISION
export DOWNLOAD_BACKEND
export MODELSCOPE_MODEL_ID
export MODELSCOPE_REVISION
export MODELSCOPE_DISABLE_SSL_VERIFY
export TARGET_DIR
export HF_ENDPOINT
export HF_DOWNLOAD_MAX_WORKERS
export HF_XET_HIGH_PERFORMANCE
export HF_ALLOW_PATTERNS
export HF_IGNORE_PATTERNS
export HF_FILE_LIST

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

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
        "  /home/qianustb/EnAR/env/bin/python -m pip install -U requests tqdm",
        file=sys.stderr,
    )
    sys.exit(2)

from fnmatch import fnmatch
from urllib.parse import quote

from tqdm import tqdm


def split_patterns(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


repo_id = os.environ["MODEL_REPO_ID"]
revision = os.environ["MODEL_REVISION"]
backend = os.environ.get("DOWNLOAD_BACKEND", "modelscope")
modelscope_model_id = os.environ.get("MODELSCOPE_MODEL_ID", "AI-ModelScope/stable-diffusion-v1-5")
modelscope_revision = os.environ.get("MODELSCOPE_REVISION", "master")
modelscope_disable_ssl_verify = os.environ.get("MODELSCOPE_DISABLE_SSL_VERIFY", "0") == "1"
target_dir = Path(os.environ["TARGET_DIR"]).resolve()
endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
max_workers = int(os.environ.get("HF_DOWNLOAD_MAX_WORKERS", "8"))
allow_patterns = split_patterns(os.environ.get("HF_ALLOW_PATTERNS", ""))
ignore_patterns = split_patterns(os.environ.get("HF_IGNORE_PATTERNS", ""))
configured_files = split_patterns(os.environ.get("HF_FILE_LIST", ""))
token = os.environ.get("HF_TOKEN") or None

print(f"Backend: {backend}")
print(f"Downloading {repo_id}@{revision}")
print(f"Target: {target_dir}")
print(f"Endpoint: {endpoint}")
print(f"Max workers: {max_workers}")

if backend == "modelscope":
    if modelscope_disable_ssl_verify:
        import urllib3

        original_request = requests.sessions.Session.request

        def request_without_ssl_verify(self, method, url, **kwargs):
            kwargs.setdefault("verify", False)
            return original_request(self, method, url, **kwargs)

        requests.sessions.Session.request = request_without_ssl_verify
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        print("Warning: ModelScope SSL certificate verification is disabled.")

    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    except ImportError:
        print(
            "Missing dependency: modelscope\n"
            "Install it with:\n"
            "  /home/qianustb/EnAR/env/bin/python -m pip install -U modelscope",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"ModelScope model: {modelscope_model_id}@{modelscope_revision}")
    snapshot_path = ms_snapshot_download(
        modelscope_model_id,
        revision=modelscope_revision,
        local_dir=str(target_dir),
    )
    manifest = {
        "backend": backend,
        "modelscope_model_id": modelscope_model_id,
        "revision": modelscope_revision,
        "snapshot_path": str(Path(snapshot_path).resolve()),
        "target_dir": str(target_dir),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = target_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    print(f"Done. Manifest written to {manifest_path}")
    sys.exit(0)

if backend != "huggingface":
    print(f"Unsupported DOWNLOAD_BACKEND: {backend}", file=sys.stderr)
    sys.exit(2)

session = requests.Session()
session.trust_env = True

api_headers = {}
if token:
    api_headers["Authorization"] = f"Bearer {token}"


def request_with_retry(method: str, url: str, **kwargs):
    last_error = None
    for attempt in range(1, 6):
        try:
            response = session.request(method, url, timeout=(20, 120), **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            print(f"retry {attempt}/5 {method} {url}: {exc}", file=sys.stderr, flush=True)
    raise last_error


if configured_files:
    info = {}
    commit_sha = revision
else:
    info_url = f"{endpoint}/api/models/{repo_id}/revision/{revision}"
    info = request_with_retry("GET", info_url, headers=api_headers, allow_redirects=False).json()
    commit_sha = info.get("sha") or revision


def selected(filename: str) -> bool:
    allowed = True if not allow_patterns else any(fnmatch(filename, pat) for pat in allow_patterns)
    ignored = any(fnmatch(filename, pat) for pat in ignore_patterns)
    return allowed and not ignored


if configured_files:
    files = [filename for filename in configured_files if selected(filename)]
else:
    files = sorted(sibling["rfilename"] for sibling in info.get("siblings", []) if selected(sibling["rfilename"]))
if not files:
    print("No files matched the configured allow/ignore patterns.", file=sys.stderr)
    sys.exit(3)

headers = {}
if token:
    headers["Authorization"] = f"Bearer {token}"


def download_one(filename: str) -> str:
    rel_url = quote(filename, safe="/")
    url = f"{endpoint}/{repo_id}/resolve/{commit_sha}/{rel_url}"
    output_path = target_dir / filename
    part_path = output_path.with_name(output_path.name + ".part")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        return f"skip {filename}"

    request_headers = dict(headers)
    existing = part_path.stat().st_size if part_path.exists() else 0
    mode = "ab" if existing else "wb"
    if existing:
        request_headers["Range"] = f"bytes={existing}-"

    with session.get(url, headers=request_headers, stream=True, timeout=(20, 120)) as response:
        if existing and response.status_code == 200:
            existing = 0
            mode = "wb"
        elif response.status_code not in (200, 206):
            response.raise_for_status()

        content_length = int(response.headers.get("Content-Length") or 0)
        total_size = existing + content_length if content_length else None
        progress = tqdm(
            total=total_size,
            initial=existing,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=filename,
            leave=True,
        )
        try:
            with part_path.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        progress.update(len(chunk))
        finally:
            progress.close()

    part_path.replace(output_path)
    return f"done {filename}"


failures = []
for filename in tqdm(files, total=len(files), desc="Downloading files"):
    try:
        result = download_one(filename)
        print(result, flush=True)
    except Exception as exc:
        failures.append((filename, str(exc)))
        print(f"fail {filename}: {exc}", file=sys.stderr, flush=True)

if failures:
    print("Download failed for these files:", file=sys.stderr)
    for filename, error in failures:
        print(f"  - {filename}: {error}", file=sys.stderr)
    sys.exit(4)

manifest = {
    "repo_id": repo_id,
    "revision": revision,
    "commit_sha": commit_sha,
    "target_dir": str(target_dir),
    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    "allow_patterns": allow_patterns,
    "ignore_patterns": ignore_patterns,
    "files": files,
}

manifest_path = target_dir / "download_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")

print(f"Done. Manifest written to {manifest_path}")
PY
