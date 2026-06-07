#!/usr/bin/env python3
"""Download the VLMBias dataset into EnAR/toy_dataset.

Official dataset repo: https://huggingface.co/datasets/anvo25/vlms-are-biased
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


REPO_ID = "anvo25/vlms-are-biased"
DEFAULT_REVISION = "main"
SPLIT_PATTERNS = {
    "main": "data/main-*",
    "identification": "data/identification-*",
    "withtitle": "data/withtitle-*",
    "original": "data/original-*",
    "remove_background_q1q2": "data/remove_background_q1q2-*",
    "remove_background_q3": "data/remove_background_q3-*",
}


def enar_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download VLMBias parquet files from Hugging Face.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=enar_root_from_script() / "toy_dataset" / "VLMBias",
        help="Directory where the dataset snapshot will be written.",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Hugging Face branch, tag, or commit.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=["all", *SPLIT_PATTERNS.keys()],
        default=None,
        help=(
            "Dataset split to download. Repeat the flag for multiple splits. "
            "Use 'all' for every split."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("HF_DOWNLOAD_MAX_WORKERS", "8")),
        help="Parallel download workers used by huggingface_hub.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "direct", "huggingface_hub"],
        default="auto",
        help=(
            "Download backend. 'direct' avoids huggingface_hub file metadata "
            "checks and works better with some HF mirror endpoints."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        help="Hugging Face endpoint or mirror endpoint.",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Skip README.md and only download data files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matched files and write no dataset files.",
    )
    return parser.parse_args()


def selected_patterns(splits: list[str] | None, include_readme: bool) -> list[str]:
    requested = splits or ["main"]
    if "all" in requested:
        patterns = sorted(SPLIT_PATTERNS.values())
    else:
        patterns = [SPLIT_PATTERNS[split] for split in requested]

    if include_readme:
        patterns.extend(["README.md", ".gitattributes"])
    return patterns


def endpoint_for_requests(endpoint: str) -> str:
    return endpoint.rstrip("/")


def should_use_direct_backend(backend: str, endpoint: str) -> bool:
    if backend == "direct":
        return True
    if backend == "huggingface_hub":
        return False
    return endpoint_for_requests(endpoint) != "https://huggingface.co"


def list_repo_files_direct(
    *,
    repo_id: str,
    revision: str,
    endpoint: str,
    token: str | None,
) -> tuple[str, list[str]]:
    try:
        import requests
    except ImportError:
        print(
            "Missing dependency: requests\n"
            "Install it with:\n"
            "  /home/qianustb/EnAR/env/bin/python -m pip install -U requests tqdm",
            file=sys.stderr,
        )
        raise SystemExit(2)

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{endpoint_for_requests(endpoint)}/api/datasets/{repo_id}/revision/{revision}"
    response = requests.get(url, headers=headers, timeout=(20, 120))
    response.raise_for_status()
    payload = response.json()
    siblings = payload.get("siblings") or []
    files = sorted(
        sibling["rfilename"]
        for sibling in siblings
        if isinstance(sibling, dict) and sibling.get("rfilename")
    )
    return payload.get("sha") or revision, files


def filter_files(files: list[str], allow_patterns: list[str]) -> list[str]:
    return [
        filename
        for filename in files
        if any(fnmatch(filename, pattern) for pattern in allow_patterns)
    ]


def download_files_direct(
    *,
    repo_id: str,
    revision: str,
    endpoint: str,
    target_dir: Path,
    files: list[str],
    token: str | None,
) -> None:
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print(
            "Missing dependency: requests/tqdm\n"
            "Install them with:\n"
            "  /home/qianustb/EnAR/env/bin/python -m pip install -U requests tqdm",
            file=sys.stderr,
        )
        raise SystemExit(2)

    session = requests.Session()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    base_url = endpoint_for_requests(endpoint)

    for index, filename in enumerate(files, start=1):
        rel_url = quote(filename, safe="/")
        url = f"{base_url}/datasets/{repo_id}/resolve/{revision}/{rel_url}"
        output_path = target_dir / filename
        part_path = output_path.with_name(output_path.name + ".part")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[{index}/{len(files)}] skip existing {filename}")
            continue

        request_headers = dict(headers)
        existing_size = part_path.stat().st_size if part_path.exists() else 0
        mode = "ab" if existing_size else "wb"
        if existing_size:
            request_headers["Range"] = f"bytes={existing_size}-"

        print(f"[{index}/{len(files)}] download {filename}")
        with session.get(
            url,
            headers=request_headers,
            stream=True,
            timeout=(20, 120),
            allow_redirects=True,
        ) as response:
            if existing_size and response.status_code == 200:
                existing_size = 0
                mode = "wb"
            elif response.status_code not in (200, 206):
                response.raise_for_status()

            content_length = int(response.headers.get("Content-Length") or 0)
            total_size = existing_size + content_length if content_length else None
            progress = tqdm(
                total=total_size,
                initial=existing_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
            )
            try:
                with part_path.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        progress.update(len(chunk))
            finally:
                progress.close()

        part_path.replace(output_path)


def main() -> int:
    args = parse_args()
    target_dir = args.target_dir.expanduser().resolve()
    allow_patterns = selected_patterns(args.split, include_readme=not args.no_readme)
    endpoint = endpoint_for_requests(args.endpoint)
    token = os.environ.get("HF_TOKEN") or None

    print(f"Repo: {args.repo_id}@{args.revision}")
    print(f"Target: {target_dir}")
    print(f"Endpoint: {endpoint}")
    print("Allow patterns:")
    for pattern in allow_patterns:
        print(f"  - {pattern}")

    target_dir.mkdir(parents=True, exist_ok=True)

    if should_use_direct_backend(args.backend, endpoint):
        print("Backend: direct")
        resolved_sha, repo_files = list_repo_files_direct(
            repo_id=args.repo_id,
            revision=args.revision,
            endpoint=endpoint,
            token=token,
        )
        selected_files = filter_files(repo_files, allow_patterns)
        if not selected_files:
            print("No files matched the configured allow patterns.", file=sys.stderr)
            return 3
        print("Files:")
        for filename in selected_files:
            print(f"  - {filename}")
        if args.dry_run:
            print("Dry run: no files downloaded.")
            return 0
        download_files_direct(
            repo_id=args.repo_id,
            revision=resolved_sha,
            endpoint=endpoint,
            target_dir=target_dir,
            files=selected_files,
            token=token,
        )
        snapshot_path = target_dir
    else:
        print("Backend: huggingface_hub")
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError:
            print(
                "Missing dependency: huggingface_hub\n"
                "Install it with:\n"
                "  /home/qianustb/EnAR/env/bin/python -m pip install -U huggingface_hub",
                file=sys.stderr,
            )
            return 2

        snapshot_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=str(target_dir),
            allow_patterns=allow_patterns,
            max_workers=args.max_workers,
        )
        api = HfApi(endpoint=endpoint)
        repo_info = api.dataset_info(repo_id=args.repo_id, revision=args.revision)
        resolved_sha = repo_info.sha
        if args.dry_run:
            print("Dry run: no files downloaded.")
            return 0

    downloaded_files = sorted(
        str(path.relative_to(target_dir))
        for path in target_dir.rglob("*")
        if path.is_file()
        and not path.name.endswith(".part")
        and not any(
            part in {".cache", ".git"} for part in path.relative_to(target_dir).parts
        )
    )

    manifest = {
        "dataset": "VLMBias",
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "revision": args.revision,
        "resolved_sha": resolved_sha,
        "endpoint": endpoint,
        "backend": "direct"
        if should_use_direct_backend(args.backend, endpoint)
        else "huggingface_hub",
        "snapshot_path": str(Path(snapshot_path).resolve()),
        "target_dir": str(target_dir),
        "requested_splits": args.split or ["main"],
        "allow_patterns": allow_patterns,
        "downloaded_files": downloaded_files,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = target_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")

    print(f"Done. Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
