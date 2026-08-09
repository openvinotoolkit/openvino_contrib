# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download a current-run artifact using concurrent HTTP ranges."""

import concurrent.futures
import os
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .artifact import MAX_ARTIFACT_BYTES
from .core import GITHUB_API_TIMEOUT, github_api_headers, required_env

_MIB = 1024 * 1024
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class ArtifactDownloadError(RuntimeError):
    pass


class _RetryableDownload(ArtifactDownloadError):
    pass


def _resolve_artifact(name: str, token: str) -> tuple[str, int]:
    headers = github_api_headers(token)
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    repository = required_env("GITHUB_REPOSITORY")
    run_id = required_env("GITHUB_RUN_ID")
    url = f"{api}/repos/{repository}/actions/runs/{run_id}/artifacts"
    try:
        response = requests.get(url, headers=headers, params={"name": name}, timeout=GITHUB_API_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("artifacts"), list):
            raise TypeError("unexpected artifact metadata")
        artifacts = [
            item
            for item in payload["artifacts"]
            if isinstance(item, dict) and item.get("name") == name and not item.get("expired")
        ]
        if len(artifacts) != 1:
            raise ArtifactDownloadError(
                f"expected one active artifact named '{name}', found {len(artifacts)}"
            )
        size = int(artifacts[0]["size_in_bytes"])
        response = requests.get(
            artifacts[0]["archive_download_url"],
            headers=headers,
            allow_redirects=False,
            timeout=GITHUB_API_TIMEOUT,
        )
    except (KeyError, TypeError, ValueError, requests.RequestException) as error:
        raise ArtifactDownloadError(f"failed to resolve artifact ({type(error).__name__})") from None

    location = response.headers.get("Location", "")
    parsed = urlparse(location)
    if response.status_code not in (302, 307) or parsed.scheme != "https" or not parsed.netloc:
        raise ArtifactDownloadError("artifact endpoint did not return an absolute HTTPS redirect")
    if not 0 < size <= MAX_ARTIFACT_BYTES:
        raise ArtifactDownloadError(f"artifact has invalid size {size}")
    return location, size


@retry(
    retry=retry_if_exception_type(_RetryableDownload),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _download_range(url: str, descriptor: int, start: int, end: int) -> int:
    try:
        with requests.get(
            url,
            headers={"Accept-Encoding": "identity", "Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=(30, 300),
        ) as response:
            if response.status_code != 206:
                error = f"range {start}-{end} returned HTTP {response.status_code}"
                if response.status_code in _RETRYABLE_STATUS:
                    raise _RetryableDownload(error)
                raise ArtifactDownloadError(error)
            offset = start
            for block in response.iter_content(chunk_size=_MIB):
                if block:
                    if offset + len(block) > end + 1:
                        raise _RetryableDownload(f"range {start}-{end} returned too much data")
                    pending = memoryview(block)
                    while pending:
                        written = os.pwrite(descriptor, pending, offset)
                        if written <= 0:
                            raise _RetryableDownload(f"failed to write range {start}-{end}")
                        offset += written
                        pending = pending[written:]
    except requests.RequestException as error:
        raise _RetryableDownload(f"range {start}-{end} failed ({type(error).__name__})") from None
    if offset != end + 1:
        raise _RetryableDownload(f"range {start}-{end} returned {offset - start} bytes")
    return offset - start


def _extract(archive_path: Path, destination: Path, expected: str) -> None:
    if Path(expected).name != expected:
        raise ArtifactDownloadError(f"invalid artifact payload name '{expected}'")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != expected or members[0].is_dir():
                raise ArtifactDownloadError(f"artifact ZIP must contain only '{expected}'")
            if members[0].file_size > MAX_ARTIFACT_BYTES:
                raise ArtifactDownloadError("artifact payload is too large")
            archive.extract(members[0], destination)
    except (RuntimeError, zipfile.BadZipFile):
        raise ArtifactDownloadError("downloaded artifact is not a valid ZIP") from None


def download(name: str, destination: Path, workers: int = 8, chunk_size: int = 16 * _MIB) -> None:
    token = required_env("GITHUB_TOKEN")
    url, size = _resolve_artifact(name, token)
    ranges = [(start, min(start + chunk_size, size) - 1) for start in range(0, size, chunk_size)]
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    temporary = tempfile.NamedTemporaryFile(prefix="artifact-", suffix=".zip", dir=destination, delete=False)
    try:
        with temporary:
            temporary.truncate(size)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                downloads = [
                    executor.submit(_download_range, url, temporary.fileno(), start, end)
                    for start, end in ranges
                ]
                completed = 0
                for future in concurrent.futures.as_completed(downloads):
                    completed += future.result()
                    print(f"  downloaded {completed / size:6.1%}")
        _extract(Path(temporary.name), destination, f"{name}.tar.gz")
    finally:
        Path(temporary.name).unlink(missing_ok=True)
    print(f"✓ downloaded {size / _MIB:.1f} MiB in {time.monotonic() - started:.1f}s")
