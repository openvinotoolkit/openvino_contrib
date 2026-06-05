#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

CHUNK_SIZE = 8 * 1024 * 1024
PROGRESS_STEP = 64 * 1024 * 1024


def github_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def request(url: str, token: str | None, accept: str = "application/vnd.github+json") -> urllib.request.Request:
    headers = {
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "openvino-notes-prebuild-downloader",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def download_request(url: str, token: str | None, range_start: int = 0) -> urllib.request.Request:
    headers = {
        "User-Agent": "openvino-notes-prebuild-downloader",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if range_start > 0:
        headers["Range"] = f"bytes={range_start}-"

    return urllib.request.Request(url, headers=headers)


def head_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={"User-Agent": "openvino-notes-prebuild-downloader"},
        method="HEAD",
    )


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def load_json(url: str, token: str | None, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(request(url, token), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def find_release_asset(repo: str, release_tag: str, artifact_name: str, token: str | None, timeout: int) -> dict[str, Any]:
    release = load_json(f"https://api.github.com/repos/{repo}/releases/tags/{release_tag}", token, timeout)
    for asset in release.get("assets", []):
        if asset.get("name") == artifact_name and asset.get("state") == "uploaded":
            return asset

    available = ", ".join(asset.get("name", "<unnamed>") for asset in release.get("assets", []))
    raise SystemExit(
        f"Release asset '{artifact_name}' was not found in {repo}@{release_tag}. Available assets: {available}",
    )


def direct_release_asset_url(repo: str, release_tag: str, artifact_name: str) -> str:
    escaped_tag = urllib.parse.quote(release_tag, safe="")
    escaped_artifact = urllib.parse.quote(artifact_name, safe="")
    return f"https://github.com/{repo}/releases/download/{escaped_tag}/{escaped_artifact}"


def resolve_asset_download_url(asset: dict[str, Any], token: str | None, timeout: int) -> str:
    browser_url = asset.get("browser_download_url")
    if browser_url:
        return str(browser_url)

    api_url = asset.get("url")
    if not api_url:
        raise SystemExit(f"Release asset metadata is missing a download URL: {asset}")

    opener = urllib.request.build_opener(NoRedirectHandler)
    try:
        with opener.open(request(str(api_url), token, accept="application/octet-stream"), timeout=timeout) as response:
            return response.url
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307, 308):
            location = exc.headers.get("Location")
            if location:
                return location
        raise


def stream_download(url: str, temp_path: Path, token: str | None, timeout: int) -> None:
    resume_from = temp_path.stat().st_size if temp_path.exists() else 0

    with urllib.request.urlopen(download_request(url, token, resume_from), timeout=timeout) as response:
        status = response.status
        append = resume_from > 0 and status == 206
        if resume_from > 0 and not append:
            print("Server did not accept byte-range resume; restarting download.", flush=True)
            resume_from = 0

        mode = "ab" if append else "wb"
        downloaded = resume_from
        next_report = ((downloaded // PROGRESS_STEP) + 1) * PROGRESS_STEP

        with temp_path.open(mode) as output:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break

                output.write(chunk)
                downloaded += len(chunk)

                if downloaded >= next_report:
                    print(f"Downloaded {downloaded // (1024 * 1024)} MiB...", flush=True)
                    next_report += PROGRESS_STEP


def download_with_retries(url: str, destination: Path, token: str | None, timeout: int, retries: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")

    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading GitHub release asset, attempt {attempt}/{retries}")
            stream_download(url, temp_path, token, timeout)
            temp_path.replace(destination)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and temp_path.exists() and temp_path.stat().st_size > 0:
                temp_path.replace(destination)
                return
            if attempt == retries:
                raise
            delay_seconds = min(30, attempt * 5)
            print(f"Download failed: {exc}. Retrying in {delay_seconds}s.", file=sys.stderr)
            time.sleep(delay_seconds)
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise
            delay_seconds = min(30, attempt * 5)
            print(f"Download failed: {exc}. Retrying in {delay_seconds}s.", file=sys.stderr)
            time.sleep(delay_seconds)


def verify_digest(file: Path, digest: str | None) -> None:
    if not digest:
        return
    algorithm, _, expected_hash = digest.partition(":")
    if algorithm.lower() != "sha256" or not expected_hash:
        print(f"Skipping unsupported release asset digest: {digest}", flush=True)
        return

    hasher = hashlib.sha256()
    with file.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()
    if actual_hash.lower() != expected_hash.lower():
        file.unlink(missing_ok=True)
        raise SystemExit(f"SHA-256 mismatch for {file}: expected {expected_hash}, got {actual_hash}")


def metadata_path(file: Path) -> Path:
    return file.with_suffix(file.suffix + ".metadata.json")


def current_asset_metadata(asset: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": asset.get("id"),
        "node_id": asset.get("node_id"),
        "name": asset.get("name"),
        "size": asset.get("size"),
        "digest": asset.get("digest"),
        "created_at": asset.get("created_at"),
        "updated_at": asset.get("updated_at"),
    }


def load_existing_metadata(file: Path) -> dict[str, Any] | None:
    metadata_file = metadata_path(file)
    if not metadata_file.is_file():
        return None
    try:
        return json.loads(metadata_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_asset_metadata(file: Path, asset: dict[str, Any]) -> None:
    metadata_path(file).write_text(
        json.dumps(current_asset_metadata(asset), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_direct_download_metadata(
    file: Path,
    repo: str,
    release_tag: str,
    artifact_name: str,
) -> None:
    metadata_path(file).write_text(
        json.dumps(
            {
                "download": "direct-release-asset",
                "name": artifact_name,
                "release_tag": release_tag,
                "repo": repo,
                "size": file.stat().st_size if file.is_file() else None,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def remote_content_length(url: str, timeout: int) -> int | None:
    try:
        with urllib.request.urlopen(head_request(url), timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return None

    if not content_length:
        return None
    try:
        return int(content_length)
    except ValueError:
        return None


def existing_file_matches_asset(file: Path, asset: dict[str, Any]) -> bool:
    if not file.is_file() or file.stat().st_size <= 0:
        return False

    expected_size = asset.get("size")
    if isinstance(expected_size, int) and file.stat().st_size != expected_size:
        return False

    return load_existing_metadata(file) == current_asset_metadata(asset)


def existing_file_matches_direct_download(
    file: Path,
    repo: str,
    release_tag: str,
    artifact_name: str,
    expected_size: int | None,
) -> bool:
    if not file.is_file() or file.stat().st_size <= 0:
        return False

    actual_size = file.stat().st_size
    if expected_size is not None:
        return actual_size == expected_size

    metadata = load_existing_metadata(file)
    if not metadata:
        return False

    metadata_size = metadata.get("size")
    if not isinstance(metadata_size, int) or actual_size != metadata_size:
        return False

    if metadata.get("download") == "direct-release-asset":
        return (
            metadata.get("repo") == repo
            and metadata.get("release_tag") == release_tag
            and metadata.get("name") == artifact_name
        )

    return metadata.get("name") == artifact_name


def download_direct_release_asset(args: argparse.Namespace) -> None:
    download_url = direct_release_asset_url(args.repo, args.release_tag, args.artifact_name)
    expected_size = remote_content_length(download_url, args.timeout)
    if existing_file_matches_direct_download(
        args.output,
        args.repo,
        args.release_tag,
        args.artifact_name,
        expected_size,
    ):
        existing_metadata = load_existing_metadata(args.output)
        if not existing_metadata or existing_metadata.get("download") == "direct-release-asset":
            write_direct_download_metadata(args.output, args.repo, args.release_tag, args.artifact_name)
        print(f"Reusing existing GitHub release asset after API fallback: {args.output}")
        return

    args.output.unlink(missing_ok=True)
    metadata_path(args.output).unlink(missing_ok=True)
    download_with_retries(download_url, args.output, None, args.timeout, args.retries)
    write_direct_download_metadata(args.output, args.repo, args.release_tag, args.artifact_name)
    print(f"Downloaded GitHub release asset directly: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--release-tag", default="openvino-android-prebuilds-nightly")
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = github_token()
    try:
        asset = find_release_asset(args.repo, args.release_tag, args.artifact_name, token, args.timeout)
    except urllib.error.HTTPError as exc:
        if token or exc.code != 403:
            raise
        print(
            "GitHub release API is rate-limited; falling back to the public release asset URL.",
            file=sys.stderr,
        )
        download_direct_release_asset(args)
        return

    if existing_file_matches_asset(args.output, asset):
        print(f"Reusing current GitHub release asset: {args.output}")
        return

    args.output.unlink(missing_ok=True)
    metadata_path(args.output).unlink(missing_ok=True)
    download_url = resolve_asset_download_url(asset, token, args.timeout)

    download_with_retries(download_url, args.output, token, args.timeout, args.retries)
    verify_digest(args.output, asset.get("digest"))
    write_asset_metadata(args.output, asset)
    print(f"Downloaded GitHub release asset: {args.output}")


if __name__ == "__main__":
    main()
