from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from github import GithubException

from .release import _get_release, _github_repository, _sync_release_assets


DEFAULT_PREBUILD_RELEASE_TAG = "openvino-android-prebuilds-nightly"
DEFAULT_PREBUILD_CHANNEL = "nightly"
DEFAULT_APK_RELEASE_TAG = "openvino-notes-android-apk-nightly"
DEFAULT_APK_RELEASE_TITLE = "OpenVINO Notes Android APK Nightly"
DEFAULT_APK_RELEASE_NOTES_PREFIX = "Rolling nightly signed release APKs built from published OpenVINO Android prebuilds."
APK_RELEASE_MANIFEST_NAME = "openvino-notes-android-apk-manifest.json"
APK_PACKAGE_MANIFEST_PREFIX = "openvino-notes-android-apk-package-manifest"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _write_output(name: str, value: str) -> None:
    output_path = _env("GITHUB_OUTPUT")
    if not output_path:
        print(f"{name}={value}", flush=True)
        return

    with Path(output_path).open("a", encoding="utf-8") as output:
        output.write(f"{name}={value}\n")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _asset_names(tag: str) -> list[str]:
    repository = _github_repository()
    release = _get_release(repository, tag)
    if release is None:
        raise SystemExit(f"GitHub release was not found: {tag}")

    try:
        return sorted(
            asset.name for asset in release.get_assets() if getattr(asset, "state", "uploaded") == "uploaded"
        )
    except GithubException as error:
        raise SystemExit(f"Failed to list GitHub release assets for {tag}: {error}") from error


def plan_apk_release_matrix_from_env() -> None:
    prebuild_tag = _env("PREBUILD_RELEASE_TAG", DEFAULT_PREBUILD_RELEASE_TAG)
    channel = _env("PREBUILD_CHANNEL", DEFAULT_PREBUILD_CHANNEL)
    common_asset = f"openvino-android-common-{channel}.zip"
    runtime_pattern = re.compile(rf"^openvino-android-runtime-(?P<abi>.+)-{re.escape(channel)}\.zip$")
    assets = _asset_names(prebuild_tag)

    if common_asset not in assets:
        raise SystemExit(f"Required common prebuild asset is missing from {prebuild_tag}: {common_asset}")

    matrix_entries: list[dict[str, str]] = []
    for asset_name in assets:
        match = runtime_pattern.match(asset_name)
        if match is None:
            continue
        abi = match.group("abi")
        matrix_entries.append(
            {
                "android_abi": abi,
                "common_asset": common_asset,
                "runtime_asset": asset_name,
                "prebuild_release_tag": prebuild_tag,
            }
        )

    if not matrix_entries:
        raise SystemExit(f"No Android runtime prebuild assets found in {prebuild_tag} for channel {channel}.")

    matrix = {"include": sorted(matrix_entries, key=lambda item: item["android_abi"])}
    matrix_json = json.dumps(matrix, sort_keys=True, separators=(",", ":"))
    _write_output("matrix", matrix_json)
    print(json.dumps(matrix, indent=2, sort_keys=True), flush=True)


def package_release_apk_from_env() -> None:
    abi = _env("ANDROID_ABI")
    if not abi:
        raise SystemExit("ANDROID_ABI must be set to package a release APK.")

    source_apk = Path(_env("RELEASE_APK_PATH", "app/build/outputs/apk/release/app-release.apk"))
    if not source_apk.is_file():
        raise SystemExit(f"Release APK was not found: {source_apk}")

    artifacts_dir_value = _env("ARTIFACTS_DIR")
    if not artifacts_dir_value:
        raise SystemExit("ARTIFACTS_DIR must be set to package a release APK.")
    artifacts_dir = Path(artifacts_dir_value)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    channel = _env("APK_CHANNEL", DEFAULT_PREBUILD_CHANNEL)
    apk_name = f"openvino-notes-{abi}-{channel}.apk"
    apk_path = artifacts_dir / apk_name
    shutil.copy2(source_apk, apk_path)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "android": {
            "abi": abi,
            "runtime_asset": _env("RUNTIME_PREBUILD_ASSET"),
        },
        "prebuilds": {
            "release_tag": _env("PREBUILD_RELEASE_TAG", DEFAULT_PREBUILD_RELEASE_TAG),
            "common_asset": _env("COMMON_PREBUILD_ASSET"),
            "runtime_asset": _env("RUNTIME_PREBUILD_ASSET"),
        },
        "source": {
            "repository": _env("GITHUB_REPOSITORY"),
            "workflow_run_id": _env("GITHUB_RUN_ID"),
            "sha": _env("GITHUB_SHA"),
            "ref": _env("GITHUB_REF_NAME"),
        },
        "asset": {
            "name": apk_name,
            "type": "apk",
            "abi": abi,
            "size_bytes": apk_path.stat().st_size,
            "sha256": _sha256(apk_path),
        },
    }
    manifest_path = artifacts_dir / f"{APK_PACKAGE_MANIFEST_PREFIX}-{abi}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{apk_path} {apk_path.stat().st_size} bytes sha256={manifest['asset']['sha256']}", flush=True)
    print(f"{manifest_path} {manifest_path.stat().st_size} bytes", flush=True)


def _load_apk_package_manifests(artifacts_dir: Path) -> list[dict[str, object]]:
    manifests = sorted(artifacts_dir.glob(f"{APK_PACKAGE_MANIFEST_PREFIX}-*.json"))
    if not manifests:
        raise SystemExit(f"No APK package manifests found in {artifacts_dir}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in manifests]


def _build_apk_release_manifest(artifacts_dir: Path) -> dict[str, object]:
    package_manifests = _load_apk_package_manifests(artifacts_dir)
    apk_assets: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    first_prebuilds = package_manifests[0].get("prebuilds", {})
    prebuilds = first_prebuilds if isinstance(first_prebuilds, dict) else {}

    for package_manifest in package_manifests:
        asset = package_manifest.get("asset")
        android = package_manifest.get("android")
        if not isinstance(asset, dict) or not isinstance(asset.get("name"), str):
            raise SystemExit("Invalid APK asset entry in package manifest.")
        if not isinstance(android, dict):
            raise SystemExit("Invalid Android target entry in package manifest.")

        apk_path = artifacts_dir / asset["name"]
        if not apk_path.is_file():
            raise SystemExit(f"APK package manifest references missing asset: {apk_path}")
        apk_assets.append(asset)
        targets.append(android)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prebuilds": {
            "release_tag": prebuilds.get("release_tag", ""),
            "common_asset": prebuilds.get("common_asset", ""),
        },
        "source": package_manifests[0].get("source", {}),
        "android_targets": sorted(targets, key=lambda item: str(item.get("abi", ""))),
        "assets": sorted(apk_assets, key=lambda item: str(item.get("name", ""))),
    }


def _format_size(size_bytes: object) -> str:
    if not isinstance(size_bytes, int):
        return ""
    return f"{size_bytes / (1024 * 1024):.1f} MiB"


def _apk_release_notes(prefix: str, release_manifest: dict[str, object]) -> str:
    source = release_manifest.get("source", {})
    prebuilds = release_manifest.get("prebuilds", {})
    targets = release_manifest.get("android_targets", [])
    assets = release_manifest.get("assets", [])

    lines = [prefix, ""]
    if isinstance(prebuilds, dict):
        lines.extend(
            [
                "Prebuild source:",
                f"- Release tag: {prebuilds.get('release_tag', '')}",
                f"- Common asset: {prebuilds.get('common_asset', '')}",
                "",
            ]
        )

    if isinstance(source, dict):
        lines.extend(
            [
                "App source:",
                f"- Repository: {source.get('repository', '')}",
                f"- Ref: {source.get('ref', '')}",
                f"- SHA: {str(source.get('sha', ''))[:12]}",
                "",
            ]
        )

    lines.append("Android targets:")
    if isinstance(targets, list):
        for target in targets:
            if isinstance(target, dict):
                lines.append(f"- {target.get('abi', '')}: {target.get('runtime_asset', '')}")

    lines.extend(["", "Assets:"])
    if isinstance(assets, list):
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            digest = str(asset.get("sha256", ""))[:12]
            lines.append(f"- {asset.get('name', '')}: {_format_size(asset.get('size_bytes'))}, sha256 {digest}")

    workflow_url = ""
    if os.environ.get("GITHUB_SERVER_URL") and os.environ.get("GITHUB_REPOSITORY") and os.environ.get("GITHUB_RUN_ID"):
        workflow_url = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    if workflow_url:
        lines.extend(["", f"Workflow run: {workflow_url}"])

    return "\n".join(lines) + "\n"


def publish_apk_release_from_env() -> None:
    artifacts_dir_value = _env("ARTIFACTS_DIR")
    if not artifacts_dir_value:
        raise SystemExit("ARTIFACTS_DIR must be set to publish APK release assets.")
    artifacts_dir = Path(artifacts_dir_value)

    release_manifest = _build_apk_release_manifest(artifacts_dir)
    release_manifest_path = artifacts_dir / APK_RELEASE_MANIFEST_NAME
    release_manifest_path.write_text(json.dumps(release_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    notes_prefix = _env("RELEASE_NOTES_PREFIX", DEFAULT_APK_RELEASE_NOTES_PREFIX)
    notes = _apk_release_notes(notes_prefix, release_manifest)
    (artifacts_dir / "apk-release-notes.md").write_text(notes, encoding="utf-8")

    tag = _env("RELEASE_TAG", DEFAULT_APK_RELEASE_TAG)
    title = _env("RELEASE_TITLE", DEFAULT_APK_RELEASE_TITLE)
    repository = _github_repository()
    release = _get_release(repository, tag)
    try:
        if release is None:
            print(f"Creating GitHub prerelease: {tag}", flush=True)
            release = repository.create_git_release(
                tag=tag,
                name=title,
                message=notes,
                prerelease=True,
                make_latest="false",
            )
        else:
            print(f"Updating GitHub prerelease: {tag}", flush=True)
            release = release.update_release(
                name=title,
                message=notes,
                draft=False,
                prerelease=True,
                tag_name=tag,
                make_latest="false",
            )
    except GithubException as error:
        raise SystemExit(f"Failed to create or update GitHub release {tag}: {error}") from error

    apks = sorted(artifacts_dir.glob("*.apk"))
    if not apks:
        raise SystemExit(f"No APK assets found in {artifacts_dir}")
    _sync_release_assets(release, [*apks, release_manifest_path])
