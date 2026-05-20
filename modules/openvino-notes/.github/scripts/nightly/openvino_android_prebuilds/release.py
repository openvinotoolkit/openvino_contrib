from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from github import Auth, Github, GithubException, UnknownObjectException
from github.GitRelease import GitRelease
from github.Repository import Repository


DEFAULT_RELEASE_TAG = "openvino-android-prebuilds-nightly"
DEFAULT_RELEASE_TITLE = "OpenVINO Android Prebuilds Nightly"
DEFAULT_RELEASE_NOTES_PREFIX = "Rolling nightly Android OpenVINO prebuilds for 64-bit ARM and x86 targets."
RELEASE_MANIFEST_NAME = "openvino-android-prebuilds-manifest.json"
SOURCE_IDENTITY_KEYS = (
    "openvino_ref",
    "openvino_commit",
    "openvino_genai_ref",
    "openvino_genai_commit",
    "genai_java_api_ref",
    "genai_java_api_commit",
    "openvino_contrib_ref",
    "openvino_contrib_commit",
    "onetbb_ref",
    "onetbb_commit",
)


def _github_repository() -> Repository:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    repository_name = os.environ.get("GITHUB_REPOSITORY")
    if not token:
        raise SystemExit("GH_TOKEN or GITHUB_TOKEN must be set to publish a release.")
    if not repository_name:
        raise SystemExit("GITHUB_REPOSITORY must be set to publish a release.")

    try:
        return Github(auth=Auth.Token(token)).get_repo(repository_name)
    except GithubException as error:
        raise SystemExit(f"Failed to open GitHub repository {repository_name}: {error}") from error


def _get_release(repository: Repository, tag: str) -> GitRelease | None:
    try:
        return repository.get_release(tag)
    except UnknownObjectException:
        return None
    except GithubException as error:
        raise SystemExit(f"Failed to read GitHub release {tag}: {error}") from error


def _load_package_manifests(artifacts_dir: Path) -> list[dict[str, object]]:
    manifest_paths = sorted(artifacts_dir.glob("openvino-android-package-manifest-*.json"))
    if not manifest_paths:
        raise SystemExit(f"No package manifests found in {artifacts_dir}")

    manifests: list[dict[str, object]] = []
    for manifest_path in manifest_paths:
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    return manifests


def _build_release_manifest(artifacts_dir: Path) -> dict[str, object]:
    package_manifests = _load_package_manifests(artifacts_dir)
    source = package_manifests[0].get("source", {})
    if not isinstance(source, dict):
        raise SystemExit("Package manifest has invalid source metadata.")
    source_identity = {key: source.get(key) for key in SOURCE_IDENTITY_KEYS}
    targets: list[dict[str, object]] = []
    assets_by_name: dict[str, dict[str, object]] = {}

    for package_manifest in package_manifests:
        current_source = package_manifest.get("source", {})
        if not isinstance(current_source, dict):
            raise SystemExit("Package manifest has invalid source metadata.")
        current_identity = {key: current_source.get(key) for key in SOURCE_IDENTITY_KEYS}
        if current_identity != source_identity:
            raise SystemExit("Package manifests were built from different source revisions.")

        android = package_manifest.get("android", {})
        if isinstance(android, dict):
            targets.append(android)

        for asset in package_manifest.get("assets", []):
            if not isinstance(asset, dict) or not isinstance(asset.get("name"), str):
                raise SystemExit("Invalid asset entry in package manifest.")
            asset_path = artifacts_dir / asset["name"]
            if not asset_path.is_file():
                raise SystemExit(f"Package manifest references missing asset: {asset_path}")
            assets_by_name[asset["name"]] = asset

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "android_targets": sorted(targets, key=lambda item: str(item.get("abi", ""))),
        "assets": [assets_by_name[name] for name in sorted(assets_by_name)],
    }


def _workflow_url() -> str:
    if os.environ.get("GITHUB_SERVER_URL") and os.environ.get("GITHUB_REPOSITORY") and os.environ.get("GITHUB_RUN_ID"):
        return f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    return ""


def _format_size(size_bytes: object) -> str:
    if not isinstance(size_bytes, int):
        return ""
    mib = size_bytes / (1024 * 1024)
    return f"{mib:.1f} MiB"


def _release_notes(prefix: str, release_manifest: dict[str, object]) -> str:
    source = release_manifest.get("source", {})
    targets = release_manifest.get("android_targets", [])
    assets = release_manifest.get("assets", [])

    lines = [prefix, ""]
    if isinstance(source, dict):
        lines.extend(
            [
                "Source refs:",
                f"- OpenVINO: {source.get('openvino_ref', '')} ({str(source.get('openvino_commit', ''))[:12]})",
                f"- OpenVINO GenAI: {source.get('openvino_genai_ref', '')} ({str(source.get('openvino_genai_commit', ''))[:12]})",
                f"- OpenVINO GenAI Java API: {source.get('genai_java_api_ref', '')} ({str(source.get('genai_java_api_commit', ''))[:12]})",
                f"- OpenVINO Contrib: {source.get('openvino_contrib_ref', '')} ({str(source.get('openvino_contrib_commit', ''))[:12]})",
                f"- oneTBB: {source.get('onetbb_ref', '')} ({str(source.get('onetbb_commit', ''))[:12]})",
                "",
            ]
        )

    lines.append("Android targets:")
    target_items = targets if isinstance(targets, list) else []
    for target in target_items:
        if not isinstance(target, dict):
            continue
        lines.append(
            f"- {target.get('abi', '')}: platform {target.get('platform', '')}, NDK {target.get('ndk_version', '')}"
        )

    lines.extend(["", "Assets:"])
    asset_items = assets if isinstance(assets, list) else []
    for asset in asset_items:
        if not isinstance(asset, dict):
            continue
        abi = asset.get("abi") or "all"
        digest = str(asset.get("sha256", ""))[:12]
        lines.append(f"- {asset.get('name', '')}: {asset.get('type', '')}, {abi}, {_format_size(asset.get('size_bytes'))}, sha256 {digest}")

    workflow_url = _workflow_url()
    if workflow_url:
        lines.extend(["", f"Workflow run: {workflow_url}"])
    return "\n".join(lines) + "\n"


def _content_type(path: Path) -> str:
    if path.suffix == ".apk":
        return "application/vnd.android.package-archive"
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".zip":
        return "application/zip"
    return "application/octet-stream"


def _sync_release_assets(release: GitRelease, assets: list[Path]) -> None:
    try:
        expected_names = {asset.name for asset in assets}
        existing_assets = {asset.name: asset for asset in release.get_assets()}

        for name, asset in existing_assets.items():
            if name not in expected_names:
                print(f"Deleting stale release asset: {name}", flush=True)
                asset.delete_asset()

        for asset_path in assets:
            existing = existing_assets.get(asset_path.name)
            if existing is not None:
                print(f"Deleting existing release asset: {existing.name}", flush=True)
                existing.delete_asset()
            print(f"Uploading release asset: {asset_path}", flush=True)
            release.upload_asset(str(asset_path), name=asset_path.name, content_type=_content_type(asset_path))
    except GithubException as error:
        raise SystemExit(f"Failed to sync GitHub release assets: {error}") from error


def publish_rolling_prerelease(
    *,
    tag: str,
    title: str,
    artifacts_dir: Path,
    notes_prefix: str,
) -> None:
    prebuilds = sorted(artifacts_dir.glob("*.zip"))
    if not prebuilds:
        raise SystemExit(f"No prebuild zip artifacts found in {artifacts_dir}")

    release_manifest = _build_release_manifest(artifacts_dir)
    release_manifest_path = artifacts_dir / RELEASE_MANIFEST_NAME
    release_manifest_path.write_text(json.dumps(release_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    notes_file = artifacts_dir / "release-notes.md"
    notes = _release_notes(notes_prefix, release_manifest)
    notes_file.write_text(notes, encoding="utf-8")

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

    _sync_release_assets(release, [*prebuilds, release_manifest_path])


def publish_release_from_env() -> None:
    artifacts_dir = os.environ.get("ARTIFACTS_DIR", "")
    if not artifacts_dir:
        raise SystemExit("ARTIFACTS_DIR must be set for publish-release stage.")

    publish_rolling_prerelease(
        tag=os.environ.get("RELEASE_TAG", DEFAULT_RELEASE_TAG),
        title=os.environ.get("RELEASE_TITLE", DEFAULT_RELEASE_TITLE),
        artifacts_dir=Path(artifacts_dir),
        notes_prefix=os.environ.get("RELEASE_NOTES_PREFIX", DEFAULT_RELEASE_NOTES_PREFIX),
    )
