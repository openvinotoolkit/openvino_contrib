from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from git import RemoteProgress, Repo
from git.exc import GitError

from .common import BuildConfig, run


class CloneProgress(RemoteProgress):
    def update(
        self,
        op_code: int,
        cur_count: str | float,
        max_count: str | float | None = None,
        message: str = "",
    ) -> None:
        if message:
            print(f"git: {message}", flush=True)


def clone_ref(repo_url: str, ref: str, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)

    print(f"+ GitPython clone {repo_url} {destination} --branch {ref}", flush=True)
    try:
        Repo.clone_from(
            repo_url,
            destination,
            branch=ref,
            depth=1,
            recursive=True,
            progress=CloneProgress(),
            multi_options=["--shallow-submodules", "--jobs=4"],
        )
    except GitError as error:
        raise SystemExit(f"Failed to clone {repo_url}@{ref}: {error}") from error


def commit_hash(repository: Path) -> str:
    try:
        return Repo(repository).head.commit.hexsha
    except GitError as error:
        raise SystemExit(f"Failed to read Git commit from {repository}: {error}") from error


def checkout_sources(config: BuildConfig) -> None:
    config.src_dir.mkdir(parents=True, exist_ok=True)
    clone_ref(config.openvino_repo, config.openvino_ref, config.src_dir / "openvino")
    clone_ref(config.openvino_contrib_repo, config.openvino_contrib_ref, config.src_dir / "openvino_contrib")
    clone_ref(config.openvino_genai_repo, config.openvino_genai_ref, config.src_dir / "openvino.genai")
    clone_ref(config.genai_java_api_repo, config.genai_java_api_ref, config.src_dir / "genai-java-api")
    clone_ref(config.onetbb_repo, config.onetbb_ref, config.src_dir / "oneTBB")


def record_source_manifest(config: BuildConfig) -> None:
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "OpenVINO Android prebuild source manifest",
        f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        f"openvino_ref={config.openvino_ref}",
        f"openvino_commit={commit_hash(config.src_dir / 'openvino')}",
        f"openvino_genai_ref={config.openvino_genai_ref}",
        f"openvino_genai_commit={commit_hash(config.src_dir / 'openvino.genai')}",
        f"genai_java_api_ref={config.genai_java_api_ref}",
        f"genai_java_api_commit={commit_hash(config.src_dir / 'genai-java-api')}",
        f"openvino_contrib_ref={config.openvino_contrib_ref}",
        f"openvino_contrib_commit={commit_hash(config.src_dir / 'openvino_contrib')}",
        f"onetbb_ref={config.onetbb_ref}",
        f"onetbb_commit={commit_hash(config.src_dir / 'oneTBB')}",
        "",
        f"android_abi={config.android_abi}",
        f"android_platform={config.android_platform}",
        f"android_ndk_version={config.android_ndk_version}",
    ]
    (config.artifacts_dir / "source-manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if shutil.which("ccache"):
        run(["ccache", "--zero-stats"])
