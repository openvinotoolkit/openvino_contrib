#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prepare and validate isolated CI work directories."""

import os
import shutil
import stat
import tempfile
from pathlib import Path

from .core import (
    TEST_CONTAINER_PATTERN,
    required_env,
    run,
    safe_name,
    write_github_file,
)

MAX_RESULTS_BYTES = 100 * 1024 * 1024
MAX_RESULTS_ENTRIES = 10_000


def _new_directory(prefix: str) -> Path:
    runner_temp = Path(required_env("RUNNER_TEMP")).absolute()
    module = safe_name(required_env("CI_MODULE_NAME"), "module name")
    return Path(tempfile.mkdtemp(prefix=f"{prefix}-{module}-", dir=runner_temp))


def _export(values: dict[str, object]) -> None:
    os.environ.update({key: str(value) for key, value in values.items()})
    write_github_file("GITHUB_ENV", values)


def prepare_build() -> Path:
    module = safe_name(required_env("CI_MODULE_NAME"), "module name")
    artifact = f"{module}_build"
    work_dir = _new_directory("contrib-ci-build")
    _export(
        {
            "WORK_DIR": work_dir,
            "OV_INSTALL_DIR": work_dir / "ov_install",
            "MODULE_BUILD_DIR": work_dir / "module_build",
            "STAGING_DIR": work_dir / "artifact",
            "ARCHIVE_PATH": work_dir / f"{artifact}.tar.gz",
            "SCCACHE_DIR": work_dir / "sccache",
            "SCCACHE_CACHE_SIZE": "2G",
        },
    )
    return work_dir


def prepare_tests() -> Path:
    preset = required_env("CI_PRESET")
    module = safe_name(required_env("CI_MODULE_NAME"), "module name")
    repository_id = safe_name(required_env("GITHUB_REPOSITORY_ID"), "repository id")
    job_dir = _new_directory("contrib-ci-tests")
    results_dir = job_dir / "test-results"
    results_dir.mkdir()
    suffix = "-".join(
        (
            repository_id,
            required_env("GITHUB_RUN_ID"),
            required_env("GITHUB_RUN_ATTEMPT"),
            module,
            preset,
        )
    )
    _export(
        {
            "JOB_DIR": job_dir,
            "RESULTS_DIR": results_dir,
            "TEST_CONTAINER": f"contrib-ci-test-{suffix}",
        },
    )
    return job_dir


def validate_results(
    results_dir: Path,
    max_bytes: int = MAX_RESULTS_BYTES,
    max_entries: int = MAX_RESULTS_ENTRIES,
) -> int:
    if results_dir.is_symlink() or not results_dir.is_dir():
        raise ValueError(f"test results directory is invalid: {results_dir}")
    total = entries = 0
    for root, directories, files in os.walk(results_dir, followlinks=False):
        entries += len(directories) + len(files)
        if entries > max_entries:
            raise ValueError(f"test results contain more than {max_entries} entries")
        for name in directories:
            path = Path(root, name)
            if not stat.S_ISDIR(path.stat(follow_symlinks=False).st_mode):
                raise ValueError(f"test result is not a regular file or directory: {path}")
        for name in files:
            path = Path(root, name)
            metadata = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"test result is not a regular file or directory: {path}")
            total += metadata.st_size
            if total > max_bytes:
                raise ValueError(f"test results exceed {max_bytes} bytes")
    return total


def cleanup_workspace() -> None:
    """Remove namespaced job state without accepting arbitrary paths or names."""
    runner_temp = Path(required_env("RUNNER_TEMP")).absolute()
    raw_work_dir = os.environ.get("WORK_DIR") or os.environ.get("JOB_DIR")
    if not raw_work_dir:
        return
    work_dir = Path(raw_work_dir).absolute()
    if work_dir.parent != runner_temp or not work_dir.name.startswith(
        ("contrib-ci-build-", "contrib-ci-tests-")
    ):
        raise ValueError(f"refusing to clean unexpected path: {work_dir}")

    test_container = os.environ.get("TEST_CONTAINER", "")
    if test_container:
        if not TEST_CONTAINER_PATTERN.fullmatch(test_container):
            raise ValueError(f"refusing to remove unexpected container name: {test_container}")
        run(["docker", "rm", "--force", test_container], check=False, quiet=True)
        tag = test_container.removeprefix("contrib-ci-test-")
        run(["docker", "image", "rm", "--force", f"contrib-ci-runtime:{tag}"], check=False, quiet=True)

    if work_dir.is_symlink():
        work_dir.unlink()
    elif work_dir.exists():
        shutil.rmtree(work_dir)


def validate_results_command() -> None:
    """Validate that publishable results are bounded regular files."""
    size = validate_results(Path(required_env("RESULTS_DIR")))
    print(f"Validated {size} bytes of test results")
