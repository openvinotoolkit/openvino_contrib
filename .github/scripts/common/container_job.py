# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run isolated module-test containers with one shared security policy."""

import os
import re
from pathlib import Path

from .artifact import extract_archive
from .ci_workspace import prepare_tests
from .core import (
    TEST_CONTAINER_PATTERN,
    empty_proxy_environment,
    relative_path,
    required_env,
    run,
    safe_name,
)

DEVICE_PATTERN = re.compile(r"[A-Za-z0-9_./:=,+-]+")


def _image(value: str, description: str) -> str:
    if not value or len(value) > 512 or value.startswith("-") or any(char.isspace() for char in value):
        raise ValueError(f"invalid {description} image reference: {value}")
    return value


def pull_images(images: list[str]) -> None:
    for image in images:
        run(["docker", "pull", _image(image, "container")])


def _mounts(checkout: Path, job_dir: Path) -> tuple[Path, Path]:
    expected_checkout = (Path(required_env("GITHUB_WORKSPACE")) / "openvino_contrib").resolve()
    checkout = checkout.resolve()
    job_dir = job_dir.resolve()
    temp = Path(required_env("RUNNER_TEMP")).resolve()
    if checkout != expected_checkout:
        raise ValueError(f"refusing to mount unexpected checkout: {checkout}")
    if job_dir.parent != temp or not job_dir.name.startswith("contrib-ci-tests-") or not job_dir.is_dir():
        raise ValueError(f"refusing to mount unexpected test job directory: {job_dir}")
    for path in (checkout, job_dir):
        if any(character in str(path) for character in (":", "\n", "\r")):
            raise ValueError(f"path cannot be mounted safely: {path}")
    return checkout, job_dir


def _runner(checkout: Path, script: str) -> Path:
    runner = relative_path(script, ".github")
    host_path = (checkout / runner).resolve()
    if checkout not in host_path.parents or not host_path.is_file():
        raise ValueError(f"module test runner is unavailable: {runner}")
    return runner


def _test_command(
    image: str,
    name: str,
    checkout: Path,
    job_dir: Path,
    runner: Path,
    devices: list[str],
    preset: str,
) -> list[str]:
    arguments = [
        "docker",
        "run",
        f"--name={name}",
        "--rm",
        "--runtime=runc",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=4096",
        "--tmpfs=/tmp:rw,nosuid,nodev,size=2g",
        f"--user={os.getuid()}:{os.getgid()}",
        "--env=HOME=/tmp",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        *empty_proxy_environment(),
    ]
    for device in devices:
        arguments.append(f"--device={device}")
    if devices:
        arguments.append("--shm-size=2g")
    return [
        *arguments,
        f"--env=CI_PRESET={preset}",
        f"--volume={checkout}:/ci:ro",
        f"--volume={job_dir}:/job:rw",
        "--workdir=/job",
        "--entrypoint=/usr/local/bin/python",
        image,
        f"/ci/{runner.as_posix()}",
    ]


def _validated_devices(devices: object) -> list[str]:
    if not isinstance(devices, list) or len(devices) > 16:
        raise ValueError("at most 16 device selectors are allowed")
    for device in devices:
        valid = (
            isinstance(device, str)
            and 0 < len(device) <= 200
            and bool(DEVICE_PATTERN.fullmatch(device))
            and not device.startswith("-")
        )
        if not valid:
            raise ValueError(f"invalid container device selector: {device}")
    return devices


def _device_selectors(value: str) -> list[str]:
    return _validated_devices([line.strip() for line in value.splitlines() if line.strip()])


def _runtime_image(python_image: str, test_image: str, container_name: str) -> str:
    tag = f"contrib-ci-runtime:{container_name.removeprefix('contrib-ci-test-')}"
    if len(tag.rsplit(":", 1)[1]) > 128:
        raise ValueError(f"generated test image tag is too long: {tag}")
    dockerfile = """\
ARG PYTHON_IMAGE=scratch
ARG TEST_IMAGE=scratch
FROM ${PYTHON_IMAGE} AS python
FROM ${TEST_IMAGE}
COPY --from=python /usr/local /usr/local
"""
    run(
        [
            "docker",
            "build",
            "--network=none",
            "--pull=false",
            f"--tag={tag}",
            f"--build-arg=PYTHON_IMAGE={_image(python_image, 'Python runtime')}",
            f"--build-arg=TEST_IMAGE={_image(test_image, 'test')}",
            "-",
        ],
        input_text=dockerfile,
    )
    return tag


def run_tests(
    python_image: str,
    image: str,
    devices: list[str],
    container_name: str,
    checkout: Path,
    job_dir: Path,
    test_script: str,
    preset: str,
) -> None:
    safe_name(container_name, "test container", TEST_CONTAINER_PATTERN)
    devices = _validated_devices(devices)
    checkout, job_dir = _mounts(checkout, job_dir)
    runner = _runner(checkout, test_script)
    runtime_image = _runtime_image(python_image, image, container_name)
    try:
        run(_test_command(runtime_image, container_name, checkout, job_dir, runner, devices, preset))
    finally:
        run(["docker", "image", "rm", "--force", runtime_image], check=False, quiet=True)


def run_tests_command() -> None:
    """Execute the module-owned test policy in a locked-down container."""
    workspace = Path(required_env("GITHUB_WORKSPACE"))
    run_tests(
        required_env("PYTHON_IMAGE"),
        required_env("TEST_IMAGE"),
        _device_selectors(os.environ.get("CI_DEVICES", "")),
        required_env("TEST_CONTAINER"),
        workspace / "openvino_contrib",
        Path(required_env("JOB_DIR")),
        required_env("CI_TEST_SCRIPT"),
        required_env("CI_PRESET"),
    )


def main() -> None:
    """Download and run one module artifact in the restricted container."""
    from .download_artifact import download

    job_dir = prepare_tests()
    pull_images([required_env("PYTHON_IMAGE"), required_env("TEST_IMAGE")])
    name = safe_name(f"{required_env('CI_MODULE_NAME')}_build", "artifact name")
    download(name, job_dir)
    extract_archive(job_dir / f"{name}.tar.gz", job_dir / "artifact")
    run_tests_command()
