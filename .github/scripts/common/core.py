# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Small shared primitives for CI commands."""

import os
import re
import shlex
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
PATH_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
TEST_CONTAINER_PATTERN = re.compile(r"contrib-ci-test-[A-Za-z0-9_.-]+")
GITHUB_API_TIMEOUT = (15, 60)
_PROXY_NAMES = ("HTTP", "HTTPS", "ALL", "FTP", "NO")
PROXY_VARIABLES = tuple(f"{name}_PROXY" for name in _PROXY_NAMES) + tuple(
    f"{name.lower()}_proxy" for name in _PROXY_NAMES
)


def run(
    command: Sequence[str | Path],
    *,
    check: bool = True,
    capture: bool = False,
    quiet: bool = False,
    input_text: str | None = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command without a shell and print its safely quoted form."""
    arguments = [str(part) for part in command]
    if not quiet:
        print("+ " + shlex.join(arguments), flush=True)
    return subprocess.run(
        arguments,
        check=check,
        capture_output=capture,
        stdout=subprocess.DEVNULL if quiet and not capture else None,
        stderr=subprocess.DEVNULL if quiet and not capture else None,
        input=input_text,
        text=True,
        cwd=cwd,
        env=dict(env) if env is not None else None,
    )


def required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"required environment variable is empty: {name}")
    return value


def env_lines(name: str, *, required: bool = False) -> list[str]:
    values = [line.strip() for line in os.environ.get(name, "").splitlines() if line.strip()]
    if required and not values:
        raise ValueError(f"{name} must not be empty")
    return values


def github_api_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def safe_name(value: str, description: str, pattern: re.Pattern[str] = NAME_PATTERN) -> str:
    if len(value) > 100 or not pattern.fullmatch(value):
        raise ValueError(f"invalid {description}: {value}")
    return value


def relative_path(value: str, root: str | None = None) -> Path:
    path = Path(value)
    invalid = (
        path.is_absolute()
        or not value
        or value != path.as_posix()
        or any(part in ("", ".", "..") for part in path.parts)
        or (root is not None and path.parts[:1] != (root,))
        or len(value) > 512
        or any(len(part) > 100 or not PATH_COMPONENT_PATTERN.fullmatch(part) for part in path.parts)
    )
    if invalid:
        suffix = f" under {root}/" if root else ""
        raise ValueError(f"path must be normalized and relative{suffix}: {value}")
    return path


def write_github_values(path: Path, values: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            if isinstance(value, bool):
                rendered = str(value).lower()
            else:
                rendered = str(value)
            if "\n" in rendered or "\r" in rendered:
                raise ValueError(f"refusing multiline value for {key}")
            output.write(f"{key}={rendered}\n")


def write_github_file(variable: str, values: Mapping[str, object]) -> None:
    write_github_values(Path(required_env(variable)), values)


def empty_proxy_environment() -> Iterable[str]:
    for variable in PROXY_VARIABLES:
        yield f"--env={variable}="
