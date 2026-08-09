# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Check changed C/C++ lines with the standard git-clang-format driver."""

import os
import shutil
from pathlib import Path

from .core import NAME_PATTERN, env_lines, required_env, run, safe_name

EXTENSIONS = "c,cc,cpp,cxx,cu,cuh,h,hh,hpp,hxx"


def _tools() -> tuple[str, str]:
    formatter_name = safe_name(required_env("CI_CLANG_FORMAT"), "formatter", NAME_PATTERN)
    driver = shutil.which(f"git-{formatter_name}")
    formatter = shutil.which(formatter_name)
    if not driver or not formatter:
        run(["apt-get", "update"])
        run(["apt-get", "install", "--assume-yes", "--no-install-recommends", formatter_name])
        driver = shutil.which(f"git-{formatter_name}")
        formatter = shutil.which(formatter_name)
    if not driver or not formatter:
        raise RuntimeError(f"git-{formatter_name} and {formatter_name} were not found in PATH")
    return driver, formatter


def main() -> None:
    paths = env_lines("CI_MODULE_PATHS", required=True)
    workspace = Path(os.environ.get("GITHUB_WORKSPACE", Path.cwd())).resolve()
    relative_paths = []
    for value in paths:
        path = Path(value).resolve()
        if workspace not in path.parents:
            raise ValueError(f"module path is outside the workspace: {path}")
        relative_paths.append(path.relative_to(workspace))

    driver, formatter = _tools()
    environment = {
        **os.environ,
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": str(workspace),
    }
    result = run(
        [
            driver,
            "--binary",
            formatter,
            "--extensions",
            EXTENSIONS,
            "--style=file",
            "--diff",
            required_env("CI_BASE_SHA"),
            required_env("CI_HEAD_SHA"),
            "--",
            *relative_paths,
        ],
        capture=True,
        check=False,
        cwd=workspace,
        env=environment,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git-clang-format failed")
    if result.stdout.strip():
        print(result.stdout)
        raise RuntimeError("changed lines are not clang-formatted")
    print(f"✓ clang-format clean under {', '.join(map(str, relative_paths))}")
