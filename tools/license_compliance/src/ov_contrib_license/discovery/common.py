# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import PurePosixPath


def owning_scope(path: str) -> str:
    parts = PurePosixPath(path).parts
    if len(parts) >= 2 and parts[0] == "modules":
        return f"module:{parts[1]}"
    if parts and parts[0] == ".github":
        return "repository-ci"
    if parts and parts[0] == "tools":
        return "repository-tooling"
    return "repository"


def owning_module(path: str) -> str | None:
    scope = owning_scope(path)
    return scope.removeprefix("module:") if scope.startswith("module:") else scope
