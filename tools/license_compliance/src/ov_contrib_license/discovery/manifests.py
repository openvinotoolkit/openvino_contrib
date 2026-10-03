# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath

from ov_contrib_license.model import Discovery, InventoryBuilder

from .common import owning_module

MANIFEST_PATTERNS: tuple[tuple[str, str], ...] = (
    ("requirements*.txt", "python"),
    ("setup.py", "python"),
    ("setup.cfg", "python"),
    ("pyproject.toml", "python"),
    ("Pipfile*", "python"),
    ("poetry.lock", "python"),
    ("package.json", "node"),
    ("package-lock.json", "node"),
    ("yarn.lock", "node"),
    ("pnpm-lock.yaml", "node"),
    ("go.mod", "go"),
    ("go.sum", "go"),
    ("build.gradle*", "gradle"),
    ("settings.gradle*", "gradle"),
    ("Cargo.toml", "cargo"),
    ("Cargo.lock", "cargo"),
    ("conanfile.*", "conan"),
    ("Dockerfile*", "docker"),
    (".gitmodules", "git"),
)


def manifest_ecosystem(path: str) -> str | None:
    name = PurePosixPath(path).name
    for pattern, ecosystem in MANIFEST_PATTERNS:
        if fnmatch.fnmatchcase(name, pattern):
            return ecosystem
    return None


def discover_manifests(builder: InventoryBuilder, files: tuple[str, ...]) -> None:
    for path in files:
        ecosystem = manifest_ecosystem(path)
        if ecosystem is None:
            continue
        builder.add_discovery(
            Discovery(
                kind="manifest",
                path=path,
                module=owning_module(path),
                ecosystem=ecosystem,
            )
        )
