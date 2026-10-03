# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

from ov_contrib_license.model import (
    Component,
    Confidence,
    Discovery,
    DistributionStatus,
    Evidence,
    EvidenceKind,
    InventoryBuilder,
    Relationship,
)
from ov_contrib_license.model.types import normalize_details

from .common import owning_module

VENDORED_DIRECTORY_NAMES = frozenset(
    {
        "third_party",
        "thirdparty",
        "3rdparty",
        "vendor",
        "vendors",
        "external",
        "extern",
        "deps",
    }
)
LICENSE_FILE_NAMES = frozenset({"copying", "copyright", "license", "notice"})
SOURCE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cl",
        ".cpp",
        ".cxx",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".kt",
        ".py",
        ".ts",
    }
)
COPYRIGHT_PATTERN = re.compile(r"copyright[^\n\r]{0,200}", re.IGNORECASE)


def _vendored_root(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    for index, part in enumerate(parts[:-1]):
        if part.lower() in VENDORED_DIRECTORY_NAMES:
            return PurePosixPath(*parts[: index + 1]).as_posix()
    return None


def _content_root(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    if len(parts) >= 4 and parts[0] == "modules":
        return PurePosixPath(*parts[:3]).as_posix()
    return None


def _license_file(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    stem = name.split(".", 1)[0]
    return stem in LICENSE_FILE_NAMES


def _has_foreign_copyright(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:8192]
    except OSError:
        return False
    matches = COPYRIGHT_PATTERN.findall(text)
    return any("intel corporation" not in match.lower() for match in matches)


def _add_vendored_component(
    builder: InventoryBuilder,
    root: str,
    *,
    heuristic: str,
    confidence: Confidence,
    signal_count: int | None = None,
) -> None:
    name = PurePosixPath(root).name
    details = normalize_details({"heuristic": heuristic, "signal_count": signal_count})
    evidence_kind = (
        EvidenceKind.LICENSE_FILE
        if heuristic == "nested-license-file"
        else EvidenceKind.VENDORED_TREE
    )
    evidence = Evidence(
        kind=evidence_kind,
        source="vendored-content-heuristic"
        if heuristic != "directory-name"
        else "vendored-directory-heuristic",
        path=root,
        value=name,
        confidence=confidence,
        details=details,
    )
    builder.add_discovery(
        Discovery(
            kind="vendored-tree",
            path=root,
            module=owning_module(root),
            details=details,
        )
    )
    builder.add_component(
        Component(
            id=f"local:{root}",
            name=name,
            module=owning_module(root),
            paths=(root,),
            relationships=(Relationship.VENDORED_SOURCE,),
            evidence=(evidence,),
            distribution=DistributionStatus.DISTRIBUTED,
        )
    )


def discover_vendored_trees(
    builder: InventoryBuilder, repository_root: Path, files: tuple[str, ...]
) -> None:
    explicit_roots = sorted(filter(None, {_vendored_root(path) for path in files}))
    for root in explicit_roots:
        assert root is not None
        _add_vendored_component(
            builder,
            root,
            heuristic="directory-name",
            confidence=Confidence.LOW,
        )

    nested_license_roots = sorted(
        {
            content_root
            for path in files
            if _license_file(path)
            for content_root in (_content_root(path),)
            if content_root and content_root not in explicit_roots
        }
    )
    for root in nested_license_roots:
        _add_vendored_component(
            builder,
            root,
            heuristic="nested-license-file",
            confidence=Confidence.MEDIUM,
        )

    copyright_signals: dict[str, list[str]] = defaultdict(list)
    for path in files:
        content_root = _content_root(path)
        if (
            content_root
            and PurePosixPath(path).suffix.lower() in SOURCE_SUFFIXES
            and _has_foreign_copyright(repository_root / path)
        ):
            copyright_signals[content_root].append(path)
    for root, signals in sorted(copyright_signals.items()):
        if len(signals) < 3 or root in explicit_roots or root in nested_license_roots:
            continue
        _add_vendored_component(
            builder,
            root,
            heuristic="foreign-copyright-cluster",
            confidence=Confidence.MEDIUM,
            signal_count=len(signals),
        )
