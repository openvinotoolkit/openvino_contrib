# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import configparser
import subprocess
from pathlib import Path

from ov_contrib_license.model import (
    Component,
    Confidence,
    Decision,
    Discovery,
    DistributionStatus,
    Evidence,
    EvidenceKind,
    Finding,
    InventoryBuilder,
    Relationship,
    Severity,
)
from ov_contrib_license.model.types import normalize_details

from .common import owning_module


def _index_revision(root: Path, path: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--stage", "--", path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return None
    if result.returncode or not result.stdout.strip():
        return None
    fields = result.stdout.split()
    return fields[1] if len(fields) >= 2 and fields[0] == "160000" else None


def discover_submodules(
    builder: InventoryBuilder, root: Path, files: tuple[str, ...]
) -> None:
    if ".gitmodules" not in files:
        return
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(root / ".gitmodules", encoding="utf-8")
    except (configparser.Error, OSError):
        return

    for section in sorted(parser.sections()):
        if not section.startswith("submodule "):
            continue
        path = parser.get(section, "path", fallback="").strip()
        url = parser.get(section, "url", fallback="").strip()
        if not path or not url:
            continue
        revision = _index_revision(root, path)
        component_id = f"vcs:{url}" + (f"@{revision}" if revision else "")
        evidence = Evidence(
            kind=EvidenceKind.GIT_SUBMODULE,
            source="gitmodules-discovery",
            path=".gitmodules",
            value=url,
            confidence=Confidence.HIGH if revision else Confidence.MEDIUM,
            details=normalize_details({"path": path, "revision": revision}),
        )
        builder.add_discovery(
            Discovery(
                kind="git-submodule",
                path=".gitmodules",
                module=owning_module(path),
                ecosystem="git",
                details=normalize_details(
                    {"submodule_path": path, "url": url, "revision": revision}
                ),
            )
        )
        builder.add_component(
            Component(
                id=component_id,
                name=section.removeprefix('submodule "').removesuffix('"'),
                version=revision,
                module=owning_module(path),
                paths=(path,),
                relationships=(Relationship.VENDORED_SOURCE,),
                evidence=(evidence,),
                distribution=DistributionStatus.DISTRIBUTED,
                details=normalize_details({"source_url": url}),
            )
        )
        if revision is None:
            builder.add_finding(
                Finding.create(
                    code="DISCOVERY_SUBMODULE_UNPINNED",
                    severity=Severity.WARNING,
                    decision=Decision.REVIEW,
                    component_id=component_id,
                    message=f"Git submodule {path!r} has no pinned commit in the Git index.",
                    evidence=(evidence,),
                    remediation=("Commit the submodule gitlink at an exact revision.",),
                )
            )
