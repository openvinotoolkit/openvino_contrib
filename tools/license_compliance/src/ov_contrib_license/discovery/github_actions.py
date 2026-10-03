# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote

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

USES_PATTERN = re.compile(r"^\s*(?:-\s*)?uses\s*:\s*([^\s#]+)", re.IGNORECASE)
FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


def extract_action_references(text: str) -> tuple[tuple[str, int], ...]:
    references: list[tuple[str, int]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        match = USES_PATTERN.match(line)
        if match:
            references.append((match.group(1).strip("'\""), line_number))
    return tuple(references)


def _is_workflow(path: str) -> bool:
    pure = PurePosixPath(path)
    return (
        len(pure.parts) >= 3
        and pure.parts[:2] == (".github", "workflows")
        and pure.suffix.lower() in {".yml", ".yaml"}
    )


def _action_component(
    reference: str, path: str, line: int
) -> tuple[Component, Finding | None] | None:
    if reference.startswith("./"):
        return None
    details = {"line": line, "reference": reference}
    if reference.startswith("docker://"):
        image = reference.removeprefix("docker://")
        name, separator, digest = image.partition("@")
        component_id = f"pkg:docker/{quote(name, safe='/.-_')}"
        if separator:
            component_id += f"@{quote(digest, safe=':.-_')}"
        evidence = Evidence(
            kind=EvidenceKind.GITHUB_ACTION,
            source="github-workflow-discovery",
            path=path,
            value=reference,
            confidence=Confidence.HIGH,
            details=normalize_details(details),
        )
        component = Component(
            id=component_id,
            name=name,
            version=digest or None,
            module=owning_module(path),
            paths=(path,),
            relationships=(Relationship.BUILD_TOOL,),
            evidence=(evidence,),
            distribution=DistributionStatus.NOT_DISTRIBUTED,
            details=normalize_details(
                {"immutable_ref": digest.startswith("sha256:") if digest else False}
            ),
        )
        finding = None
        if not digest.startswith("sha256:"):
            finding = Finding.create(
                code="DISCOVERY_ACTION_UNPINNED",
                severity=Severity.WARNING,
                decision=Decision.REVIEW,
                component_id=component_id,
                message=f"Container action {reference!r} is not pinned by digest.",
                evidence=(evidence,),
                remediation=(
                    "Pin the container action with an immutable sha256 digest.",
                ),
                fingerprint_values=(reference,),
            )
        return component, finding

    action, separator, ref = reference.rpartition("@")
    if not separator:
        action, ref = reference, ""
    parts = action.split("/")
    if len(parts) < 2:
        return None
    owner, repository = parts[:2]
    component_id = f"pkg:github/{quote(owner, safe='')}/{quote(repository, safe='')}"
    if ref:
        component_id += f"@{quote(ref, safe='.-_')}"
    pinned = bool(FULL_SHA.fullmatch(ref))
    evidence = Evidence(
        kind=EvidenceKind.GITHUB_ACTION,
        source="github-workflow-discovery",
        path=path,
        value=reference,
        confidence=Confidence.HIGH,
        details=normalize_details(details),
    )
    component = Component(
        id=component_id,
        name=action,
        version=ref or None,
        module=owning_module(path),
        paths=(path,),
        relationships=(Relationship.BUILD_TOOL,),
        evidence=(evidence,),
        distribution=DistributionStatus.NOT_DISTRIBUTED,
        details=normalize_details(
            {"action_path": "/".join(parts[2:]), "immutable_ref": pinned}
        ),
    )
    finding = None
    if not pinned:
        finding = Finding.create(
            code="DISCOVERY_ACTION_UNPINNED",
            severity=Severity.WARNING,
            decision=Decision.REVIEW,
            component_id=component_id,
            message=f"GitHub Action {reference!r} is not pinned to a full commit SHA.",
            evidence=(evidence,),
            remediation=("Pin the action to a full 40-character commit SHA.",),
            fingerprint_values=(reference,),
        )
    return component, finding


def discover_github_actions(
    builder: InventoryBuilder, root: Path, files: tuple[str, ...]
) -> None:
    for path in files:
        if not _is_workflow(path):
            continue
        try:
            text = (root / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for reference, line in extract_action_references(text):
            builder.add_discovery(
                Discovery(
                    kind="github-action",
                    path=path,
                    module=owning_module(path),
                    ecosystem="github-actions",
                    details=normalize_details(
                        {
                            "line": line,
                            "local": reference.startswith("./"),
                            "reference": reference,
                        }
                    ),
                )
            )
            result = _action_component(reference, path, line)
            if result:
                component, finding = result
                builder.add_component(component)
                if finding:
                    builder.add_finding(finding)
