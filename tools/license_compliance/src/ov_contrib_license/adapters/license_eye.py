# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

from ov_contrib_license.model import (
    Decision,
    Evidence,
    EvidenceKind,
    Finding,
    Inventory,
    InventoryBuilder,
    Provider,
    Severity,
)

from .common import ProviderError, read_json


def _paths(value: Any) -> tuple[str, ...]:
    if isinstance(value, int):
        return tuple(f"<count:{value}>" for _ in range(1 if value else 0))
    if not isinstance(value, list):
        return ()
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict) and item.get("path"):
            result.append(str(item["path"]))
    return tuple(sorted(set(result)))


def import_license_eye(inventory: Inventory, path: Path) -> Inventory:
    data = read_json(path, "License Eye")
    summary = data.get("header", data)
    if not isinstance(summary, dict):
        raise ProviderError("License Eye result does not contain a header summary")
    builder = InventoryBuilder.from_inventory(inventory)
    invalid = _paths(summary.get("invalid"))
    ignored = _paths(summary.get("ignored"))
    if invalid:
        evidence = tuple(
            Evidence(EvidenceKind.SPDX_HEADER, "license-eye", item, "invalid")
            for item in invalid
        )
        builder.add_finding(
            Finding.create(
                code="HEADER_INVALID",
                severity=Severity.ERROR,
                decision=Decision.FAIL,
                component_id=None,
                message=f"License Eye reported {len(invalid)} invalid header path(s).",
                evidence=evidence,
                remediation=(
                    "Add an approved first-party header or a reviewed exclusion.",
                ),
            )
        )
    if ignored:
        evidence = tuple(
            Evidence(EvidenceKind.SPDX_HEADER, "license-eye", item, "ignored")
            for item in ignored
        )
        builder.add_finding(
            Finding.create(
                code="HEADER_IGNORED_PATHS",
                severity=Severity.WARNING,
                decision=Decision.REVIEW,
                component_id=None,
                message=f"License Eye ignored {len(ignored)} path(s); ignored content is not auto-approved.",
                evidence=evidence,
                remediation=(
                    "Audit broad ignores and classify embedded third-party content separately.",
                ),
            )
        )
    builder.add_provider(
        Provider("license-eye", str(data.get("version", "unknown")), "SUCCESS")
    )
    return builder.build()
