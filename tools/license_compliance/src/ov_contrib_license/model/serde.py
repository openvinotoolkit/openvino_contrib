# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inventory import Inventory, Provider, RepositoryInfo
from .types import (
    Component,
    Confidence,
    Decision,
    Discovery,
    DistributionStatus,
    Evidence,
    EvidenceKind,
    Finding,
    Obligation,
    Relationship,
    Severity,
    normalize_details,
)


def _evidence(data: dict[str, Any]) -> Evidence:
    return Evidence(
        kind=EvidenceKind(data["kind"]),
        source=str(data["source"]),
        path=data.get("path"),
        value=data.get("value"),
        confidence=Confidence(data.get("confidence", Confidence.MEDIUM.value)),
        details=normalize_details(data.get("details")),
    )


def inventory_from_dict(data: dict[str, Any]) -> Inventory:
    repository_data = data["repository"]
    repository = RepositoryInfo(
        path=str(repository_data.get("path", ".")),
        revision=repository_data.get("revision"),
        base_ref=repository_data.get("base_ref"),
        head_ref=repository_data.get("head_ref"),
        impacted_scopes=tuple(repository_data.get("impacted_scopes", ())),
    )
    components = tuple(
        Component(
            id=str(item["id"]),
            name=str(item["name"]),
            version=item.get("version"),
            module=item.get("module"),
            paths=tuple(item.get("paths", ())),
            declared_license=item.get("declared_license"),
            detected_licenses=tuple(item.get("detected_licenses", ())),
            relationships=tuple(
                Relationship(value) for value in item.get("relationships", ("UNKNOWN",))
            ),
            evidence=tuple(_evidence(value) for value in item.get("evidence", ())),
            distribution=DistributionStatus(item.get("distribution", "UNKNOWN")),
            obligations=tuple(
                Obligation(value) for value in item.get("obligations", ())
            ),
            details=normalize_details(item.get("details")),
        )
        for item in data.get("components", ())
    )
    discoveries = tuple(
        Discovery(
            kind=str(item["kind"]),
            path=str(item["path"]),
            module=item.get("module"),
            ecosystem=item.get("ecosystem"),
            details=normalize_details(item.get("details")),
        )
        for item in data.get("discoveries", ())
    )
    findings = tuple(
        Finding(
            id=str(item["id"]),
            code=str(item["code"]),
            severity=Severity(item["severity"]),
            decision=Decision(item["decision"]),
            component_id=item.get("component_id"),
            message=str(item["message"]),
            evidence=tuple(_evidence(value) for value in item.get("evidence", ())),
            remediation=tuple(item.get("remediation", ())),
            suppressed=bool(item.get("suppressed", False)),
            suppression_reason=item.get("suppression_reason"),
        )
        for item in data.get("findings", ())
    )
    providers = tuple(
        Provider(str(item["name"]), str(item["version"]), str(item["status"]))
        for item in data.get("providers", ())
    )
    inventory = Inventory(
        repository=repository,
        components=components,
        discoveries=discoveries,
        findings=findings,
        providers=providers,
        schema_version=int(data.get("schema_version", 1)),
    )
    inventory.validate()
    return inventory


def read_inventory(path: Path) -> Inventory:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read inventory {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Inventory {path} must contain a JSON object")  # noqa: TRY004
    return inventory_from_dict(data)
