# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from ov_contrib_license.model import (
    Component,
    Confidence,
    DistributionStatus,
    Evidence,
    EvidenceKind,
    Inventory,
    InventoryBuilder,
    Provider,
    Relationship,
)
from ov_contrib_license.model.types import normalize_details

from .common import ProviderError, read_json


def _identifier(value: Any) -> tuple[str, str, str, str]:
    if isinstance(value, dict):
        return (
            str(value.get("type", "Generic")),
            str(value.get("namespace", "")),
            str(value.get("name", "unknown")),
            str(value.get("version", "")),
        )
    if isinstance(value, str):
        parts = value.split(":")
        if len(parts) >= 4:
            return parts[0], parts[1], parts[2], ":".join(parts[3:])
    raise ProviderError(f"ORT package has an invalid identifier: {value!r}")


def _purl(item: dict[str, Any], identifier: tuple[str, str, str, str]) -> str:
    value = item.get("purl")
    if isinstance(value, str) and value.startswith("pkg:"):
        return value
    package_type, namespace, name, version = identifier
    ecosystem = {
        "PyPI": "pypi",
        "NPM": "npm",
        "Maven": "maven",
        "Gradle": "maven",
        "GoMod": "golang",
        "Cargo": "cargo",
        "Conan": "conan",
    }.get(package_type, "generic")
    identity = "/".join(filter(None, (namespace, name)))
    result = f"pkg:{ecosystem}/{quote(identity, safe='/.-_')}"
    return result + (f"@{quote(version, safe='.-_+')}" if version else "")


def _license(item: dict[str, Any]) -> str | None:
    processed = item.get("declared_licenses_processed")
    if isinstance(processed, dict):
        expression = processed.get("spdx_expression")
        if isinstance(expression, str) and expression:
            return expression
    values = item.get("declared_licenses")
    if isinstance(values, list):
        licenses = sorted({str(value) for value in values if value})
        if licenses:
            return " AND ".join(licenses)
    return None


def _packages(data: dict[str, Any]) -> list[dict[str, Any]]:
    analyzer = data.get("analyzer", data)
    if isinstance(analyzer, dict):
        analyzer = analyzer.get("result", analyzer)
    packages = analyzer.get("packages", ()) if isinstance(analyzer, dict) else ()
    if not isinstance(packages, list):
        raise ProviderError("ORT result does not contain analyzer.result.packages")
    return [item for item in packages if isinstance(item, dict)]


def import_ort(inventory: Inventory, path: Path) -> Inventory:
    data = read_json(path, "ORT")
    builder = InventoryBuilder.from_inventory(inventory)
    for item in _packages(data):
        identifier = _identifier(item.get("id"))
        component_id = _purl(item, identifier)
        package_type, namespace, name, version = identifier
        evidence = Evidence(
            kind=EvidenceKind.ORT,
            source="ort-analyzer",
            value=component_id,
            confidence=Confidence.HIGH,
            details=normalize_details(
                {
                    "package_type": package_type,
                    "namespace": namespace,
                    "source": item.get("source_artifact", {}).get("url")
                    if isinstance(item.get("source_artifact"), dict)
                    else None,
                }
            ),
        )
        builder.add_component(
            Component(
                id=component_id,
                name=name,
                version=version or None,
                declared_license=_license(item),
                relationships=(Relationship.RUNTIME_DEPENDENCY,),
                evidence=(evidence,),
                distribution=DistributionStatus.UNKNOWN,
                details=normalize_details({"namespace": namespace, "provider": "ort"}),
            )
        )
    version = str(data.get("ort_version") or data.get("version") or "unknown")
    builder.add_provider(Provider("ort", version, "SUCCESS"))
    return builder.build()
