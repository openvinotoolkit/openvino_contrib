# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
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


def _license(values: Any) -> str | None:
    if isinstance(values, str):
        return values if values and values != "NOASSERTION" else None
    if not isinstance(values, list):
        return None
    result: set[str] = set()
    for item in values:
        if isinstance(item, str):
            result.add(item)
        elif isinstance(item, dict):
            value = item.get("spdxExpression") or item.get("value")
            if value:
                result.add(str(value))
    return " AND ".join(sorted(result)) if result else None


def _native_components(data: dict[str, Any]) -> tuple[Component, ...]:
    artifacts = data.get("artifacts", ())
    if not isinstance(artifacts, list):
        raise ProviderError("Syft result does not contain an artifacts list")
    result: list[Component] = []
    for item in artifacts:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"])
        version = str(item.get("version") or "")
        purl = item.get("purl")
        component_id = (
            str(purl)
            if isinstance(purl, str) and purl.startswith("pkg:")
            else f"pkg:generic/{quote(name.lower(), safe='.-_')}"
            + (f"@{quote(version, safe='.-_+')}" if version else "")
        )
        paths = tuple(
            sorted(
                {
                    str(location.get("path", "")).lstrip("/")
                    for location in item.get("locations", ())
                    if isinstance(location, dict) and location.get("path")
                }
            )
        )
        evidence = Evidence(
            EvidenceKind.ARTIFACT_SBOM,
            "syft",
            paths[0] if paths else None,
            component_id,
            Confidence.HIGH,
            normalize_details({"syft_id": item.get("id"), "type": item.get("type")}),
        )
        result.append(
            Component(
                id=component_id,
                name=name,
                version=version or None,
                paths=paths,
                declared_license=_license(item.get("licenses")),
                relationships=(Relationship.BUNDLED_BINARY,),
                evidence=(evidence,),
                distribution=DistributionStatus.DISTRIBUTED,
                details=normalize_details(
                    {"artifact_scope": "runtime-package", "provider": "syft"}
                ),
            )
        )
    return tuple(result)


def _spdx_components(data: dict[str, Any]) -> tuple[Component, ...]:
    packages = data.get("packages", ())
    if not isinstance(packages, list):
        raise ProviderError("SPDX result does not contain a packages list")
    result: list[Component] = []
    for item in packages:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"])
        version = str(item.get("versionInfo") or "")
        refs = item.get("externalRefs", ())
        purl = next(
            (
                str(ref.get("referenceLocator"))
                for ref in refs
                if isinstance(ref, dict)
                and str(ref.get("referenceType", "")).lower() == "purl"
            ),
            None,
        )
        component_id = purl or f"pkg:generic/{quote(name.lower(), safe='.-_')}" + (
            f"@{quote(version, safe='.-_+')}" if version else ""
        )
        evidence = Evidence(
            EvidenceKind.ARTIFACT_SBOM,
            "syft-spdx",
            value=component_id,
            confidence=Confidence.HIGH,
        )
        result.append(
            Component(
                id=component_id,
                name=name,
                version=version or None,
                declared_license=_license(item.get("licenseDeclared")),
                detected_licenses=tuple(
                    value
                    for value in (_license(item.get("licenseConcluded")),)
                    if value
                ),
                relationships=(Relationship.BUNDLED_BINARY,),
                evidence=(evidence,),
                distribution=DistributionStatus.DISTRIBUTED,
                details=normalize_details(
                    {"artifact_scope": "runtime-package", "provider": "syft"}
                ),
            )
        )
    return tuple(result)


def import_syft(path: Path) -> Inventory:
    data = read_json(path, "Syft")
    from ov_contrib_license.model import RepositoryInfo

    builder = InventoryBuilder(RepositoryInfo("artifact", None))
    components = (
        _native_components(data) if "artifacts" in data else _spdx_components(data)
    )
    for component in components:
        builder.add_component(component)
    descriptor = data.get("descriptor", {})
    version = (
        str(descriptor.get("version", "unknown"))
        if isinstance(descriptor, dict)
        else "unknown"
    )
    builder.add_provider(Provider("syft", version, "SUCCESS"))
    return builder.build()


def run_syft(artifact: Path, command: str = "syft") -> Inventory:
    try:
        result = subprocess.run(
            [command, str(artifact), "-o", "syft-json"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError as error:
        raise ProviderError(
            f"Unable to execute Syft command {command!r}: {error}"
        ) from error
    if result.returncode:
        reason = result.stderr.strip() or f"exit status {result.returncode}"
        raise ProviderError(f"Syft failed: {reason}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ProviderError(f"Syft returned invalid JSON: {error}") from error
    if not isinstance(data, dict):
        raise ProviderError("Syft returned a non-object JSON document")
    components = _native_components(data)
    from ov_contrib_license.model import RepositoryInfo

    builder = InventoryBuilder(RepositoryInfo(str(artifact), None))
    for component in components:
        builder.add_component(component)
    descriptor = data.get("descriptor", {})
    version = (
        str(descriptor.get("version", "unknown"))
        if isinstance(descriptor, dict)
        else "unknown"
    )
    builder.add_provider(Provider("syft", version, "SUCCESS"))
    return builder.build()
