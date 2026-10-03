# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json

from ov_contrib_license.model import Inventory


def _spdx_id(component_id: str) -> str:
    digest = hashlib.sha256(component_id.encode("utf-8")).hexdigest()[:20]
    return f"SPDXRef-Package-{digest}"


def render_spdx(inventory: Inventory) -> str:
    canonical = json.dumps(inventory.to_dict(), sort_keys=True, separators=(",", ":"))
    namespace_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    packages: list[dict[str, object]] = []
    relationships: list[dict[str, str]] = []
    for component in inventory.components:
        spdx_id = _spdx_id(component.id)
        package: dict[str, object] = {
            "SPDXID": spdx_id,
            "name": component.name,
            "downloadLocation": dict(component.details).get(
                "source_url", "NOASSERTION"
            ),
            "filesAnalyzed": False,
            "licenseConcluded": (
                " AND ".join(component.detected_licenses)
                if component.detected_licenses
                else "NOASSERTION"
            ),
            "licenseDeclared": component.declared_license or "NOASSERTION",
            "copyrightText": "NOASSERTION",
        }
        if component.version:
            package["versionInfo"] = component.version
        if component.id.startswith("pkg:"):
            package["externalRefs"] = [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": component.id,
                }
            ]
        packages.append(package)
        relationships.append(
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": spdx_id,
            }
        )
    data = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "openvino-contrib-license-inventory",
        "documentNamespace": f"https://openvinotoolkit.org/spdx/openvino-contrib/{namespace_hash}",
        "creationInfo": {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Tool: ov-contrib-license"],
        },
        "packages": packages,
        "relationships": relationships,
    }
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
