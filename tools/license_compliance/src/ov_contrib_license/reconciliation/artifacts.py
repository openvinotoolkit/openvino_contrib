# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

from ov_contrib_license.model import (
    Component,
    Decision,
    DistributionStatus,
    Finding,
    Inventory,
    Relationship,
    Severity,
)

_NON_RUNTIME = frozenset(
    {
        Relationship.BUILD_TOOL,
        Relationship.CODE_GENERATOR,
        Relationship.TEST_ONLY,
        Relationship.DEV_ONLY,
        Relationship.DOCUMENTATION_ASSET,
    }
)


def _identity(component: Component) -> tuple[str, str]:
    return component.name.casefold(), component.version or ""


def _expected(component: Component) -> bool:
    return component.distribution is DistributionStatus.DISTRIBUTED and not set(
        component.relationships
    ).issubset(_NON_RUNTIME)


def reconcile_artifact(source: Inventory, artifact: Inventory) -> Inventory:
    by_id = {component.id: component for component in source.components}
    by_identity = {_identity(component): component for component in source.components}
    matched_source: set[str] = set()
    components = dict(by_id)
    findings = list(source.findings)

    for actual in artifact.components:
        expected = by_id.get(actual.id) or by_identity.get(_identity(actual))
        if expected is None:
            components[actual.id] = actual
            findings.append(
                Finding.create(
                    code="ARTIFACT_UNDECLARED_COMPONENT",
                    severity=Severity.ERROR,
                    decision=Decision.FAIL,
                    component_id=actual.id,
                    message=f"Artifact contains undeclared component {actual.name!r}.",
                    evidence=actual.evidence,
                    remediation=(
                        "Declare the source dependency and classify its distributed use.",
                    ),
                    fingerprint_values=(
                        actual.version or "",
                        Relationship.BUNDLED_BINARY.value,
                        DistributionStatus.DISTRIBUTED.value,
                    ),
                )
            )
            continue
        matched_source.add(expected.id)
        components[expected.id] = replace(
            expected,
            distribution=DistributionStatus.DISTRIBUTED,
            relationships=tuple(
                sorted(
                    set(expected.relationships + (Relationship.BUNDLED_BINARY,)),
                    key=lambda item: item.value,
                )
            ),
            evidence=tuple(
                sorted(
                    set(expected.evidence + actual.evidence),
                    key=lambda item: item.sort_key(),
                )
            ),
            details=tuple(sorted(set(expected.details + actual.details))),
        )
        source_license = expected.declared_license
        artifact_license = actual.declared_license
        if source_license and artifact_license and source_license != artifact_license:
            findings.append(
                Finding.create(
                    code="ARTIFACT_LICENSE_MISMATCH",
                    severity=Severity.ERROR,
                    decision=Decision.REVIEW,
                    component_id=expected.id,
                    message=(
                        f"Artifact license {artifact_license!r} differs from source license "
                        f"{source_license!r} for {expected.name}."
                    ),
                    evidence=expected.evidence + actual.evidence,
                    remediation=(
                        "Resolve the source/artifact license evidence mismatch.",
                    ),
                    fingerprint_values=(source_license, artifact_license),
                )
            )

    for expected in source.components:
        if _expected(expected) and expected.id not in matched_source:
            findings.append(
                Finding.create(
                    code="ARTIFACT_MISSING_EXPECTED_COMPONENT",
                    severity=Severity.WARNING,
                    decision=Decision.REVIEW,
                    component_id=expected.id,
                    message=f"Expected distributed component {expected.name!r} was not found in the artifact.",
                    evidence=expected.evidence,
                    remediation=(
                        "Confirm the artifact scope or correct the distribution classification.",
                    ),
                    fingerprint_values=(
                        expected.version or "",
                        *[item.value for item in expected.relationships],
                    ),
                )
            )

    providers = {item.name: item for item in source.providers}
    providers.update({item.name: item for item in artifact.providers})
    return replace(
        source,
        components=tuple(sorted(components.values(), key=lambda item: item.id)),
        findings=tuple(
            sorted({item.id: item for item in findings}.values(), key=Finding.sort_key)
        ),
        providers=tuple(sorted(providers.values(), key=lambda item: item.name)),
    )
