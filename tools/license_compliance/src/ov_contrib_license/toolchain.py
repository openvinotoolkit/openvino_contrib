# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from ov_contrib_license.model import (
    Decision,
    Evidence,
    EvidenceKind,
    Finding,
    Inventory,
    Severity,
)
from ov_contrib_license.policy import PolicyConfig

_DEPENDENCY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class ToolchainSummary:
    inventory: Inventory
    review_count: int
    fail_count: int
    suppressed_count: int


def _license_allowed(license_id: str, config: PolicyConfig) -> bool:
    if license_id in config.toolchain.allowed_explicit_licenses:
        return True
    return any(
        class_name in config.toolchain.allowed_license_classes
        and license_id in licenses
        for class_name, licenses in config.license_classes.items()
    )


def _action_name(value: str) -> str:
    reference = value.split("@", 1)[0]
    return "/".join(reference.split("/")[:2]).casefold()


def _dependencies(root: Path) -> tuple[str, ...]:
    pyproject = root / "tools" / "license_compliance" / "pyproject.toml"
    if not pyproject.is_file():
        return ()
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(
            f"Unable to parse compliance tool pyproject.toml: {error}"
        ) from error
    values = list(data.get("project", {}).get("dependencies", ()))
    values.extend(data.get("build-system", {}).get("requires", ()))
    names = []
    for value in values:
        match = _DEPENDENCY_NAME.match(str(value))
        if match:
            names.append(match.group(0).casefold())
    return tuple(sorted(set(names)))


def audit_toolchain(
    root: Path, inventory: Inventory, config: PolicyConfig
) -> ToolchainSummary:
    findings = list(inventory.findings)
    registry = config.toolchain.actions
    for component in inventory.components:
        action_evidence = [
            item
            for item in component.evidence
            if item.kind is EvidenceKind.GITHUB_ACTION
        ]
        if not action_evidence:
            continue
        evidence = action_evidence[0]
        reference = evidence.value or component.name
        if reference.startswith("docker://"):
            pinned = "@sha256:" in reference
            action_name = reference.split("@", 1)[0].casefold()
        else:
            ref = reference.rpartition("@")[2]
            pinned = bool(re.fullmatch(r"[0-9a-fA-F]{40}", ref))
            action_name = _action_name(reference)
        if config.toolchain.require_full_sha_for_github_actions and not pinned:
            findings.append(
                Finding.create(
                    code="TOOL_ACTION_UNPINNED",
                    severity=Severity.ERROR,
                    decision=Decision.FAIL,
                    component_id=component.id,
                    message=f"Toolchain action {reference!r} is not pinned immutably.",
                    evidence=action_evidence,
                    remediation=(
                        "Pin GitHub Actions to a full commit SHA or containers to sha256.",
                    ),
                    fingerprint_values=(reference,),
                )
            )
        license_id = registry.get(action_name)
        if license_id is None:
            findings.append(
                Finding.create(
                    code="TOOL_ACTION_UNKNOWN_LICENSE",
                    severity=Severity.WARNING,
                    decision=Decision.REVIEW,
                    component_id=component.id,
                    message=f"No reviewed toolchain license is registered for {action_name!r}.",
                    evidence=action_evidence,
                    remediation=(
                        "Review the public action source and add its license to toolchain.yml.",
                    ),
                    fingerprint_values=(action_name,),
                )
            )
        elif not _license_allowed(license_id, config):
            findings.append(
                Finding.create(
                    code="TOOL_ACTION_DISALLOWED_LICENSE",
                    severity=Severity.ERROR,
                    decision=Decision.FAIL,
                    component_id=component.id,
                    message=f"Toolchain action {action_name!r} has disallowed license {license_id}.",
                    evidence=action_evidence,
                    remediation=(
                        "Replace the action or approve a narrowly reviewed toolchain policy change.",
                    ),
                    fingerprint_values=(action_name, license_id),
                )
            )

    dependency_registry = config.toolchain.dependencies
    for dependency in _dependencies(root):
        license_id = dependency_registry.get(dependency)
        evidence = Evidence(
            EvidenceKind.PACKAGE_METADATA,
            "toolchain-policy",
            "tools/license_compliance/pyproject.toml",
            dependency,
        )
        if license_id is None:
            findings.append(
                Finding.create(
                    code="TOOL_DEPENDENCY_UNKNOWN_LICENSE",
                    severity=Severity.WARNING,
                    decision=Decision.REVIEW,
                    component_id=f"pkg:pypi/{dependency}",
                    message=f"Direct tool dependency {dependency!r} has no reviewed license entry.",
                    evidence=(evidence,),
                    remediation=(
                        "Review and register the dependency license in toolchain.yml.",
                    ),
                    fingerprint_values=(dependency,),
                )
            )
        elif not _license_allowed(license_id, config):
            findings.append(
                Finding.create(
                    code="TOOL_DEPENDENCY_DISALLOWED_LICENSE",
                    severity=Severity.ERROR,
                    decision=Decision.FAIL,
                    component_id=f"pkg:pypi/{dependency}",
                    message=f"Direct tool dependency {dependency!r} has disallowed license {license_id}.",
                    evidence=(evidence,),
                    remediation=(
                        "Remove the dependency or approve an explicit toolchain policy change.",
                    ),
                    fingerprint_values=(dependency, license_id),
                )
            )

    workflow_paths = sorted(
        {item.path for item in inventory.discoveries if item.kind == "github-action"}
    )
    for service_id, reason in config.toolchain.forbidden_services:
        for path in workflow_paths:
            try:
                text = (root / path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if service_id.casefold() not in text.casefold():
                continue
            evidence = Evidence(
                EvidenceKind.GITHUB_ACTION, "toolchain-policy", path, service_id
            )
            findings.append(
                Finding.create(
                    code="TOOL_PROPRIETARY_BACKEND_REQUIRED",
                    severity=Severity.ERROR,
                    decision=Decision.FAIL,
                    component_id=None,
                    message=f"Workflow references forbidden service {service_id!r}: {reason}",
                    evidence=(evidence,),
                    remediation=("Remove the mandatory proprietary decision backend.",),
                    fingerprint_values=(service_id, path),
                )
            )

    unique = {item.id: item for item in findings}
    baseline = config.baseline_by_id
    evaluated = tuple(
        sorted(
            (
                item.suppress(baseline[item.id].reason) if item.id in baseline else item
                for item in unique.values()
            ),
            key=Finding.sort_key,
        )
    )
    result = replace(inventory, findings=evaluated)
    active = [item for item in evaluated if not item.suppressed]
    return ToolchainSummary(
        inventory=result,
        review_count=sum(item.decision is Decision.REVIEW for item in active),
        fail_count=sum(item.decision is Decision.FAIL for item in active),
        suppressed_count=sum(item.suppressed for item in evaluated),
    )
