# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ov_contrib_license.model import (
    Decision,
    DistributionStatus,
    Obligation,
    Relationship,
)

_SHA256_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Unable to read policy file {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in policy file {path}: {error}") from error
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Policy file {path} must contain a mapping")  # noqa: TRY004
    return data


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected a string or list of strings, got {value!r}")
    return tuple(value)


def _mapping_list(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"Policy key {key!r} must contain a list")  # noqa: TRY004
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Every {key!r} entry must be a mapping")
    return value


def _string_mapping(data: dict[str, Any], key: str) -> dict[str, str]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Policy key {key!r} must contain a mapping")  # noqa: TRY004
    if not all(
        isinstance(item_key, str)
        and item_key.strip()
        and isinstance(item_value, str)
        and item_value.strip()
        for item_key, item_value in value.items()
    ):
        raise ValueError(f"Every {key!r} key and value must be a non-empty string")
    return value


@dataclass(frozen=True)
class PolicyRule:
    id: str
    decision: Decision
    licenses: tuple[str, ...] = ()
    license_classes: tuple[str, ...] = ()
    relationships: tuple[Relationship, ...] = ()
    distributions: tuple[DistributionStatus, ...] = ()
    modules: tuple[str, ...] = ()
    artifact_scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExceptionRule:
    id: str
    component: str
    module: str
    relationships: tuple[Relationship, ...]
    distributions: tuple[DistributionStatus, ...]
    rationale: str
    approved_by: tuple[str, ...]
    expires: dt.date
    decision: Decision = Decision.PASS


@dataclass(frozen=True)
class BaselineEntry:
    finding_fingerprint: str
    reason: str


@dataclass(frozen=True)
class ToolchainPolicy:
    allowed_license_classes: tuple[str, ...] = ("permissive",)
    allowed_explicit_licenses: tuple[str, ...] = ()
    require_full_sha_for_github_actions: bool = True
    action_licenses: tuple[tuple[str, str], ...] = ()
    dependency_licenses: tuple[tuple[str, str], ...] = ()
    forbidden_services: tuple[tuple[str, str], ...] = ()

    @property
    def actions(self) -> dict[str, str]:
        return dict(self.action_licenses)

    @property
    def dependencies(self) -> dict[str, str]:
        return dict(self.dependency_licenses)


@dataclass(frozen=True)
class PolicyConfig:
    directory: Path
    license_classes: dict[str, frozenset[str]]
    rules: tuple[PolicyRule, ...]
    obligations_by_license: dict[str, tuple[Obligation, ...]]
    obligations_by_class: dict[str, tuple[Obligation, ...]]
    exceptions: tuple[ExceptionRule, ...]
    baseline: tuple[BaselineEntry, ...]
    toolchain: ToolchainPolicy
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def baseline_by_id(self) -> dict[str, BaselineEntry]:
        return {item.finding_fingerprint: item for item in self.baseline}


def _load_rules(data: dict[str, Any]) -> tuple[PolicyRule, ...]:
    rules: list[PolicyRule] = []
    for item in _mapping_list(data, "rules"):
        if not isinstance(item.get("when", {}), dict):
            raise ValueError(  # noqa: TRY004
                "Each policy rule must be a mapping with a 'when' mapping"
            )
        identifier = item.get("id")
        decision = item.get("decision")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Each policy rule must have a non-empty string ID")
        if not isinstance(decision, str):
            raise ValueError(  # noqa: TRY004
                f"Policy rule {identifier} must have a decision"
            )
        when = item.get("when", {})
        rules.append(
            PolicyRule(
                id=identifier,
                decision=Decision(decision),
                licenses=_strings(when.get("license")),
                license_classes=_strings(when.get("license_class")),
                relationships=tuple(
                    Relationship(value) for value in _strings(when.get("relationship"))
                ),
                distributions=tuple(
                    DistributionStatus(value)
                    for value in _strings(when.get("distribution"))
                ),
                modules=_strings(when.get("module")),
                artifact_scopes=_strings(when.get("artifact_scope")),
            )
        )
    identifiers = [item.id for item in rules]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Policy rule IDs must be unique")
    return tuple(rules)


def _load_exceptions(data: dict[str, Any]) -> tuple[ExceptionRule, ...]:
    result: list[ExceptionRule] = []
    for item in _mapping_list(data, "exceptions"):
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Each exception must have a non-empty string ID")
        component = item.get("component")
        if (
            not isinstance(component, str)
            or "@" not in component
            or not all(part.strip() for part in component.rsplit("@", 1))
        ):
            raise ValueError(
                "Exception component must identify an exact version/revision"
            )
        module = item.get("module")
        rationale = item.get("rationale")
        if not isinstance(module, str) or not module.strip():
            raise ValueError(f"Exception {identifier} must constrain a module")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"Exception {identifier} must contain a rationale")
        approved_by = _strings(item.get("approved_by"))
        if not approved_by or not all(item.strip() for item in approved_by):
            raise ValueError(f"Exception {identifier} must contain approval metadata")
        expires_value = item.get("expires")
        try:
            expires = (
                expires_value
                if isinstance(expires_value, dt.date)
                else dt.date.fromisoformat(str(expires_value))
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Exception {identifier} has an invalid expiration date"
            ) from error
        relationships = tuple(
            Relationship(value) for value in _strings(item.get("allowed_relationships"))
        )
        distributions = tuple(
            DistributionStatus(value)
            for value in _strings(item.get("allowed_distribution"))
        )
        if not relationships or not distributions:
            raise ValueError(
                f"Exception {identifier} must constrain relationship and distribution"
            )
        result.append(
            ExceptionRule(
                id=identifier,
                component=component,
                module=module,
                relationships=relationships,
                distributions=distributions,
                rationale=rationale,
                approved_by=approved_by,
                expires=expires,
                decision=Decision(str(item.get("decision", "PASS"))),
            )
        )
    identifiers = [item.id for item in result]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Policy exception IDs must be unique")
    return tuple(result)


def _load_obligations(
    data: dict[str, Any],
) -> tuple[dict[str, tuple[Obligation, ...]], dict[str, tuple[Obligation, ...]]]:
    def convert(values: Any) -> dict[str, tuple[Obligation, ...]]:
        if values is None:
            return {}
        if not isinstance(values, dict):
            raise ValueError("Obligation mappings must be mappings")  # noqa: TRY004
        return {
            str(key): tuple(Obligation(value) for value in _strings(item))
            for key, item in values.items()
        }

    return convert(data.get("licenses")), convert(data.get("classes"))


def _load_toolchain(data: dict[str, Any]) -> tuple[ToolchainPolicy, dict[str, Any]]:
    actions = _string_mapping(data, "actions")
    dependencies = _string_mapping(data, "dependencies")
    require_full_sha = data.get("require_full_sha_for_github_actions", True)
    if not isinstance(require_full_sha, bool):
        raise ValueError(  # noqa: TRY004
            "Toolchain require_full_sha_for_github_actions must be a boolean"
        )
    forbidden: list[tuple[str, str]] = []
    for item in _mapping_list(data, "forbidden_services"):
        identifier = item.get("id")
        reason = item.get("reason", "forbidden by policy")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Each forbidden service must have a non-empty string ID")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Forbidden service {identifier} must have a reason")
        forbidden.append((identifier, reason))
    forbidden_ids = [item[0] for item in forbidden]
    if len(forbidden_ids) != len(set(forbidden_ids)):
        raise ValueError("Forbidden service IDs must be unique")
    providers = data.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("Toolchain providers must be a mapping")  # noqa: TRY004
    for name, provider in providers.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(provider, dict)
        ):
            raise ValueError("Every toolchain provider must be a named mapping")
        if "enabled" in provider and not isinstance(provider["enabled"], bool):
            raise ValueError(
                f"Toolchain provider {name} enabled flag must be a boolean"
            )
        if "command" in provider and (
            not isinstance(provider["command"], str) or not provider["command"].strip()
        ):
            raise ValueError(f"Toolchain provider {name} command must be a string")
    return ToolchainPolicy(
        allowed_license_classes=_strings(data.get("allowed_license_classes"))
        or ("permissive",),
        allowed_explicit_licenses=_strings(data.get("allowed_explicit_licenses")),
        require_full_sha_for_github_actions=require_full_sha,
        action_licenses=tuple(
            sorted((str(key).lower(), str(value)) for key, value in actions.items())
        ),
        dependency_licenses=tuple(
            sorted(
                (str(key).lower(), str(value)) for key, value in dependencies.items()
            )
        ),
        forbidden_services=tuple(sorted(forbidden)),
    ), providers


def _load_baseline_entries(data: dict[str, Any]) -> tuple[BaselineEntry, ...]:
    result: list[BaselineEntry] = []
    for item in _mapping_list(data, "baseline"):
        fingerprint = item.get("finding_fingerprint")
        reason = item.get("reason")
        if not isinstance(fingerprint, str) or not _SHA256_FINGERPRINT.fullmatch(
            fingerprint
        ):
            raise ValueError("Baseline entries require a sha256 finding_fingerprint")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Baseline entry {fingerprint} requires a reason")
        result.append(BaselineEntry(fingerprint, reason))
    fingerprints = [item.finding_fingerprint for item in result]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("Baseline finding fingerprints must be unique")
    return tuple(result)


def load_policy(directory: Path) -> PolicyConfig:
    directory = directory.resolve()
    required = (
        "licenses.yml",
        "rules.yml",
        "obligations.yml",
        "exceptions.yml",
        "baseline.yml",
        "toolchain.yml",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError(
            f"Policy directory {directory} is missing: {', '.join(missing)}"
        )

    licenses_data = _load_mapping(directory / "licenses.yml")
    classes_data = licenses_data.get("classes", {})
    if not isinstance(classes_data, dict):
        raise ValueError("licenses.yml 'classes' must be a mapping")  # noqa: TRY004
    classes = {
        str(key): frozenset(_strings(value)) for key, value in classes_data.items()
    }
    duplicate_licenses: dict[str, list[str]] = {}
    for class_name, values in classes.items():
        for license_id in values:
            duplicate_licenses.setdefault(license_id, []).append(class_name)
    duplicates = {
        key: value for key, value in duplicate_licenses.items() if len(value) > 1
    }
    if duplicates:
        raise ValueError(f"Licenses must belong to one class: {duplicates}")

    obligations = _load_obligations(_load_mapping(directory / "obligations.yml"))
    rules = _load_rules(_load_mapping(directory / "rules.yml"))
    unknown_rule_classes = {
        class_name
        for rule in rules
        for class_name in rule.license_classes
        if class_name not in classes
    }
    unknown_obligation_classes = set(obligations[1]) - set(classes)
    if unknown_rule_classes or unknown_obligation_classes:
        unknown = sorted(unknown_rule_classes | unknown_obligation_classes)
        raise ValueError(
            f"Policy references unknown license classes: {', '.join(unknown)}"
        )
    baseline = _load_baseline_entries(_load_mapping(directory / "baseline.yml"))
    toolchain, providers = _load_toolchain(_load_mapping(directory / "toolchain.yml"))
    unknown_toolchain_classes = set(toolchain.allowed_license_classes) - set(classes)
    if unknown_toolchain_classes:
        raise ValueError(
            "Toolchain policy references unknown license classes: "
            + ", ".join(sorted(unknown_toolchain_classes))
        )
    return PolicyConfig(
        directory=directory,
        license_classes=classes,
        rules=rules,
        obligations_by_license=obligations[0],
        obligations_by_class=obligations[1],
        exceptions=_load_exceptions(_load_mapping(directory / "exceptions.yml")),
        baseline=baseline,
        toolchain=toolchain,
        providers=providers,
    )


def load_baseline(path: Path) -> tuple[BaselineEntry, ...]:
    return _load_baseline_entries(_load_mapping(path))
