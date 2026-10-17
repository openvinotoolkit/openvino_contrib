# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ov_contrib_license import __version__

from .types import Component, Discovery, Evidence, Finding, normalize_details


@dataclass(frozen=True)
class RepositoryInfo:
    path: str
    revision: str | None
    base_ref: str | None = None
    head_ref: str | None = None
    impacted_scopes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "revision": self.revision,
            "base_ref": self.base_ref,
            "head_ref": self.head_ref,
            "impacted_scopes": sorted(self.impacted_scopes),
        }


@dataclass(frozen=True)
class Provider:
    name: str
    version: str
    status: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "status": self.status}


@dataclass(frozen=True)
class Inventory:
    repository: RepositoryInfo
    components: tuple[Component, ...] = field(default_factory=tuple)
    discoveries: tuple[Discovery, ...] = field(default_factory=tuple)
    findings: tuple[Finding, ...] = field(default_factory=tuple)
    providers: tuple[Provider, ...] = field(default_factory=tuple)
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported inventory schema version: {self.schema_version}"
            )
        ids = [item.id for item in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError("Inventory contains duplicate component IDs")
        finding_ids = [item.id for item in self.findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("Inventory contains duplicate finding IDs")
        for component in self.components:
            if not component.id or not component.name:
                raise ValueError("Component ID and name must not be empty")
            for path in component.paths:
                if path.startswith(("/", "../")) or path == "..":
                    raise ValueError(
                        f"Component path is not repository-relative: {path}"
                    )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "repository": self.repository.to_dict(),
            "components": [item.to_dict() for item in self.components],
            "discoveries": [item.to_dict() for item in self.discoveries],
            "findings": [item.to_dict() for item in self.findings],
            "providers": [item.to_dict() for item in self.providers],
        }


class InventoryBuilder:
    def __init__(self, repository: RepositoryInfo) -> None:
        self.repository = repository
        self._components: dict[str, Component] = {}
        self._discoveries: set[Discovery] = set()
        self._findings: dict[str, Finding] = {}
        self._providers: dict[str, Provider] = {
            "repository-discovery": Provider(
                "repository-discovery", __version__, "SUCCESS"
            )
        }

    @classmethod
    def from_inventory(cls, inventory: Inventory) -> InventoryBuilder:
        builder = cls(inventory.repository)
        for component in inventory.components:
            builder.add_component(component)
        for discovery in inventory.discoveries:
            builder.add_discovery(discovery)
        for finding in inventory.findings:
            builder.add_finding(finding)
        for provider in inventory.providers:
            builder.add_provider(provider)
        return builder

    def add_component(self, component: Component) -> None:
        previous = self._components.get(component.id)
        if previous is None:
            self._components[component.id] = component
            return

        modules = set(filter(None, (previous.module, component.module)))
        modules.update(
            filter(None, dict(previous.details).get("modules", "").split(","))
        )
        modules.update(
            filter(None, dict(component.details).get("modules", "").split(","))
        )
        module = next(iter(modules)) if len(modules) == 1 else None
        details = dict(previous.details)
        for key, value in component.details:
            if key in details and details[key] != value:
                details[key] = ",".join(
                    sorted(set(details[key].split(",")) | set(value.split(",")))
                )
            else:
                details[key] = value
        if len(modules) > 1:
            details["modules"] = ",".join(sorted(modules))

        evidence = tuple(
            sorted(set(previous.evidence + component.evidence), key=Evidence.sort_key)
        )
        self._components[component.id] = replace(
            previous,
            name=previous.name or component.name,
            version=previous.version or component.version,
            module=module,
            paths=tuple(sorted(set(previous.paths + component.paths))),
            declared_license=previous.declared_license or component.declared_license,
            detected_licenses=tuple(
                sorted(set(previous.detected_licenses + component.detected_licenses))
            ),
            relationships=tuple(
                sorted(
                    set(previous.relationships + component.relationships),
                    key=lambda item: item.value,
                )
            ),
            evidence=evidence,
            distribution=(
                previous.distribution
                if previous.distribution == component.distribution
                else type(previous.distribution).UNKNOWN
            ),
            obligations=tuple(
                sorted(
                    set(previous.obligations + component.obligations),
                    key=lambda item: item.value,
                )
            ),
            details=normalize_details(details),
        )

    def add_discovery(self, discovery: Discovery) -> None:
        self._discoveries.add(discovery)

    def add_finding(self, finding: Finding) -> None:
        self._findings[finding.id] = finding

    def add_provider(self, provider: Provider) -> None:
        self._providers[provider.name] = provider

    def build(self) -> Inventory:
        inventory = Inventory(
            repository=self.repository,
            components=tuple(
                sorted(self._components.values(), key=lambda item: item.id)
            ),
            discoveries=tuple(sorted(self._discoveries, key=Discovery.sort_key)),
            findings=tuple(sorted(self._findings.values(), key=Finding.sort_key)),
            providers=tuple(
                sorted(self._providers.values(), key=lambda item: item.name)
            ),
        )
        inventory.validate()
        return inventory
