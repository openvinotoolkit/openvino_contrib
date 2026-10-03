# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from ov_contrib_license.model import (
    Decision,
    DistributionStatus,
    Finding,
    Inventory,
    Obligation,
    Severity,
)

_SEPARATOR = re.compile(r"(?m)^-{20,}\s*$")
_NAME_NORMALIZER = re.compile(r"[^a-z0-9]+")
_NOTICE_OBLIGATIONS = frozenset(
    {
        Obligation.RETAIN_COPYRIGHT,
        Obligation.RETAIN_LICENSE_TEXT,
        Obligation.RETAIN_NOTICE,
        Obligation.SEPARATE_LICENSE_TERMS,
    }
)


@dataclass(frozen=True)
class TppEntry:
    name: str
    text: str
    license_expression: str | None


def _normalized_name(value: str) -> str:
    return _NAME_NORMALIZER.sub("", value.casefold())


def _detected_license(text: str) -> str | None:
    lowered = text.casefold()
    if "apache license" in lowered and "version 2.0" in lowered:
        return "Apache-2.0"
    if (
        "mit license" in lowered
        or "permission is hereby granted, free of charge" in lowered
    ):
        return "MIT"
    if "bsd 3-clause" in lowered:
        return "BSD-3-Clause"
    if "bsd 2-clause" in lowered:
        return "BSD-2-Clause"
    return None


def parse_tpp(path: Path) -> tuple[str, tuple[TppEntry, ...]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise ValueError(
            f"Unable to read third-party programs file {path}: {error}"
        ) from error
    sections = _SEPARATOR.split(text)
    header = sections[0].rstrip()
    entries: list[TppEntry] = []
    for section in sections[1:]:
        stripped = section.strip()
        if not stripped:
            continue
        lines = stripped.splitlines()
        name = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if name:
            entries.append(TppEntry(name, body, _detected_license(body)))
    return header, tuple(entries)


def _component_names(component_id: str, name: str) -> set[str]:
    result = {_normalized_name(name)}
    identity = component_id.rsplit("/", 1)[-1].split("@", 1)[0]
    result.add(_normalized_name(identity))
    return result


def reconcile_tpp(inventory: Inventory, tpp_path: Path) -> Inventory:
    _, entries = parse_tpp(tpp_path)
    entry_by_name = {_normalized_name(entry.name): entry for entry in entries}
    matched: set[str] = set()
    findings = list(inventory.findings)
    for component in inventory.components:
        candidates = _component_names(component.id, component.name)
        entry = next(
            (entry_by_name[value] for value in candidates if value in entry_by_name),
            None,
        )
        if entry:
            matched.add(_normalized_name(entry.name))
        required = component.distribution is DistributionStatus.DISTRIBUTED and bool(
            set(component.obligations) & _NOTICE_OBLIGATIONS
        )
        if required and entry is None:
            findings.append(
                Finding.create(
                    code="TPP_MISSING_COMPONENT",
                    severity=Severity.ERROR,
                    decision=Decision.FAIL,
                    component_id=component.id,
                    message=f"Distributed component {component.name!r} requires a TPP entry.",
                    evidence=component.evidence,
                    remediation=(
                        "Add the reviewed attribution and applicable license/NOTICE text.",
                    ),
                    fingerprint_values=(
                        component.declared_license or "NOASSERTION",
                        *component.obligations,
                    ),
                )
            )
        if (
            entry
            and entry.license_expression
            and component.declared_license
            and entry.license_expression != component.declared_license
        ):
            findings.append(
                Finding.create(
                    code="TPP_LICENSE_MISMATCH",
                    severity=Severity.ERROR,
                    decision=Decision.REVIEW,
                    component_id=component.id,
                    message=(
                        f"TPP entry for {component.name!r} contains {entry.license_expression}, "
                        f"but inventory declares {component.declared_license}."
                    ),
                    evidence=component.evidence,
                    remediation=(
                        "Reconcile the component license evidence with the reviewed TPP text.",
                    ),
                    fingerprint_values=(
                        entry.license_expression,
                        component.declared_license,
                    ),
                )
            )
    for entry in entries:
        normalized = _normalized_name(entry.name)
        if normalized not in matched:
            findings.append(
                Finding.create(
                    code="TPP_UNRESOLVED_ENTRY",
                    severity=Severity.WARNING,
                    decision=Decision.REVIEW,
                    component_id=None,
                    message=f"TPP entry {entry.name!r} could not be mapped to discovered evidence.",
                    remediation=(
                        "Add canonical component mapping evidence or remove a confirmed stale entry.",
                    ),
                    fingerprint_values=(
                        normalized,
                        entry.license_expression or "NOASSERTION",
                    ),
                )
            )
    return replace(
        inventory,
        findings=tuple(
            sorted({item.id: item for item in findings}.values(), key=Finding.sort_key)
        ),
    )


def generate_tpp_preview(
    inventory: Inventory, existing_path: Path | None = None
) -> str:
    existing: dict[str, TppEntry] = {}
    if existing_path and existing_path.is_file():
        _, entries = parse_tpp(existing_path)
        existing = {_normalized_name(item.name): item for item in entries}
    lines = [
        "OpenVINO Contrib Third Party Programs File (PREVIEW)",
        "",
        "Generated output is a review aid and is not authoritative.",
    ]
    components = [
        item
        for item in inventory.components
        if item.distribution is DistributionStatus.DISTRIBUTED
        and bool(set(item.obligations) & _NOTICE_OBLIGATIONS)
    ]
    for component in sorted(
        components, key=lambda item: (item.name.casefold(), item.id)
    ):
        entry = next(
            (
                existing[name]
                for name in _component_names(component.id, component.name)
                if name in existing
            ),
            None,
        )
        lines.extend(
            (
                "",
                "-------------------------------------------------------------",
                "",
                component.name,
                "",
            )
        )
        if entry and entry.text:
            lines.append(entry.text)
        else:
            lines.append(
                f"SPDX license expression: {component.declared_license or 'NOASSERTION'}"
            )
            lines.append("TODO: insert reviewed copyright, license, and NOTICE text.")
    return "\n".join(lines).rstrip() + "\n"
