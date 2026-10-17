# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Decision(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class DistributionStatus(str, Enum):
    DISTRIBUTED = "DISTRIBUTED"
    NOT_DISTRIBUTED = "NOT_DISTRIBUTED"
    UNKNOWN = "UNKNOWN"


class Relationship(str, Enum):
    VENDORED_SOURCE = "VENDORED_SOURCE"
    COPIED_SNIPPET = "COPIED_SNIPPET"
    HEADER_ONLY = "HEADER_ONLY"
    STATIC_LINK = "STATIC_LINK"
    DYNAMIC_LINK = "DYNAMIC_LINK"
    RUNTIME_DEPENDENCY = "RUNTIME_DEPENDENCY"
    RUNTIME_EXTERNAL = "RUNTIME_EXTERNAL"
    BUILD_TOOL = "BUILD_TOOL"
    CODE_GENERATOR = "CODE_GENERATOR"
    TEST_ONLY = "TEST_ONLY"
    DEV_ONLY = "DEV_ONLY"
    FETCHED_AT_BUILD = "FETCHED_AT_BUILD"
    FETCHED_AT_RUNTIME = "FETCHED_AT_RUNTIME"
    BUNDLED_BINARY = "BUNDLED_BINARY"
    DATASET = "DATASET"
    MODEL = "MODEL"
    DOCUMENTATION_ASSET = "DOCUMENTATION_ASSET"
    UNKNOWN = "UNKNOWN"


class Obligation(str, Enum):
    RETAIN_COPYRIGHT = "RETAIN_COPYRIGHT"
    RETAIN_LICENSE_TEXT = "RETAIN_LICENSE_TEXT"
    RETAIN_NOTICE = "RETAIN_NOTICE"
    MARK_MODIFICATIONS = "MARK_MODIFICATIONS"
    PROVIDE_SOURCE = "PROVIDE_SOURCE"
    PROVIDE_SOURCE_OFFER = "PROVIDE_SOURCE_OFFER"
    SEPARATE_LICENSE_TERMS = "SEPARATE_LICENSE_TERMS"
    NO_TRADEDOWN = "NO_TRADEDOWN"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class EvidenceKind(str, Enum):
    SPDX_HEADER = "SPDX_HEADER"
    LICENSE_FILE = "LICENSE_FILE"
    NOTICE_FILE = "NOTICE_FILE"
    PACKAGE_METADATA = "PACKAGE_METADATA"
    ORT = "ORT"
    CMAKE_FETCHCONTENT = "CMAKE_FETCHCONTENT"
    CMAKE_EXTERNAL_PROJECT = "CMAKE_EXTERNAL_PROJECT"
    CMAKE_FIND_PACKAGE = "CMAKE_FIND_PACKAGE"
    CMAKE_FIND_LIBRARY = "CMAKE_FIND_LIBRARY"
    CMAKE_ADD_SUBDIRECTORY = "CMAKE_ADD_SUBDIRECTORY"
    CMAKE_CPM = "CMAKE_CPM"
    GIT_SUBMODULE = "GIT_SUBMODULE"
    DOWNLOAD_URL = "DOWNLOAD_URL"
    VENDORED_TREE = "VENDORED_TREE"
    GITHUB_ACTION = "GITHUB_ACTION"
    ARTIFACT_SBOM = "ARTIFACT_SBOM"
    MANUAL_CURATED = "MANUAL_CURATED"


Details = tuple[tuple[str, str], ...]


def normalize_details(
    details: Mapping[str, object] | Iterable[tuple[str, object]] | None = None,
) -> Details:
    """Return immutable, deterministic string metadata."""
    if details is None:
        return ()
    items = details.items() if isinstance(details, Mapping) else details
    return tuple(
        sorted((str(key), str(value)) for key, value in items if value is not None)
    )


@dataclass(frozen=True)
class Evidence:
    kind: EvidenceKind
    source: str
    path: str | None = None
    value: str | None = None
    confidence: Confidence = Confidence.MEDIUM
    details: Details = field(default_factory=tuple)

    def sort_key(self) -> tuple[object, ...]:
        return (
            self.kind.value,
            self.path or "",
            self.value or "",
            self.source,
            self.details,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "source": self.source,
            "path": self.path,
            "value": self.value,
            "confidence": self.confidence.value,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class Component:
    id: str
    name: str
    version: str | None = None
    module: str | None = None
    paths: tuple[str, ...] = field(default_factory=tuple)
    declared_license: str | None = None
    detected_licenses: tuple[str, ...] = field(default_factory=tuple)
    relationships: tuple[Relationship, ...] = (Relationship.UNKNOWN,)
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    distribution: DistributionStatus = DistributionStatus.UNKNOWN
    obligations: tuple[Obligation, ...] = field(default_factory=tuple)
    details: Details = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "module": self.module,
            "paths": sorted(self.paths),
            "declared_license": self.declared_license,
            "detected_licenses": sorted(self.detected_licenses),
            "relationships": sorted(item.value for item in self.relationships),
            "evidence": [
                item.to_dict() for item in sorted(self.evidence, key=Evidence.sort_key)
            ],
            "distribution": self.distribution.value,
            "obligations": sorted(item.value for item in self.obligations),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class Discovery:
    kind: str
    path: str
    module: str | None = None
    ecosystem: str | None = None
    details: Details = field(default_factory=tuple)

    def sort_key(self) -> tuple[object, ...]:
        return (
            self.kind,
            self.path,
            self.module or "",
            self.ecosystem or "",
            self.details,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "module": self.module,
            "ecosystem": self.ecosystem,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class Finding:
    id: str
    code: str
    severity: Severity
    decision: Decision
    component_id: str | None
    message: str
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    remediation: tuple[str, ...] = field(default_factory=tuple)
    suppressed: bool = False
    suppression_reason: str | None = None

    @classmethod
    def create(
        cls,
        *,
        code: str,
        severity: Severity,
        decision: Decision,
        component_id: str | None,
        message: str,
        evidence: Iterable[Evidence] = (),
        remediation: Iterable[str] = (),
        fingerprint_values: Iterable[str] = (),
    ) -> Finding:
        normalized_evidence = tuple(sorted(set(evidence), key=Evidence.sort_key))
        stable_values = [code, component_id or ""]
        stable_values.extend(str(item) for item in fingerprint_values)
        for item in normalized_evidence:
            stable_values.extend((item.kind.value, item.path or "", item.value or ""))
        digest = hashlib.sha256("\x00".join(stable_values).encode("utf-8")).hexdigest()
        return cls(
            id=f"sha256:{digest}",
            code=code,
            severity=severity,
            decision=decision,
            component_id=component_id,
            message=message,
            evidence=normalized_evidence,
            remediation=tuple(remediation),
        )

    def suppress(self, reason: str) -> Finding:
        return replace(self, suppressed=True, suppression_reason=reason)

    def sort_key(self) -> tuple[str, str, str]:
        return (self.decision.value, self.code, self.id)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "code": self.code,
            "severity": self.severity.value,
            "decision": self.decision.value,
            "component_id": self.component_id,
            "message": self.message,
            "evidence": [
                item.to_dict() for item in sorted(self.evidence, key=Evidence.sort_key)
            ],
            "remediation": list(self.remediation),
            "suppressed": self.suppressed,
            "suppression_reason": self.suppression_reason,
        }
