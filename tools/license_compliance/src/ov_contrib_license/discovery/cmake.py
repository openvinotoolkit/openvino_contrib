# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from ov_contrib_license.model import (
    Component,
    Confidence,
    Decision,
    Discovery,
    DistributionStatus,
    Evidence,
    EvidenceKind,
    Finding,
    InventoryBuilder,
    Relationship,
    Severity,
)
from ov_contrib_license.model.types import normalize_details

from .common import owning_module

CM_COMMANDS = frozenset(
    {
        "fetchcontent_declare",
        "fetchcontent_makeavailable",
        "fetchcontent_populate",
        "externalproject_add",
        "cpmaddpackage",
        "find_package",
        "find_library",
        "add_subdirectory",
    }
)
REMOTE_COMMANDS = frozenset(
    {
        "fetchcontent_declare",
        "fetchcontent_populate",
        "externalproject_add",
        "cpmaddpackage",
    }
)
KEYWORDS = frozenset(
    {
        "NAME",
        "GIT_REPOSITORY",
        "GIT_TAG",
        "URL",
        "URL_HASH",
        "GITHUB_REPOSITORY",
        "VERSION",
        "DOWNLOAD_COMMAND",
    }
)


@dataclass(frozen=True)
class CMakeCommand:
    name: str
    arguments: tuple[str, ...]
    line: int


def _bracket_delimiter(text: str, start: int) -> tuple[str, int] | None:
    match = re.match(r"\[(=*)\[", text[start:])
    if match is None:
        return None
    equals = match.group(1)
    return f"]{equals}]", len(match.group(0))


def _skip_space_and_comments(text: str, start: int) -> int:
    index = start
    while index < len(text):
        if text[index].isspace():
            index += 1
        elif text[index] == "#":
            bracket = _bracket_delimiter(text, index + 1)
            if bracket:
                delimiter, prefix_length = bracket
                end = text.find(delimiter, index + 1 + prefix_length)
                index = len(text) if end < 0 else end + len(delimiter)
            else:
                end = text.find("\n", index)
                index = len(text) if end < 0 else end + 1
        else:
            break
    return index


def _command_body(text: str, start: int) -> tuple[str, int] | None:
    depth = 1
    index = start
    body: list[str] = []
    quoted = False
    escaped = False
    while index < len(text):
        char = text[index]
        if quoted:
            body.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            index += 1
            continue
        if char == '"':
            quoted = True
            body.append(char)
            index += 1
            continue
        bracket = _bracket_delimiter(text, index) if char == "[" else None
        if bracket:
            delimiter, prefix_length = bracket
            end = text.find(delimiter, index + prefix_length)
            if end < 0:
                return None
            body.append(text[index + prefix_length : end])
            index = end + len(delimiter)
            continue
        if char == "#":
            bracket_comment = _bracket_delimiter(text, index + 1)
            if bracket_comment:
                delimiter, prefix_length = bracket_comment
                end = text.find(delimiter, index + 1 + prefix_length)
                if end < 0:
                    return None
                body.append(" ")
                index = end + len(delimiter)
            else:
                end = text.find("\n", index)
                body.append("\n")
                index = len(text) if end < 0 else end + 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return "".join(body), index + 1
        body.append(char)
        index += 1
    return None


def _tokenize(body: str) -> tuple[str, ...]:
    tokens: list[str] = []
    token: list[str] = []
    quoted = False
    escaped = False
    for char in body:
        if quoted:
            if escaped:
                token.append(char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            else:
                token.append(char)
        elif char == '"':
            quoted = True
        elif char.isspace() or char == ";":
            if token:
                tokens.append("".join(token))
                token = []
        else:
            token.append(char)
    if escaped:
        token.append("\\")
    if token:
        tokens.append("".join(token))
    return tuple(tokens)


def parse_cmake(text: str) -> tuple[CMakeCommand, ...]:
    """Parse CMake command calls while preserving unresolved arguments."""
    commands: list[CMakeCommand] = []
    index = 0
    while index < len(text):
        index = _skip_space_and_comments(text, index)
        match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[index:])
        if match is None:
            index += 1
            continue
        name = match.group(0)
        command_start = index
        index += len(name)
        index = _skip_space_and_comments(text, index)
        if index >= len(text) or text[index] != "(":
            continue
        parsed = _command_body(text, index + 1)
        if parsed is None:
            break
        body, index = parsed
        if name.lower() in CM_COMMANDS:
            commands.append(
                CMakeCommand(
                    name.lower(),
                    _tokenize(body),
                    text.count("\n", 0, command_start) + 1,
                )
            )
    return tuple(commands)


def _keyword_values(arguments: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for index, argument in enumerate(arguments[:-1]):
        keyword = argument.upper()
        if keyword in KEYWORDS and arguments[index + 1].upper() not in KEYWORDS:
            values.setdefault(keyword, arguments[index + 1])
    return values


def _github_url(url: str) -> tuple[str, str] | None:
    normalized = url.strip().removesuffix(".git").rstrip("/")
    match = re.match(
        r"(?:https?://github\.com/|git@github\.com:)([^/]+)/([^/]+)$", normalized
    )
    return (match.group(1), match.group(2)) if match else None


def _component_id(
    name: str, source_url: str | None, revision: str | None, path: str
) -> str:
    if source_url:
        github = _github_url(source_url)
        if github and revision:
            owner = quote(github[0], safe="")
            repository = quote(github[1], safe="")
            return f"pkg:github/{owner}/{repository}@{quote(revision, safe='.-_')}"
        normalized = source_url.removesuffix(".git").rstrip("/")
        return f"vcs:{normalized}" + (f"@{revision}" if revision else "")
    escaped_name = quote(name.lower(), safe=".-_")
    if revision:
        return f"pkg:generic/{escaped_name}@{quote(revision, safe='.-_')}"
    return f"local:cmake/{path}#{escaped_name}"


def _is_unresolved(value: str | None) -> bool:
    return value is None or "${" in value or "$<" in value


def _remote_component(
    command: CMakeCommand, path: str
) -> tuple[Component, Finding | None] | None:
    if not command.arguments:
        return None
    values = _keyword_values(command.arguments)
    name = values.get("NAME", command.arguments[0])
    source_url = values.get("GIT_REPOSITORY") or values.get("URL")
    revision = values.get("GIT_TAG") or values.get("VERSION") or values.get("URL_HASH")

    if command.name == "cpmaddpackage" and command.arguments[0].lower().startswith(
        "gh:"
    ):
        shorthand = command.arguments[0][3:]
        repository, separator, shorthand_revision = shorthand.partition("@")
        source_url = f"https://github.com/{repository}"
        revision = shorthand_revision if separator else revision
        name = values.get("NAME", repository.rsplit("/", 1)[-1])
    elif values.get("GITHUB_REPOSITORY"):
        source_url = f"https://github.com/{values['GITHUB_REPOSITORY']}"

    if not source_url and command.name != "cpmaddpackage":
        return None

    unresolved_expression = revision if _is_unresolved(revision) else None
    resolved_revision = None if unresolved_expression else revision
    evidence_kind = {
        "externalproject_add": EvidenceKind.CMAKE_EXTERNAL_PROJECT,
        "cpmaddpackage": EvidenceKind.CMAKE_CPM,
    }.get(command.name, EvidenceKind.CMAKE_FETCHCONTENT)
    details = {
        "line": command.line,
        "mechanism": command.name,
        "revision": resolved_revision,
        "source_url": source_url,
        "unresolved_revision_expression": unresolved_expression,
    }
    evidence = Evidence(
        kind=evidence_kind,
        source="cmake-static-discovery",
        path=path,
        value=source_url or name,
        confidence=Confidence.HIGH if source_url or name else Confidence.LOW,
        details=normalize_details(details),
    )
    component_id = _component_id(name, source_url, resolved_revision, path)
    component = Component(
        id=component_id,
        name=name,
        version=resolved_revision,
        module=owning_module(path),
        paths=(path,),
        relationships=(Relationship.FETCHED_AT_BUILD,),
        evidence=(evidence,),
        distribution=DistributionStatus.UNKNOWN,
        details=normalize_details({"source_url": source_url}),
    )
    finding = None
    if source_url and _is_unresolved(revision):
        finding = Finding.create(
            code="DISCOVERY_CMAKE_UNRESOLVED_REVISION",
            severity=Severity.WARNING,
            decision=Decision.REVIEW,
            component_id=component_id,
            message=f"CMake dependency {name!r} does not have a statically resolved revision.",
            evidence=(evidence,),
            remediation=(
                "Pin an exact revision or provide resolution evidence in a later provider stage.",
            ),
            fingerprint_values=(unresolved_expression or "missing",),
        )
    return component, finding


def _lookup_component(command: CMakeCommand, path: str) -> Component | None:
    if not command.arguments:
        return None
    if command.name == "find_package":
        name = command.arguments[0]
        evidence_kind = EvidenceKind.CMAKE_FIND_PACKAGE
    elif command.name == "find_library":
        upper_arguments = tuple(argument.upper() for argument in command.arguments)
        if "NAMES" in upper_arguments:
            names_index = upper_arguments.index("NAMES")
            name = (
                command.arguments[names_index + 1]
                if names_index + 1 < len(command.arguments)
                else command.arguments[0]
            )
        else:
            name = (
                command.arguments[1]
                if len(command.arguments) > 1
                else command.arguments[0]
            )
        evidence_kind = EvidenceKind.CMAKE_FIND_LIBRARY
    else:
        return None
    evidence = Evidence(
        kind=evidence_kind,
        source="cmake-static-discovery",
        path=path,
        value=name,
        confidence=Confidence.MEDIUM,
        details=normalize_details({"line": command.line, "mechanism": command.name}),
    )
    return Component(
        id=f"pkg:generic/{quote(name.lower(), safe='.-_')}",
        name=name,
        module=owning_module(path),
        paths=(path,),
        relationships=(Relationship.UNKNOWN,),
        evidence=(evidence,),
    )


def discover_cmake(
    builder: InventoryBuilder, root: Path, files: tuple[str, ...]
) -> None:
    for path in files:
        pure_path = PurePosixPath(path)
        if pure_path.name != "CMakeLists.txt" and pure_path.suffix.lower() != ".cmake":
            continue
        try:
            text = (root / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for command in parse_cmake(text):
            details = {"command": command.name, "line": command.line}
            if command.arguments:
                details["subject"] = command.arguments[0]
            builder.add_discovery(
                Discovery(
                    kind="cmake",
                    path=path,
                    module=owning_module(path),
                    ecosystem="cmake",
                    details=normalize_details(details),
                )
            )
            if command.name in REMOTE_COMMANDS:
                result = _remote_component(command, path)
                if result:
                    component, finding = result
                    builder.add_component(component)
                    if finding:
                        builder.add_finding(finding)
            else:
                component = _lookup_component(command, path)
                if component:
                    builder.add_component(component)
