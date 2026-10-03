# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import re
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlparse

from ov_contrib_license.model import (
    Component,
    Confidence,
    Discovery,
    Evidence,
    EvidenceKind,
    InventoryBuilder,
    Relationship,
)
from ov_contrib_license.model.types import normalize_details

from .common import owning_module

DOWNLOAD_FILE_SUFFIXES = frozenset(
    {".sh", ".bash", ".zsh", ".py", ".cmake", ".yml", ".yaml"}
)
COMMAND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "git-clone",
        re.compile(
            r"\bgit\s+clone(?:\s+--?[\w-]+(?:=\S+|\s+\S+)?)*\s+([^\s;&|]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "git-submodule",
        re.compile(
            r"\bgit\s+submodule\s+(?:add|update)(?:\s+--?[\w-]+)*\s+([^\s;&|]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "curl",
        re.compile(r"\bcurl\b[^\n]*?((?:git\+)?https?://[^\s'\";&|)]+)", re.IGNORECASE),
    ),
    (
        "wget",
        re.compile(r"\bwget\b[^\n]*?((?:git\+)?https?://[^\s'\";&|)]+)", re.IGNORECASE),
    ),
    (
        "pip-install",
        re.compile(
            r"\b(?:python\s+-m\s+)?pip\s+install\b[^\n]*?((?:git\+)?https?://[^\s'\";&|)]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "npm-install",
        re.compile(
            r"\bnpm\s+(?:install|i)\b[^\n]*?(github:[^\s'\";&|)]+)", re.IGNORECASE
        ),
    ),
)
PYTHON_COMMAND_CALLS = frozenset(
    {
        "os.popen",
        "os.system",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "subprocess.run",
    }
)


@dataclass(frozen=True)
class DownloadCandidate:
    mechanism: str
    url: str
    line: int


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _literal_command(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values: list[str] = []
        for element in node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(
                element.value, str
            ):
                return None
            values.append(element.value)
        return " ".join(values)
    return None


def _python_command_text(text: str) -> tuple[tuple[str, int], ...]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(textwrap.dedent(text))
    except SyntaxError:
        return ()
    commands: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or _call_name(node.func) not in PYTHON_COMMAND_CALLS
            or not node.args
        ):
            continue
        command = _literal_command(node.args[0])
        if command:
            commands.append((command, node.lineno))
    return tuple(commands)


def _strip_comment(line: str) -> str:
    single_quoted = False
    double_quoted = False
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
        elif char == "\\" and not single_quoted:
            escaped = True
        elif char == "'" and not double_quoted:
            single_quoted = not single_quoted
        elif char == '"' and not single_quoted:
            double_quoted = not double_quoted
        elif char == "#" and not single_quoted and not double_quoted:
            return line[:index]
    return line


def extract_downloads(
    text: str, *, python: bool = False
) -> tuple[DownloadCandidate, ...]:
    command_lines = (
        _python_command_text(text)
        if python
        else tuple(
            (line, index)
            for index, line in enumerate(text.replace("\\\n", " ").splitlines(), 1)
        )
    )
    candidates: set[DownloadCandidate] = set()
    for command_text, line_number in command_lines:
        executable = command_text if python else _strip_comment(command_text)
        for mechanism, pattern in COMMAND_PATTERNS:
            for match in pattern.finditer(executable):
                url = match.group(1).strip("'\"")
                if url.startswith(
                    ("http://", "https://", "git+http", "git@", "ssh://", "github:")
                ):
                    candidates.add(DownloadCandidate(mechanism, url, line_number))
    return tuple(
        sorted(candidates, key=lambda item: (item.line, item.mechanism, item.url))
    )


def _normalized_remote(url: str) -> str:
    if url.startswith("github:"):
        return f"https://github.com/{url.removeprefix('github:')}"
    return url.removeprefix("git+").split("#", 1)[0]


def _download_identity(url: str) -> tuple[str, str, str | None]:
    normalized = _normalized_remote(url)
    revision = None
    if "@" in normalized and not normalized.startswith("git@"):
        before, candidate = normalized.rsplit("@", 1)
        if "/" not in candidate:
            normalized, revision = before, candidate
    clean = normalized.removesuffix(".git").rstrip("/")
    github = re.match(
        r"(?:https?://github\.com/|git@github\.com:)([^/]+)/([^/]+)$", clean
    )
    if github:
        owner, repository = github.groups()
        component_id = (
            f"pkg:github/{quote(owner, safe='')}/{quote(repository, safe='')}"
        )
        if revision:
            component_id += f"@{quote(revision, safe='.-_')}"
        return component_id, repository, revision
    parsed = urlparse(clean)
    name = PurePosixPath(parsed.path).name or parsed.netloc or "external-download"
    digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()
    return f"url:sha256:{digest}", name, revision


def _eligible_file(path: str) -> bool:
    pure = PurePosixPath(path)
    return (
        pure.suffix.lower() in DOWNLOAD_FILE_SUFFIXES - {".yml", ".yaml"}
        or (
            pure.parts[:2] == (".github", "workflows")
            and pure.suffix.lower() in {".yml", ".yaml"}
        )
        or pure.name.startswith("Dockerfile")
        or pure.name == "CMakeLists.txt"
    )


def discover_downloads(
    builder: InventoryBuilder, root: Path, files: tuple[str, ...]
) -> None:
    for path in files:
        if not _eligible_file(path):
            continue
        try:
            text = (root / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for candidate in extract_downloads(
            text, python=PurePosixPath(path).suffix.lower() == ".py"
        ):
            component_id, name, revision = _download_identity(candidate.url)
            details = normalize_details(
                {"line": candidate.line, "mechanism": candidate.mechanism}
            )
            evidence = Evidence(
                kind=EvidenceKind.DOWNLOAD_URL,
                source="executable-download-discovery",
                path=path,
                value=candidate.url,
                confidence=Confidence.MEDIUM,
                details=details,
            )
            builder.add_discovery(
                Discovery(
                    kind="download",
                    path=path,
                    module=owning_module(path),
                    details=normalize_details(
                        {
                            "line": candidate.line,
                            "mechanism": candidate.mechanism,
                            "url": candidate.url,
                        }
                    ),
                )
            )
            builder.add_component(
                Component(
                    id=component_id,
                    name=name,
                    version=revision,
                    module=owning_module(path),
                    paths=(path,),
                    relationships=(Relationship.FETCHED_AT_BUILD,),
                    evidence=(evidence,),
                    details=normalize_details(
                        {"source_url": _normalized_remote(candidate.url)}
                    ),
                )
            )
