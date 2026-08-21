# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Select one module workflow for the current event."""

import fnmatch
import json
import os
from pathlib import Path

from .core import env_lines, required_env, run, write_github_values


def _git(*arguments: str, input_text: str | None = None) -> str:
    result = run(
        ["git", *arguments],
        check=False,
        capture=True,
        input_text=input_text,
        quiet=True,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def _event_revisions(event_name: str, event: dict) -> tuple[str, str, str]:
    if event_name == "pull_request":
        pull = event["pull_request"]
        return pull["base"]["sha"], pull["head"]["sha"], pull["base"]["ref"]
    if event_name == "merge_group":
        group = event["merge_group"]
        head = group["head_sha"]
        return (
            group.get("base_sha") or _git("rev-parse", f"{head}^"),
            head,
            group.get("base_ref", "").removeprefix("refs/heads/"),
        )
    if event_name == "push":
        base = event.get("before", "")
        if not base or set(base) == {"0"}:
            base = _git("hash-object", "-t", "tree", "--stdin", input_text="")
        return base, event["after"], event.get("ref", "").removeprefix("refs/heads/")

    head = event.get("after") or os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD")
    ref = os.environ.get("GITHUB_REF_NAME", "master")
    branch = ref if ref == "master" or ref.startswith("releases/") else "master"
    return _git("rev-parse", f"{head}^"), head, branch


def selected(patterns: list[str], changed: list[str], select_all: bool = False) -> bool:
    if not patterns:
        raise ValueError("CI_MODULE_PATHS is empty")
    return select_all or any(fnmatch.fnmatchcase(path, pattern) for path in changed for pattern in patterns)


def main() -> None:
    try:
        event = json.loads(Path(required_env("GITHUB_EVENT_PATH")).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid event JSON: {error}") from None

    event_name = required_env("GITHUB_EVENT_NAME")
    try:
        base, head, branch = _event_revisions(event_name, event)
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid {event_name} event: {error}") from None
    all_modules = event_name in {"schedule", "workflow_dispatch"}
    changed = [] if all_modules else _git("diff", "--name-only", "-z", "--no-renames", base, head).split("\0")
    changed = [path for path in changed if path]
    module_selected = selected(env_lines("CI_MODULE_PATHS", required=True), changed, all_modules)

    print(f"Range: {base}..{head} (base branch: {branch or '<unknown>'})")
    print(f"Selected: {'yes' if module_selected else 'no'}")
    write_github_values(
        Path(required_env("GITHUB_OUTPUT")),
        {
            "selected": module_selected,
            "base_sha": base,
            "head_sha": head,
            "base_ref": branch,
        },
    )
