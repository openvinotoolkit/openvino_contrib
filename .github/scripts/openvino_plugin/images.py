# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Resolve pinned OpenVINO CI image references."""

import re
from pathlib import Path

from common.core import NAME_PATTERN, env_lines, required_env, write_github_file

_PATH = re.compile(r"[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)*")
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")


def _parse_images(values: list[str]) -> dict[str, str]:
    images = {}
    for entry in values:
        key, separator, value = entry.partition("=")
        if not separator or not NAME_PATTERN.fullmatch(key) or not _PATH.fullmatch(value):
            raise ValueError(f"invalid image entry: {entry}")
        if key in images:
            raise ValueError(f"duplicate image entry: {key}")
        images[key] = value
    return images


def main() -> None:
    tag_file = Path(required_env("CI_IMAGE_TAG_FILE"))
    tag = tag_file.read_text(encoding="utf-8").strip()
    if not _TAG.fullmatch(tag):
        raise ValueError(f"invalid Docker tag in {tag_file}: {tag}")
    registry = required_env("CI_IMAGE_REGISTRY").rstrip("/")
    images = {
        key: f"{registry}/{path}:{tag}"
        for key, path in _parse_images(env_lines("CI_IMAGES", required=True)).items()
    }
    for key, image in images.items():
        print(f"  {key} -> {image}")
    write_github_file("GITHUB_OUTPUT", images)
