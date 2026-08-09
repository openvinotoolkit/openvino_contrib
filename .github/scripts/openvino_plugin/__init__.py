# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared lifecycle for OpenVINO C++ plugin modules."""

import os
from pathlib import Path

from common.core import run


def source_setupvars(setupvars: Path) -> None:
    """Import the environment produced by OpenVINO setupvars.sh."""
    if not setupvars.is_file():
        raise ValueError(f"setupvars.sh not found: {setupvars}")
    result = run(
        ["bash", "-c", 'set -a && source "$1" >/dev/null 2>&1 && env -0', "bash", setupvars],
        capture=True,
    )
    os.environ.update(entry.split("=", 1) for entry in result.stdout.split("\0") if "=" in entry)
