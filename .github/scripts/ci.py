#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Single entry point for module-independent contrib CI commands."""

import subprocess
import sys
from importlib import import_module

COMMANDS = {
    "build-plugin": "openvino_plugin.build.main",
    "clang-format": "common.run_clang_format.main",
    "cleanup": "common.ci_workspace.cleanup_workspace",
    "detect-changes": "common.detect_module_changes.main",
    "prepare-build": "common.ci_workspace.prepare_build",
    "resolve-images": "openvino_plugin.images.main",
    "test-module": "common.container_job.main",
    "validate-results": "common.ci_workspace.validate_results_command",
}


def main(command: str | None = None) -> None:
    command = command or (sys.argv[1] if len(sys.argv) == 2 else "")
    if command not in COMMANDS:
        available = "\n  ".join(sorted(COMMANDS))
        raise SystemExit(f"usage: {sys.argv[0]} <command>\n\ncommands:\n  {available}")
    try:
        module_name, function_name = COMMANDS[command].rsplit(".", 1)
        getattr(import_module(module_name), function_name)()
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"::error::{error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
