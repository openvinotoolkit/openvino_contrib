#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify that CODEOWNERS module rules point at directories that still exist.

Every rule whose path is '/modules/<name>/...' must reference a directory that
exists under modules/. This catches ownership rules left behind after a module
is renamed or removed. A module without a dedicated owner is NOT an error: the
top-level '* @...-maintainers' rule already covers it.
"""

import os
import re
import sys

MODULES_DIR = "modules"
CODEOWNERS_FILE = os.path.join(".github", "CODEOWNERS")


def main():
    modules = {
        d for d in os.listdir(MODULES_DIR)
        if os.path.isdir(os.path.join(MODULES_DIR, d))
    }

    errors = []
    with open(CODEOWNERS_FILE, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # First whitespace-separated token is the path pattern.
            pattern = line.split()[0]
            match = re.match(r"/modules/([^/]+)/", pattern)
            if match and match.group(1) not in modules:
                errors.append(
                    f"{CODEOWNERS_FILE}:{lineno}: rule '{pattern}' points at "
                    f"'modules/{match.group(1)}/' which does not exist"
                )

    if errors:
        print("Stale CODEOWNERS rules detected:\n", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            f"\nRemove or fix the stale rules in {CODEOWNERS_FILE} so every "
            f"'/modules/<name>/' rule points at an existing module.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: all CODEOWNERS module rules point at existing modules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
