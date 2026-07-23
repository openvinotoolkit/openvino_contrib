#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify that 'category:' labels stay in sync with the modules/ directory.

Source of truth is .github/labeler.yml (path -> label mapping). This check
fails when a module has no category label, when a label points at a module
that no longer exists, or when labels.yml and labeler.yml drift apart. It is
not a generator: label names remain curated by hand, this only guards them.
"""

import os
import re
import sys

import yaml

MODULES_DIR = "modules"
LABELS_FILE = os.path.join(".github", "labels.yml")
LABELER_FILE = os.path.join(".github", "labeler.yml")

# Path-based category labels that intentionally do not map to a single module.
NON_MODULE_CATEGORIES = {"category: build", "category: CI"}


def flatten_patterns(node):
    """Yield every glob string from a labeler.yml value (handles any:/all: dicts)."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from flatten_patterns(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from flatten_patterns(value)


def main():
    errors = []

    modules = sorted(
        d for d in os.listdir(MODULES_DIR)
        if os.path.isdir(os.path.join(MODULES_DIR, d))
    )

    with open(LABELER_FILE, encoding="utf-8") as f:
        labeler = yaml.safe_load(f) or {}
    with open(LABELS_FILE, encoding="utf-8") as f:
        labels = yaml.safe_load(f) or []

    label_names = {entry["name"] for entry in labels}

    # Modules referenced by any 'modules/<name>/...' pattern in labeler.yml.
    referenced_modules = {}  # module dir -> set of labels mapping to it
    for label, patterns in labeler.items():
        for pattern in flatten_patterns(patterns):
            match = re.match(r"modules/([^/]+)/", pattern)
            if match:
                referenced_modules.setdefault(match.group(1), set()).add(label)

    # 1. Every module directory must be covered by a category label.
    for module in modules:
        if module not in referenced_modules:
            errors.append(
                f"module 'modules/{module}' has no category label mapping in "
                f"{LABELER_FILE} (add the module path there and a matching label "
                f"in {LABELS_FILE})"
            )

    # 2. Every module referenced in labeler.yml must actually exist.
    for module in sorted(referenced_modules):
        if module not in modules:
            labels_using = ", ".join(sorted(referenced_modules[module]))
            errors.append(
                f"{LABELER_FILE} maps [{labels_using}] to 'modules/{module}/**' "
                f"but that module directory does not exist"
            )

    # 3. category: labels must be defined in BOTH files.
    labeler_categories = {name for name in labeler if name.startswith("category:")}
    labels_categories = {name for name in label_names if name.startswith("category:")}
    for name in sorted(labeler_categories - labels_categories):
        errors.append(f"'{name}' is used in {LABELER_FILE} but missing from {LABELS_FILE}")
    for name in sorted(labels_categories - labeler_categories):
        errors.append(f"'{name}' is defined in {LABELS_FILE} but not used in {LABELER_FILE}")

    if errors:
        print("Label/module drift detected:\n", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            f"\nFix {LABELS_FILE} and {LABELER_FILE} so every module under "
            f"{MODULES_DIR}/ has exactly one category label.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {len(modules)} modules and category labels are in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
