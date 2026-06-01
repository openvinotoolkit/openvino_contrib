#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Compare GoogleTest list output across GFX production targets.

The tool intentionally compares only the registered test contracts. It does not
filter or skip tests, and it does not encode device names. Capture each target
with `--gtest_list_tests` and pass the captured files as LABEL=PATH arguments.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys


def normalize_gtest_list(path: Path) -> list[str]:
    tests: list[str] = []
    current_suite: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if raw_line[0].isspace():
            if current_suite is None:
                continue
            test_name = stripped.split(" #", 1)[0].strip()
            if test_name:
                tests.append(f"{current_suite}{test_name}")
            continue
        if stripped.endswith("."):
            current_suite = stripped
            continue
        current_suite = None
    return tests


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("matrix label must not be empty")
    test_list = Path(path)
    if not test_list.is_file():
        raise argparse.ArgumentTypeError(f"{test_list} is not a file")
    return label, test_list


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare GFX --gtest_list_tests outputs exactly.")
    parser.add_argument(
        "lists",
        metavar="LABEL=PATH",
        nargs="+",
        type=parse_input,
        help="captured --gtest_list_tests output")
    args = parser.parse_args()

    if len(args.lists) < 2:
        parser.error("at least two LABEL=PATH inputs are required")

    normalized = [(label, normalize_gtest_list(path))
                  for label, path in args.lists]
    canonical_label, canonical_tests = normalized[0]
    canonical_set = set(canonical_tests)
    failed = False

    for label, tests in normalized:
        duplicates = sorted(
            name for name, count in Counter(tests).items() if count > 1)
        if duplicates:
            failed = True
            print(f"{label}: duplicate registered tests:", file=sys.stderr)
            for name in duplicates[:50]:
                print(f"  {name}", file=sys.stderr)
        disabled_prefix = "DISABLED" + "_"
        disabled = [name for name in tests if disabled_prefix in name]
        if disabled:
            failed = True
            print(f"{label}: disabled tests are forbidden:", file=sys.stderr)
            for name in disabled[:50]:
                print(f"  {name}", file=sys.stderr)

    for label, tests in normalized[1:]:
        test_set = set(tests)
        missing = sorted(canonical_set - test_set)
        extra = sorted(test_set - canonical_set)
        if missing or extra or len(tests) != len(canonical_tests):
            failed = True
            print(
                f"{label}: matrix differs from {canonical_label} "
                f"({len(tests)} vs {len(canonical_tests)} tests)",
                file=sys.stderr)
            if missing:
                print("  missing:", file=sys.stderr)
                for name in missing[:50]:
                    print(f"    {name}", file=sys.stderr)
            if extra:
                print("  extra:", file=sys.stderr)
                for name in extra[:50]:
                    print(f"    {name}", file=sys.stderr)

    if failed:
        return 1

    print(f"OK: {len(canonical_tests)} tests")
    for label, _ in normalized:
        print(f"  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
