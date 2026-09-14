#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Capture and compare GoogleTest list output across GFX production targets.

The tool intentionally compares only the registered test contracts. It does not
filter or skip tests, and it does not encode device names. Capture each target
with `--gtest_list_tests` and pass the captured files as LABEL=PATH arguments.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import subprocess
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


def parse_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"expected LABEL=DIR, got {value!r}")
    label, root = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("matrix root label must not be empty")
    root_path = Path(root)
    if not root_path.is_dir():
        raise argparse.ArgumentTypeError(f"{root_path} is not a directory")
    return label, root_path


def parse_capture(value: str) -> tuple[str, Path, Path]:
    parts = value.split("::", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected LABEL::EXECUTABLE::OUTPUT, got {value!r}")
    label, executable, output = parts
    if not label:
        raise argparse.ArgumentTypeError("capture label must not be empty")
    binary = Path(executable)
    if not binary.is_file():
        raise argparse.ArgumentTypeError(f"{binary} is not an executable file")
    return label, binary, Path(output)


def capture_gtest_list(label: str, binary: Path, output: Path) -> tuple[str, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(binary), "--gtest_list_tests"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError(
            f"{label}: --gtest_list_tests failed for {binary} "
            f"with exit code {completed.returncode}")
    output.write_text(completed.stdout, encoding="utf-8")
    return label, output


def validate_single_list(label: str, tests: list[str]) -> bool:
    failed = False
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
    return failed


def compare_gtest_lists(
        lists: list[tuple[str, Path]], check_only: bool,
        group_name: str | None = None) -> bool:
    normalized = [(label, normalize_gtest_list(path))
                  for label, path in lists]
    canonical_label, canonical_tests = normalized[0]
    canonical_set = set(canonical_tests)
    failed = False

    for label, tests in normalized:
        failed = validate_single_list(label, tests) or failed

    if not check_only:
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
        return True

    if check_only:
        if group_name:
            print(f"OK: {group_name}: captured GFX test registrations")
        else:
            print("OK: captured GFX test registrations")
        for label, tests in normalized:
            print(f"  {label}: {len(tests)} tests")
    else:
        if group_name:
            print(f"OK: {group_name}: {len(canonical_tests)} tests")
        else:
            print(f"OK: {len(canonical_tests)} tests")
        for label, _ in normalized:
            print(f"  {label}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare GFX --gtest_list_tests outputs exactly.")
    parser.add_argument(
        "--capture",
        metavar="LABEL::EXECUTABLE::OUTPUT",
        action="append",
        type=parse_capture,
        default=[],
        help="run EXECUTABLE --gtest_list_tests and store OUTPUT")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="only reject duplicates/disabled tests; do not compare lists")
    parser.add_argument(
        "--root",
        metavar="LABEL=DIR",
        action="append",
        type=parse_root,
        default=[],
        help="compare TARGET.txt files from captured matrix root directories")
    parser.add_argument(
        "--target",
        metavar="GTEST_BINARY",
        action="append",
        default=[],
        help="test binary name to compare when --root is used")
    parser.add_argument(
        "lists",
        metavar="LABEL=PATH",
        nargs="*",
        type=parse_input,
        help="captured --gtest_list_tests output")
    args = parser.parse_args()

    if args.root:
        if args.capture or args.lists:
            parser.error(
                "--root mode cannot be combined with --capture or LABEL=PATH inputs")
        if not args.target:
            parser.error("--target is required with --root")
        failed = False
        for target in args.target:
            target_lists: list[tuple[str, Path]] = []
            for label, root in args.root:
                matrix_file = root / f"{target}.txt"
                if not matrix_file.is_file():
                    failed = True
                    print(
                        f"{label}:{target}: missing matrix file {matrix_file}",
                        file=sys.stderr)
                    continue
                target_lists.append((label, matrix_file))
            if not target_lists:
                continue
            if len(target_lists) < 2 and not args.check_only:
                failed = True
                print(
                    f"{target}: at least two matrix roots are required",
                    file=sys.stderr)
                continue
            failed = compare_gtest_lists(
                target_lists, args.check_only, target) or failed
        return 1 if failed else 0

    lists = list(args.lists)
    for label, binary, output in args.capture:
        try:
            lists.append(capture_gtest_list(label, binary, output))
        except RuntimeError as ex:
            print(str(ex), file=sys.stderr)
            return 1

    if not lists:
        parser.error("at least one LABEL=PATH input or --capture is required")
    if len(lists) < 2 and not args.check_only:
        parser.error("at least two LABEL=PATH inputs are required")

    return 1 if compare_gtest_lists(lists, args.check_only) else 0


if __name__ == "__main__":
    raise SystemExit(main())
