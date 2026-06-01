#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Validate native/adaptor GoogleTest source parity for GFX backends."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys


TEST_REGISTRATION_RE = re.compile(
    r"\bTEST(?:_[FP])?\s*\(\s*"
    r"([A-Za-z_][A-Za-z0-9_:]*)\s*,\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\)",
    re.MULTILINE | re.DOTALL,
)


def remove_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//.*", "", text)


def parse_source(path: Path) -> list[str]:
    text = remove_comments(path.read_text(encoding="utf-8", errors="replace"))
    return [f"{suite}.{name}" for suite, name in TEST_REGISTRATION_RE.findall(text)]


def parse_paths(value: str) -> list[Path]:
    return [Path(item) for item in value.split(",") if item]


def collect_tests(paths: list[Path]) -> list[str]:
    tests: list[str] = []
    missing: list[Path] = []
    for path in paths:
        if not path.is_file():
            missing.append(path)
            continue
        tests.extend(parse_source(path))
    if missing:
        for path in missing:
            print(f"missing source: {path}", file=sys.stderr)
        raise ValueError("missing source files")
    return tests


def parse_group(value: str) -> tuple[str, list[Path], list[Path]]:
    parts = value.split("::", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected LABEL::NATIVE_SOURCES::ADAPTER_SOURCES, got {value!r}")
    label, native_sources, adapter_sources = parts
    if not label:
        raise argparse.ArgumentTypeError("group label must not be empty")
    native_paths = parse_paths(native_sources)
    adapter_paths = parse_paths(adapter_sources)
    if not native_paths or not adapter_paths:
        raise argparse.ArgumentTypeError(
            f"{label}: native and adapter source lists must be non-empty")
    return label, native_paths, adapter_paths


def report_duplicates(label: str, kind: str, tests: list[str]) -> bool:
    duplicates = sorted(name for name, count in Counter(tests).items() if count > 1)
    if not duplicates:
        return False
    print(f"{label}: duplicate {kind} GoogleTest registrations:", file=sys.stderr)
    for name in duplicates:
        print(f"  {name}", file=sys.stderr)
    return True


def validate_group(label: str, native_paths: list[Path], adapter_paths: list[Path]) -> bool:
    native_tests = collect_tests(native_paths)
    adapter_tests = collect_tests(adapter_paths)
    failed = False
    if not native_tests:
        print(f"{label}: native source list registers no tests", file=sys.stderr)
        failed = True
    if not adapter_tests:
        print(f"{label}: adapter source list registers no tests", file=sys.stderr)
        failed = True
    failed = report_duplicates(label, "native", native_tests) or failed
    failed = report_duplicates(label, "adapter", adapter_tests) or failed

    native_set = set(native_tests)
    adapter_set = set(adapter_tests)
    missing = sorted(native_set - adapter_set)
    extra = sorted(adapter_set - native_set)
    if missing or extra or len(native_tests) != len(adapter_tests):
        failed = True
        print(
            f"{label}: adapter test source contract differs "
            f"({len(adapter_tests)} vs {len(native_tests)} tests)",
            file=sys.stderr,
        )
        if missing:
            print("  missing in adapter:", file=sys.stderr)
            for name in missing:
                print(f"    {name}", file=sys.stderr)
        if extra:
            print("  extra in adapter:", file=sys.stderr)
            for name in extra:
                print(f"    {name}", file=sys.stderr)

    if not failed:
        print(f"OK: {label} {len(native_tests)} tests")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare native and unavailable-adapter GFX test sources.")
    parser.add_argument(
        "groups",
        metavar="LABEL::NATIVE_SOURCES::ADAPTER_SOURCES",
        nargs="+",
        type=parse_group,
        help="comma-separated source groups to compare")
    args = parser.parse_args()

    failed = False
    for label, native_paths, adapter_paths in args.groups:
        try:
            failed = validate_group(label, native_paths, adapter_paths) or failed
        except ValueError:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
