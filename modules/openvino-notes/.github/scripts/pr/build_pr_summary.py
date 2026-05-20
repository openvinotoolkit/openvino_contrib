#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def sum_junit_metrics(files: list[Path]) -> dict[str, int]:
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for file in files:
        try:
            root = ET.parse(file).getroot()
        except ET.ParseError:
            continue

        if root.tag == "testsuites":
            suites = list(root.findall("testsuite"))
        elif root.tag == "testsuite":
            suites = [root]
        else:
            suites = []

        for suite in suites:
            for key in totals:
                totals[key] += int(suite.attrib.get(key, "0"))

    return totals


def count_lint_issues(files: list[Path]) -> int:
    count = 0
    for file in files:
        try:
            root = ET.parse(file).getroot()
        except ET.ParseError:
            continue
        count += len(root.findall(".//issue"))
    return count


def count_ktlint_issues(files: list[Path]) -> int:
    count = 0
    for file in files:
        try:
            root = ET.parse(file).getroot()
        except ET.ParseError:
            continue
        count += len(root.findall(".//error"))
    return count


def count_sarif_results(files: list[Path]) -> int:
    count = 0
    for file in files:
        try:
            data = json.loads(file.read_text())
        except Exception:
            continue
        for run in data.get("runs", []):
            count += len(run.get("results", []))
    return count


def aggregate_line_coverage(files: list[Path]) -> tuple[int, int]:
    covered = 0
    missed = 0
    for file in files:
        try:
            root = ET.parse(file).getroot()
        except ET.ParseError:
            continue

        for counter in root.findall("./counter"):
            if counter.attrib.get("type") == "LINE":
                covered += int(counter.attrib.get("covered", "0"))
                missed += int(counter.attrib.get("missed", "0"))
                break
    return covered, missed


def format_test_line(label: str, metrics: dict[str, int]) -> str:
    total_failures = metrics["failures"] + metrics["errors"]
    status = "PASS" if total_failures == 0 else "FAIL"
    return (
        f"- {label}: {status} "
        f"({metrics['tests']} tests, {total_failures} failed, {metrics['skipped']} skipped)"
    )


def main() -> int:
    if len(sys.argv) != 7:
        print(
            "usage: build_pr_summary.py <artifacts_dir> <output_file> <pr_number> <head_sha> <run_url> <ci_conclusion>",
            file=sys.stderr,
        )
        return 2

    artifacts_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    unit_xml = sorted(artifacts_dir.glob("**/build/test-results/**/*.xml"))
    instrumentation_xml = sorted(artifacts_dir.glob("**/instrumentation-results.xml"))
    lint_xml = sorted(artifacts_dir.glob("**/build/reports/lint-results-*.xml"))
    ktlint_xml = sorted(artifacts_dir.glob("**/build/reports/ktlint/**/*.xml"))
    kover_xml = sorted(artifacts_dir.glob("**/build/reports/kover/**/*.xml"))
    detekt_sarif = sorted(artifacts_dir.glob("**/build/reports/detekt/*.sarif"))
    gitleaks_sarif = sorted(artifacts_dir.glob("**/build/reports/gitleaks/*.sarif"))

    unit_metrics = sum_junit_metrics(unit_xml)
    instrumentation_metrics = sum_junit_metrics(instrumentation_xml)
    lint_issues = count_lint_issues(lint_xml)
    ktlint_issues = count_ktlint_issues(ktlint_xml)
    detekt_findings = count_sarif_results(detekt_sarif)
    gitleaks_findings = count_sarif_results(gitleaks_sarif)
    covered, missed = aggregate_line_coverage(kover_xml)
    total_lines = covered + missed
    coverage = (covered / total_lines * 100.0) if total_lines else None

    pr_number = sys.argv[3]
    head_sha = sys.argv[4]
    run_url = sys.argv[5]
    ci_conclusion = sys.argv[6]

    lines = [
        "## CI Quality Overview",
        "",
        f"- CI conclusion: `{ci_conclusion or 'unknown'}`",
        f"- Head SHA: `{head_sha[:12]}`" if head_sha else None,
        f"- PR: `#{pr_number}`" if pr_number else None,
        f"- Workflow run: [view run]({run_url})" if run_url else None,
        "",
        "### Tests",
        format_test_line("Unit", unit_metrics) if unit_xml else "- Unit: no XML reports found",
        (
            format_test_line("Instrumentation", instrumentation_metrics)
            if instrumentation_xml
            else "- Instrumentation: no XML reports found"
        ),
        "",
        "### Static Analysis",
        f"- Android Lint: {lint_issues} issue(s) from XML reports" if lint_xml else "- Android Lint: no XML reports found",
        f"- ktlint: {ktlint_issues} issue(s) from XML reports" if ktlint_xml else "- ktlint: no XML reports found",
        f"- detekt: {detekt_findings} finding(s) uploaded via SARIF" if detekt_sarif else "- detekt: no SARIF reports found",
        f"- gitleaks: {gitleaks_findings} finding(s) uploaded via SARIF" if gitleaks_sarif else "- gitleaks: no SARIF reports found",
        "- CodeQL: see the Code Scanning section in this PR",
        "",
        "### Coverage",
        (
            f"- Line coverage: {coverage:.2f}% ({covered} covered / {total_lines} total)"
            if coverage is not None
            else "- Line coverage: no Kover XML reports found"
        ),
    ]

    output_file.write_text("\n".join(line for line in lines if line is not None) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
