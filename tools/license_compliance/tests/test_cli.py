# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from ov_contrib_license.cli import (
    EXIT_CONFIGURATION_ERROR,
    EXIT_DISCOVERY_ERROR,
    EXIT_POLICY_FAIL,
    EXIT_REVIEW_BLOCKING,
    EXIT_SUCCESS,
    main,
)
from ov_contrib_license.model import (
    Component,
    DistributionStatus,
    InventoryBuilder,
    Relationship,
    RepositoryInfo,
)
from ov_contrib_license.reporters import write_inventory

FIXTURES = Path(__file__).parent / "fixtures"
POLICY = Path(__file__).parents[1] / "policy"


class CliTests(unittest.TestCase):
    def test_inventory_writes_canonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "requirements.txt").write_text("example==1\n", encoding="utf-8")
            output = root / "result.json"

            exit_code = main(
                ["inventory", str(root), "--offline", "--output", str(output)]
            )
            data = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, EXIT_SUCCESS)
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["discoveries"][0]["path"], "requirements.txt")

    def test_partial_incremental_configuration_has_distinct_exit_code(self) -> None:
        exit_code = main(["inventory", ".", "--base-ref", "HEAD~1"])

        self.assertEqual(exit_code, EXIT_CONFIGURATION_ERROR)

    def test_malformed_provider_result_has_distinct_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = root / "ort-result.json"
            result.write_text("not JSON", encoding="utf-8")

            exit_code = main(
                [
                    "check",
                    str(root),
                    "--policy-dir",
                    str(POLICY),
                    "--ort-result",
                    str(result),
                    "--report",
                    str(root / "report.md"),
                ]
            )

        self.assertEqual(exit_code, EXIT_DISCOVERY_ERROR)

    def test_new_unknown_vendored_component_is_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "modules" / "sample" / "vendor" / "lib" / "source.cpp"
            source.parent.mkdir(parents=True)
            source.write_text("// imported source\n", encoding="utf-8")
            report = root / "report.md"

            exit_code = main(
                [
                    "check",
                    str(root),
                    "--policy-dir",
                    str(POLICY),
                    "--fail-on",
                    "FAIL",
                    "--report",
                    str(report),
                ]
            )

        self.assertEqual(exit_code, EXIT_POLICY_FAIL)

    def test_review_has_a_distinct_blocking_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cmake = root / "modules" / "sample" / "CMakeLists.txt"
            cmake.parent.mkdir(parents=True)
            cmake.write_text(
                "FetchContent_Declare(example URL https://example.org/example.tar.gz VERSION 1)\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "check",
                    str(root),
                    "--policy-dir",
                    str(POLICY),
                    "--fail-on",
                    "REVIEW",
                    "--report",
                    str(root / "report.md"),
                ]
            )

        self.assertEqual(exit_code, EXIT_REVIEW_BLOCKING)

    def test_artifact_command_returns_policy_fail_for_artifact_only_package(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.json"
            report = root / "artifact.md"
            builder = InventoryBuilder(RepositoryInfo(".", "revision"))
            builder.add_component(
                Component(
                    id="pkg:pypi/expected-runtime@1.0.0",
                    name="expected-runtime",
                    version="1.0.0",
                    declared_license="MIT",
                    relationships=(Relationship.RUNTIME_DEPENDENCY,),
                    distribution=DistributionStatus.DISTRIBUTED,
                )
            )
            write_inventory(builder.build(), source_path)

            exit_code = main(
                [
                    "artifact",
                    "check",
                    str(root),
                    "--source-inventory",
                    str(source_path),
                    "--syft-result",
                    str(FIXTURES / "syft-result.json"),
                    "--policy-dir",
                    str(POLICY),
                    "--report",
                    str(report),
                ]
            )
            rendered = report.read_text(encoding="utf-8")

        self.assertEqual(exit_code, EXIT_POLICY_FAIL)
        self.assertIn("ARTIFACT_UNDECLARED_COMPONENT", rendered)

    def test_toolchain_command_rejects_unpinned_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workflow = root / ".github" / "workflows" / "test.yml"
            workflow.parent.mkdir(parents=True)
            workflow.write_text(
                "steps:\n  - uses: actions/checkout@v4\n", encoding="utf-8"
            )

            exit_code = main(
                [
                    "toolchain",
                    "check",
                    str(root),
                    "--policy-dir",
                    str(POLICY),
                    "--report",
                    str(root / "toolchain.md"),
                ]
            )

        self.assertEqual(exit_code, EXIT_POLICY_FAIL)


if __name__ == "__main__":
    unittest.main()
