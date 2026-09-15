# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from ov_contrib_license.cli import EXIT_POLICY_FAIL, EXIT_SUCCESS, main

FIXTURES = Path(__file__).parent / "fixtures"
REPOSITORY = FIXTURES / "repository"
POLICY = Path(__file__).parents[1] / "policy"


class EndToEndTests(unittest.TestCase):
    def test_repository_to_policy_artifact_tpp_and_spdx(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            source_inventory = output / "source.json"
            source_report = output / "source.md"
            artifact_inventory = output / "artifact.json"
            artifact_report = output / "artifact.md"
            tpp_report = output / "tpp.md"
            spdx = output / "sbom.spdx.json"

            source_exit = main(
                [
                    "check",
                    str(REPOSITORY),
                    "--policy-dir",
                    str(POLICY),
                    "--ort-result",
                    str(REPOSITORY / "ort-result.json"),
                    "--inventory-output",
                    str(source_inventory),
                    "--report",
                    str(source_report),
                ]
            )
            source = json.loads(source_inventory.read_text(encoding="utf-8"))
            decisions = {
                item["component_id"]: item["decision"]
                for item in source["findings"]
                if item["code"].startswith("POLICY_COMPONENT_")
            }

            toolchain_exit = main(
                [
                    "toolchain",
                    "check",
                    str(REPOSITORY),
                    "--policy-dir",
                    str(POLICY),
                    "--report",
                    str(output / "toolchain.md"),
                ]
            )
            artifact_exit = main(
                [
                    "artifact",
                    "check",
                    str(output),
                    "--source-inventory",
                    str(source_inventory),
                    "--syft-result",
                    str(REPOSITORY / "syft-result.json"),
                    "--policy-dir",
                    str(POLICY),
                    "--inventory-output",
                    str(artifact_inventory),
                    "--report",
                    str(artifact_report),
                ]
            )
            tpp_exit = main(
                [
                    "tpp",
                    "check",
                    str(REPOSITORY),
                    "--source-inventory",
                    str(artifact_inventory),
                    "--policy-dir",
                    str(POLICY),
                    "--report",
                    str(tpp_report),
                ]
            )
            sbom_exit = main(
                [
                    "sbom",
                    str(REPOSITORY),
                    "--source-inventory",
                    str(artifact_inventory),
                    "--output",
                    str(spdx),
                ]
            )

            tpp_text = tpp_report.read_text(encoding="utf-8")
            spdx_data = json.loads(spdx.read_text(encoding="utf-8"))

        self.assertEqual(source_exit, EXIT_POLICY_FAIL)
        self.assertEqual(decisions["pkg:github/example/permissive@1.0.0"], "PASS")
        self.assertEqual(decisions["local:modules/b/vendor"], "FAIL")
        self.assertEqual(toolchain_exit, EXIT_POLICY_FAIL)
        self.assertEqual(artifact_exit, EXIT_POLICY_FAIL)
        self.assertEqual(tpp_exit, EXIT_POLICY_FAIL)
        self.assertIn("TPP_MISSING_COMPONENT", tpp_text)
        self.assertEqual(sbom_exit, EXIT_SUCCESS)
        self.assertEqual(spdx_data["spdxVersion"], "SPDX-2.3")


if __name__ == "__main__":
    unittest.main()
