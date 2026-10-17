# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from ov_contrib_license.adapters import import_license_eye, import_ort, import_syft
from ov_contrib_license.model import (
    Component,
    DistributionStatus,
    InventoryBuilder,
    Obligation,
    Relationship,
    RepositoryInfo,
)
from ov_contrib_license.reconciliation import reconcile_artifact, reconcile_tpp
from ov_contrib_license.reporters import render_spdx

FIXTURES = Path(__file__).parent / "fixtures"


class AdapterTests(unittest.TestCase):
    def test_ort_maps_package_and_license_to_canonical_model(self) -> None:
        source = InventoryBuilder(RepositoryInfo(".", "revision")).build()
        inventory = import_ort(source, FIXTURES / "ort-result.json")

        self.assertEqual(inventory.components[0].id, "pkg:pypi/example-runtime@1.2.3")
        self.assertEqual(inventory.components[0].declared_license, "MIT OR Apache-2.0")
        self.assertIn("ort", {item.name for item in inventory.providers})

    def test_license_eye_failure_and_ignore_are_not_silent_passes(self) -> None:
        source = InventoryBuilder(RepositoryInfo(".", "revision")).build()
        inventory = import_license_eye(source, FIXTURES / "license-eye-result.json")

        self.assertEqual(
            {item.code for item in inventory.findings},
            {"HEADER_IGNORED_PATHS", "HEADER_INVALID"},
        )

    def test_syft_maps_artifact_packages_as_distributed(self) -> None:
        inventory = import_syft(FIXTURES / "syft-result.json")

        self.assertEqual(len(inventory.components), 2)
        self.assertTrue(
            all(
                item.distribution is DistributionStatus.DISTRIBUTED
                for item in inventory.components
            )
        )


class ReconciliationTests(unittest.TestCase):
    def test_artifact_only_component_fails_but_absent_build_tool_does_not(self) -> None:
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
        builder.add_component(
            Component(
                id="pkg:generic/build-helper@1",
                name="build-helper",
                version="1",
                declared_license="MIT",
                relationships=(Relationship.BUILD_TOOL,),
                distribution=DistributionStatus.NOT_DISTRIBUTED,
            )
        )
        result = reconcile_artifact(
            builder.build(), import_syft(FIXTURES / "syft-result.json")
        )
        codes = [item.code for item in result.findings]

        self.assertEqual(codes.count("ARTIFACT_UNDECLARED_COMPONENT"), 1)
        self.assertNotIn("ARTIFACT_MISSING_EXPECTED_COMPONENT", codes)

    def test_tpp_missing_distributed_attribution_is_a_failure(self) -> None:
        builder = InventoryBuilder(RepositoryInfo(".", "revision"))
        builder.add_component(
            Component(
                id="pkg:generic/new-component@1",
                name="new-component",
                version="1",
                declared_license="MIT",
                relationships=(Relationship.BUNDLED_BINARY,),
                distribution=DistributionStatus.DISTRIBUTED,
                obligations=(Obligation.RETAIN_LICENSE_TEXT,),
            )
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            tpp = Path(temporary_directory) / "third-party-programs.txt"
            tpp.write_text("Third Party Programs\n", encoding="utf-8")
            result = reconcile_tpp(builder.build(), tpp)

        self.assertIn("TPP_MISSING_COMPONENT", {item.code for item in result.findings})

    def test_spdx_output_is_deterministic_and_contains_purl(self) -> None:
        inventory = import_syft(FIXTURES / "syft-result.json")
        first = render_spdx(inventory)
        second = render_spdx(inventory)
        data = json.loads(first)

        self.assertEqual(first, second)
        self.assertEqual(data["spdxVersion"], "SPDX-2.3")
        self.assertEqual(
            data["packages"][0]["externalRefs"][0]["referenceType"], "purl"
        )


if __name__ == "__main__":
    unittest.main()
