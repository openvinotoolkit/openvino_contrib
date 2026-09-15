# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import unittest

from ov_contrib_license.model import (
    Component,
    Confidence,
    Decision,
    Evidence,
    EvidenceKind,
    Finding,
    InventoryBuilder,
    Relationship,
    RepositoryInfo,
    Severity,
)
from ov_contrib_license.reporters import render_inventory


class ModelAndReporterTests(unittest.TestCase):
    def test_component_evidence_is_merged_deterministically(self) -> None:
        builder = InventoryBuilder(RepositoryInfo(".", "revision"))
        for path in ("modules/z/CMakeLists.txt", "modules/a/CMakeLists.txt"):
            evidence = Evidence(
                EvidenceKind.CMAKE_FIND_PACKAGE,
                "test",
                path,
                "Example",
                Confidence.MEDIUM,
            )
            builder.add_component(
                Component(
                    id="pkg:generic/example",
                    name="Example",
                    module=path.split("/")[1],
                    paths=(path,),
                    relationships=(Relationship.UNKNOWN,),
                    evidence=(evidence,),
                )
            )

        component = builder.build().components[0]

        self.assertIsNone(component.module)
        self.assertEqual(
            component.paths, ("modules/a/CMakeLists.txt", "modules/z/CMakeLists.txt")
        )
        self.assertEqual(dict(component.details)["modules"], "a,z")

    def test_fingerprint_ignores_line_metadata_but_not_component(self) -> None:
        first = Evidence(
            EvidenceKind.CMAKE_FETCHCONTENT,
            "test",
            "CMakeLists.txt",
            "example",
            details=(("line", "10"),),
        )
        second = Evidence(
            EvidenceKind.CMAKE_FETCHCONTENT,
            "test",
            "CMakeLists.txt",
            "example",
            details=(("line", "20"),),
        )
        finding_one = Finding.create(
            code="TEST",
            severity=Severity.WARNING,
            decision=Decision.REVIEW,
            component_id="pkg:generic/a@1",
            message="one",
            evidence=(first,),
        )
        finding_two = Finding.create(
            code="TEST",
            severity=Severity.WARNING,
            decision=Decision.REVIEW,
            component_id="pkg:generic/a@1",
            message="two",
            evidence=(second,),
        )
        changed_component = Finding.create(
            code="TEST",
            severity=Severity.WARNING,
            decision=Decision.REVIEW,
            component_id="pkg:generic/a@2",
            message="one",
            evidence=(first,),
        )

        self.assertEqual(finding_one.id, finding_two.id)
        self.assertNotEqual(finding_one.id, changed_component.id)

    def test_json_and_yaml_are_deterministic(self) -> None:
        inventory = InventoryBuilder(RepositoryInfo(".", "revision")).build()

        first_json = render_inventory(inventory)
        second_json = render_inventory(inventory)
        yaml = render_inventory(inventory, "yaml")

        self.assertEqual(first_json, second_json)
        self.assertEqual(json.loads(first_json)["schema_version"], 1)
        self.assertIn("schema_version: 1", yaml)
        self.assertNotIn("timestamp", first_json)


if __name__ == "__main__":
    unittest.main()
