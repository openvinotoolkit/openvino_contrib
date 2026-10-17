# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ov_contrib_license.model import (
    Component,
    Decision,
    DistributionStatus,
    InventoryBuilder,
    Relationship,
    RepositoryInfo,
)
from ov_contrib_license.policy import (
    BaselineEntry,
    ExceptionRule,
    evaluate_inventory,
    load_policy,
)
from ov_contrib_license.policy.expressions import ExpressionError, normalize_expression

POLICY = Path(__file__).parents[1] / "policy"


class ExpressionTests(unittest.TestCase):
    def test_normalizes_nested_spdx_without_flattening_or(self) -> None:
        self.assertEqual(
            normalize_expression("MIT OR (Apache-2.0 AND BSD-3-Clause)"),
            "MIT OR Apache-2.0 AND BSD-3-Clause",
        )
        self.assertEqual(
            normalize_expression("GPL-2.0-only WITH Classpath-exception-2.0"),
            "GPL-2.0-only WITH Classpath-exception-2.0",
        )

    def test_invalid_expression_is_rejected(self) -> None:
        with self.assertRaises(ExpressionError):
            normalize_expression("MIT OR")


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_policy(POLICY)

    def _decision(
        self,
        license_expression: str | None,
        relationship: Relationship,
        distribution: DistributionStatus,
        *,
        component_id: str = "pkg:generic/example@1",
    ) -> tuple[Decision, str, bool]:
        builder = InventoryBuilder(RepositoryInfo(".", "revision"))
        builder.add_component(
            Component(
                id=component_id,
                name="example",
                version="1",
                module="sample",
                declared_license=license_expression,
                relationships=(relationship,),
                distribution=distribution,
            )
        )
        summary = evaluate_inventory(builder.build(), self.config)
        finding = next(
            item
            for item in summary.inventory.findings
            if item.code.startswith("POLICY_")
        )
        return finding.decision, finding.id, finding.suppressed

    def test_policy_table(self) -> None:
        cases = (
            (
                "MIT",
                Relationship.HEADER_ONLY,
                DistributionStatus.DISTRIBUTED,
                Decision.PASS,
            ),
            (
                "GPL-3.0-only",
                Relationship.STATIC_LINK,
                DistributionStatus.DISTRIBUTED,
                Decision.FAIL,
            ),
            (
                "GPL-3.0-only",
                Relationship.BUILD_TOOL,
                DistributionStatus.NOT_DISTRIBUTED,
                Decision.PASS,
            ),
            (
                "LGPL-3.0-only",
                Relationship.DYNAMIC_LINK,
                DistributionStatus.DISTRIBUTED,
                Decision.REVIEW,
            ),
            (
                None,
                Relationship.BUNDLED_BINARY,
                DistributionStatus.DISTRIBUTED,
                Decision.FAIL,
            ),
        )
        for license_expression, relationship, distribution, expected in cases:
            with self.subTest(license=license_expression, relationship=relationship):
                self.assertEqual(
                    self._decision(license_expression, relationship, distribution)[0],
                    expected,
                )

    def test_or_selects_an_allowed_branch_and_and_combines_obligations(self) -> None:
        builder = InventoryBuilder(RepositoryInfo(".", "revision"))
        builder.add_component(
            Component(
                id="pkg:generic/example@1",
                name="example",
                version="1",
                module="sample",
                declared_license="MIT OR GPL-3.0-only",
                relationships=(Relationship.BUNDLED_BINARY,),
                distribution=DistributionStatus.DISTRIBUTED,
            )
        )
        permissive_summary = evaluate_inventory(builder.build(), self.config)
        permissive_finding = next(
            item
            for item in permissive_summary.inventory.findings
            if item.code == "POLICY_COMPONENT_PASS"
        )
        combined_and = self._decision(
            "MIT AND GPL-3.0-only",
            Relationship.BUNDLED_BINARY,
            DistributionStatus.DISTRIBUTED,
        )[0]
        self.assertEqual(permissive_finding.decision, Decision.PASS)
        self.assertIn("rule=LIC-PERMISSIVE-001", permissive_finding.message)
        self.assertNotIn(
            "LIC-STRONG-COPYLEFT-DISTRIBUTED-001", permissive_finding.message
        )
        self.assertEqual(combined_and, Decision.FAIL)

    def test_expired_exception_is_a_failure(self) -> None:
        exception = ExceptionRule(
            id="TEST-EXCEPTION",
            component="pkg:generic/example@1",
            module="sample",
            relationships=(Relationship.STATIC_LINK,),
            distributions=(DistributionStatus.DISTRIBUTED,),
            rationale="fixture",
            approved_by=("legal",),
            expires=dt.date(2024, 1, 1),
        )
        config = replace(self.config, exceptions=(exception,))
        builder = InventoryBuilder(RepositoryInfo(".", "revision"))
        builder.add_component(
            Component(
                id=exception.component,
                name="example",
                version="1",
                module="sample",
                declared_license="GPL-3.0-only",
                relationships=(Relationship.STATIC_LINK,),
                distribution=DistributionStatus.DISTRIBUTED,
            )
        )
        summary = evaluate_inventory(builder.build(), config, today=dt.date(2026, 1, 1))
        self.assertIn(
            "POLICY_EXCEPTION_EXPIRED",
            {item.code for item in summary.inventory.findings},
        )
        self.assertEqual(summary.fail_count, 1)

    def test_exact_exception_applies_but_relationship_mismatch_does_not(self) -> None:
        exception = ExceptionRule(
            id="TEST-EXCEPTION",
            component="pkg:generic/example@1",
            module="sample",
            relationships=(Relationship.BUILD_TOOL,),
            distributions=(DistributionStatus.DISTRIBUTED,),
            rationale="fixture",
            approved_by=("legal",),
            expires=dt.date(2027, 1, 1),
        )
        self.config = replace(self.config, exceptions=(exception,))
        allowed = self._decision(
            "GPL-3.0-only",
            Relationship.BUILD_TOOL,
            DistributionStatus.DISTRIBUTED,
        )[0]

        builder = InventoryBuilder(RepositoryInfo(".", "revision"))
        builder.add_component(
            Component(
                id=exception.component,
                name="example",
                version="1",
                module="sample",
                declared_license="GPL-3.0-only",
                relationships=(Relationship.STATIC_LINK,),
                distribution=DistributionStatus.DISTRIBUTED,
            )
        )
        mismatch = evaluate_inventory(
            builder.build(), self.config, today=dt.date(2026, 1, 1)
        )

        self.assertEqual(allowed, Decision.PASS)
        self.assertIn(
            "POLICY_EXCEPTION_RELATIONSHIP_MISMATCH",
            {item.code for item in mismatch.inventory.findings},
        )
        self.assertEqual(mismatch.fail_count, 1)

    def test_exception_version_mismatch_is_explicit(self) -> None:
        exception = ExceptionRule(
            id="TEST-EXCEPTION",
            component="pkg:generic/example@1",
            module="sample",
            relationships=(Relationship.HEADER_ONLY,),
            distributions=(DistributionStatus.DISTRIBUTED,),
            rationale="fixture",
            approved_by=("legal",),
            expires=dt.date(2027, 1, 1),
        )
        config = replace(self.config, exceptions=(exception,))
        builder = InventoryBuilder(RepositoryInfo(".", "revision"))
        builder.add_component(
            Component(
                id="pkg:generic/example@2",
                name="example",
                version="2",
                module="sample",
                declared_license="MIT",
                relationships=(Relationship.HEADER_ONLY,),
                distribution=DistributionStatus.DISTRIBUTED,
            )
        )
        summary = evaluate_inventory(builder.build(), config, today=dt.date(2026, 1, 1))

        self.assertIn(
            "POLICY_EXCEPTION_VERSION_MISMATCH",
            {item.code for item in summary.inventory.findings},
        )
        self.assertEqual(summary.review_count, 1)

    def test_exception_module_mismatch_and_unused_exception_are_explicit(self) -> None:
        mismatch = ExceptionRule(
            id="MODULE-EXCEPTION",
            component="pkg:generic/example@1",
            module="different-module",
            relationships=(Relationship.HEADER_ONLY,),
            distributions=(DistributionStatus.DISTRIBUTED,),
            rationale="fixture",
            approved_by=("legal",),
            expires=dt.date(2027, 1, 1),
        )
        unused = replace(
            mismatch,
            id="UNUSED-EXCEPTION",
            component="pkg:generic/not-present@1",
        )
        config = replace(self.config, exceptions=(mismatch, unused))
        builder = InventoryBuilder(RepositoryInfo(".", "revision"))
        builder.add_component(
            Component(
                id="pkg:generic/example@1",
                name="example",
                version="1",
                module="sample",
                declared_license="MIT",
                relationships=(Relationship.HEADER_ONLY,),
                distribution=DistributionStatus.DISTRIBUTED,
            )
        )
        summary = evaluate_inventory(builder.build(), config, today=dt.date(2026, 1, 1))
        codes = {item.code for item in summary.inventory.findings}

        self.assertIn("POLICY_EXCEPTION_MODULE_MISMATCH", codes)
        self.assertIn("POLICY_EXCEPTION_UNUSED", codes)

    def test_exact_baseline_does_not_suppress_changed_version(self) -> None:
        _, fingerprint, _ = self._decision(
            None,
            Relationship.BUNDLED_BINARY,
            DistributionStatus.DISTRIBUTED,
        )
        config = replace(
            self.config,
            baseline=(BaselineEntry(fingerprint, "existing-before-enforcement"),),
        )
        self.config = config
        self.assertTrue(
            self._decision(
                None, Relationship.BUNDLED_BINARY, DistributionStatus.DISTRIBUTED
            )[2]
        )
        self.assertFalse(
            self._decision(
                None,
                Relationship.BUNDLED_BINARY,
                DistributionStatus.DISTRIBUTED,
                component_id="pkg:generic/example@2",
            )[2]
        )

    def test_exact_baseline_does_not_suppress_changed_relationship(self) -> None:
        _, fingerprint, _ = self._decision(
            None,
            Relationship.BUNDLED_BINARY,
            DistributionStatus.DISTRIBUTED,
        )
        self.config = replace(
            self.config,
            baseline=(BaselineEntry(fingerprint, "existing-before-enforcement"),),
        )

        self.assertFalse(
            self._decision(
                None,
                Relationship.VENDORED_SOURCE,
                DistributionStatus.DISTRIBUTED,
            )[2]
        )

    def test_invalid_exception_configuration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            policy = Path(temporary_directory) / "policy"
            shutil.copytree(POLICY, policy)
            (policy / "exceptions.yml").write_text(
                "exceptions:\n  - id: BROAD\n    component: pkg:generic/example\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_policy(policy)

    def test_invalid_baseline_and_toolchain_configuration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            policy = Path(temporary_directory) / "policy"
            shutil.copytree(POLICY, policy)
            (policy / "baseline.yml").write_text(
                "baseline:\n  - finding_fingerprint: sha256:not-a-digest\n"
                "    reason: fixture\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_policy(policy)

            shutil.copy(POLICY / "baseline.yml", policy / "baseline.yml")
            (policy / "toolchain.yml").write_text(
                "require_full_sha_for_github_actions: 'false'\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_policy(policy)


if __name__ == "__main__":
    unittest.main()
