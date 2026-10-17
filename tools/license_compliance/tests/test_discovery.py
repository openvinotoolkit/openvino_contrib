# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ov_contrib_license.discovery.downloads import extract_downloads
from ov_contrib_license.discovery.github_actions import extract_action_references
from ov_contrib_license.discovery.repository import (
    DiscoveryOptions,
    RepositoryDiscovery,
)


class DiscoveryTests(unittest.TestCase):
    def test_manifest_ownership_and_vendored_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "modules" / "sample" / "requirements-test.txt"
            vendored = (
                root / "modules" / "sample" / "third_party" / "lib" / "source.cpp"
            )
            manifest.parent.mkdir(parents=True)
            vendored.parent.mkdir(parents=True)
            manifest.write_text("example==1.0\n", encoding="utf-8")
            vendored.write_text("// source\n", encoding="utf-8")

            inventory = RepositoryDiscovery(root).run()

        manifests = [item for item in inventory.discoveries if item.kind == "manifest"]
        self.assertEqual(manifests[0].ecosystem, "python")
        self.assertEqual(manifests[0].module, "sample")
        self.assertIn(
            "local:modules/sample/third_party",
            {item.id for item in inventory.components},
        )

    def test_virtual_environment_contents_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            vendored = root / "venv" / "lib" / "package" / "vendor" / "source.py"
            vendored.parent.mkdir(parents=True)
            vendored.write_text("# installed package\n", encoding="utf-8")

            inventory = RepositoryDiscovery(root).run()

        self.assertFalse(inventory.components)
        self.assertFalse(inventory.discoveries)

    def test_foreign_copyright_cluster_is_a_vendored_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            imported = root / "modules" / "sample" / "imported"
            imported.mkdir(parents=True)
            for index in range(3):
                (imported / f"source_{index}.cpp").write_text(
                    "// Copyright (C) Example Authors\n",
                    encoding="utf-8",
                )

            inventory = RepositoryDiscovery(root).run()

        candidate = next(
            item
            for item in inventory.components
            if item.id == "local:modules/sample/imported"
        )
        self.assertEqual(candidate.relationships[0].value, "VENDORED_SOURCE")
        self.assertEqual(
            dict(candidate.evidence[0].details)["heuristic"],
            "foreign-copyright-cluster",
        )

    def test_downloads_ignore_comments_and_plain_python_strings(self) -> None:
        shell = """
        # git clone https://example.invalid/commented
        git clone https://github.com/example/real.git
        curl -L https://example.org/archive.tar.gz
        """
        python = """
        import subprocess
        documentation = "wget https://example.invalid/not-executed"
        subprocess.run(["pip", "install", "git+https://github.com/example/package.git@abc123"])
        """

        shell_candidates = extract_downloads(shell)
        python_candidates = extract_downloads(python, python=True)

        self.assertEqual(len(shell_candidates), 2)
        self.assertNotIn("commented", " ".join(item.url for item in shell_candidates))
        self.assertEqual(len(python_candidates), 1)
        self.assertEqual(python_candidates[0].mechanism, "pip-install")

    def test_action_references_and_pinning_findings(self) -> None:
        full_sha = "a" * 40
        workflow = f"""
        steps:
          - uses: actions/checkout@{full_sha} # pinned
          - uses: actions/setup-python@v5
          - uses: ./local-action
          - uses: docker://alpine@sha256:{"b" * 64}
        """
        self.assertEqual(len(extract_action_references(workflow)), 4)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / ".github" / "workflows" / "test.yml"
            path.parent.mkdir(parents=True)
            path.write_text(workflow, encoding="utf-8")
            inventory = RepositoryDiscovery(root).run()

        self.assertEqual(
            len(
                [item for item in inventory.discoveries if item.kind == "github-action"]
            ),
            4,
        )
        self.assertEqual(len(inventory.findings), 1)
        self.assertIn("setup-python@v5", inventory.findings[0].message)

    def test_git_submodule_metadata_and_index_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / ".gitmodules").write_text(
                '[submodule "library"]\npath = modules/sample/vendor/library\nurl = https://example.org/library.git\n',
                encoding="utf-8",
            )
            with patch(
                "ov_contrib_license.discovery.submodules._index_revision",
                return_value="c" * 40,
            ):
                inventory = RepositoryDiscovery(root).run()

        submodule = next(
            item for item in inventory.discoveries if item.kind == "git-submodule"
        )
        self.assertEqual(submodule.module, "sample")
        self.assertEqual(inventory.components[0].version, "c" * 40)
        self.assertFalse(inventory.findings)

    def test_incremental_mode_expands_to_module_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for module in ("a", "b"):
                path = root / "modules" / module / "requirements.txt"
                path.parent.mkdir(parents=True)
                path.write_text("example==1\n", encoding="utf-8")
            self._git(root, "init")
            self._git(root, "config", "user.email", "tests@example.org")
            self._git(root, "config", "user.name", "Tests")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "initial")
            base = self._git(root, "rev-parse", "HEAD").strip()
            (root / "modules" / "a" / "README.md").write_text(
                "change\n", encoding="utf-8"
            )
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "change a")
            head = self._git(root, "rev-parse", "HEAD").strip()

            inventory = RepositoryDiscovery(
                root,
                DiscoveryOptions(base_ref=base, head_ref=head),
            ).run()

        self.assertEqual(inventory.repository.impacted_scopes, ("module:a",))
        self.assertEqual({item.module for item in inventory.discoveries}, {"a"})

    def test_incremental_mode_with_no_changes_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "requirements.txt").write_text("example==1\n", encoding="utf-8")
            self._git(root, "init")
            self._git(root, "config", "user.email", "tests@example.org")
            self._git(root, "config", "user.name", "Tests")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "initial")
            revision = self._git(root, "rev-parse", "HEAD").strip()

            inventory = RepositoryDiscovery(
                root,
                DiscoveryOptions(base_ref=revision, head_ref=revision),
            ).run()

        self.assertFalse(inventory.discoveries)
        self.assertFalse(inventory.components)

    def test_policy_change_triggers_full_repository_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "modules" / "a" / "requirements.txt"
            policy = root / "tools" / "license_compliance" / "policy" / "rules.yml"
            manifest.parent.mkdir(parents=True)
            policy.parent.mkdir(parents=True)
            manifest.write_text("example==1\n", encoding="utf-8")
            policy.write_text("rules: []\n", encoding="utf-8")
            self._git(root, "init")
            self._git(root, "config", "user.email", "tests@example.org")
            self._git(root, "config", "user.name", "Tests")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "initial")
            base = self._git(root, "rev-parse", "HEAD").strip()
            policy.write_text("rules:\n  - id: changed\n", encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "change policy")
            head = self._git(root, "rev-parse", "HEAD").strip()

            inventory = RepositoryDiscovery(
                root,
                DiscoveryOptions(base_ref=base, head_ref=head),
            ).run()

        self.assertIn("repository-tooling", inventory.repository.impacted_scopes)
        self.assertIn(
            "modules/a/requirements.txt", {item.path for item in inventory.discoveries}
        )

    @staticmethod
    def _git(root: Path, *arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout


if __name__ == "__main__":
    unittest.main()
