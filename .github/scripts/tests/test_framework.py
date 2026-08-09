# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPTS))

import ci  # noqa: E402
from common import ci_workspace, container_job, detect_module_changes  # noqa: E402
from common.core import PROXY_VARIABLES  # noqa: E402
from openvino_plugin import images  # noqa: E402


class SelectionTests(unittest.TestCase):
    def test_module_path_selection(self):
        patterns = [".github/scripts/common/**", "modules/alpha/**"]
        self.assertTrue(detect_module_changes.selected(patterns, ["modules/alpha/source.cpp"]))
        self.assertTrue(detect_module_changes.selected(patterns, [".github/scripts/common/core.py"]))
        self.assertFalse(detect_module_changes.selected(patterns, ["modules/beta/source.cpp"]))
        self.assertTrue(detect_module_changes.selected(patterns, [], select_all=True))
        with self.assertRaises(ValueError):
            detect_module_changes.selected([], [])

    def test_event_revisions(self):
        pull = {
            "pull_request": {
                "base": {"sha": "base", "ref": "releases/2026/3"},
                "head": {"sha": "head"},
            }
        }
        merge = {
            "merge_group": {
                "base_sha": "base",
                "head_sha": "head",
                "base_ref": "refs/heads/master",
            }
        }
        self.assertEqual(
            detect_module_changes._event_revisions("pull_request", pull),
            ("base", "head", "releases/2026/3"),
        )
        self.assertEqual(
            detect_module_changes._event_revisions("merge_group", merge),
            ("base", "head", "master"),
        )


class ArchitectureTests(unittest.TestCase):
    def test_module_workflow_owns_nvidia_policy(self):
        overall = (REPOSITORY / ".github/workflows/overall_status.yml").read_text(encoding="utf-8")
        workflow = (REPOSITORY / ".github/workflows/module_nvidia_plugin.yml").read_text(encoding="utf-8")
        lifecycle = (REPOSITORY / ".github/workflows/module_openvino_plugin.yml").read_text(encoding="utf-8")
        self.assertIn("./.github/workflows/module_nvidia_plugin.yml", overall)
        self.assertLess(len(workflow.splitlines()), 80)
        self.assertEqual(list((REPOSITORY / ".github/ci").rglob("*.json")), [])
        for value in ("ENABLE_NVIDIA", "RTX-4090", "ov_nvidia_func_tests", "nvidia/cuda"):
            with self.subTest(value=value):
                self.assertIn(value, workflow)
                self.assertNotIn(value, overall)
                self.assertNotIn(value, lifecycle)

    def test_common_layer_has_no_module_policy(self):
        forbidden = ("nvidia", "cuda", "cmake", "gtest", "register_plugin")
        for path in (REPOSITORY / ".github/scripts/common").glob("*.py"):
            content = path.read_text(encoding="utf-8").casefold()
            for term in forbidden:
                with self.subTest(path=path.name, term=term):
                    self.assertNotIn(term, content)

    def test_public_command_surface_stays_small(self):
        self.assertLessEqual(len(ci.COMMANDS), 8)

    def test_image_paths_are_validated(self):
        self.assertEqual(images._parse_images(["build=ci/build"]), {"build": "ci/build"})
        with self.assertRaises(ValueError):
            images._parse_images(["build=../image"])
        with self.assertRaises(ValueError):
            images._parse_images(["build=ci/build", "build=ci/other"])


class ContainerTests(unittest.TestCase):
    def test_restricted_container_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "openvino_contrib"
            runner = checkout / ".github/scripts/module/test.py"
            runner.parent.mkdir(parents=True)
            runner.touch()
            job_dir = root / "contrib-ci-tests-module-work"
            job_dir.mkdir()
            with (
                mock.patch.dict(
                    os.environ,
                    {"GITHUB_WORKSPACE": str(root), "RUNNER_TEMP": str(root)},
                    clear=False,
                ),
                mock.patch.object(container_job, "run") as run,
            ):
                container_job.run_tests(
                    "python@sha256:digest",
                    "test@sha256:digest",
                    ["vendor.example/gpu=all"],
                    "contrib-ci-test-1-module",
                    checkout,
                    job_dir,
                    ".github/scripts/module/test.py",
                    "pr",
                )
            command = run.call_args_list[1].args[0]
            self.assertIn("--network=none", command)
            self.assertIn("--read-only", command)
            self.assertIn("--cap-drop=ALL", command)
            self.assertIn("--device=vendor.example/gpu=all", command)
            self.assertNotIn("/var/run/docker.sock", " ".join(command))
            for variable in PROXY_VARIABLES:
                self.assertIn(f"--env={variable}=", command)
            self.assertEqual(command[-1], "/ci/.github/scripts/module/test.py")

    def test_device_input_rejects_docker_options(self):
        self.assertEqual(container_job._device_selectors("/dev/dri"), ["/dev/dri"])
        for value in ("--privileged", "invalid device"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                container_job._device_selectors(value)


class WorkspaceTests(unittest.TestCase):
    def test_result_upload_rejects_symlinks_and_oversize_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            target = results / "result.xml"
            target.write_bytes(b"12345")
            self.assertEqual(ci_workspace.validate_results(results), 5)
            with self.assertRaisesRegex(ValueError, "exceed"):
                ci_workspace.validate_results(results, max_bytes=4)
            (results / "link.xml").symlink_to(target.name)
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                ci_workspace.validate_results(results)

    def test_workspace_exports_namespaced_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "github-env"
            environment = {
                "RUNNER_TEMP": str(root),
                "GITHUB_ENV": str(output),
                "GITHUB_RUN_ID": "123",
                "GITHUB_RUN_ATTEMPT": "2",
                "GITHUB_REPOSITORY_ID": "456",
                "CI_MODULE_NAME": "module",
                "CI_PRESET": "pr",
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                job_dir = ci_workspace.prepare_tests()
                self.assertEqual(os.environ["JOB_DIR"], str(job_dir))
            self.assertIn("TEST_CONTAINER=contrib-ci-test-456-123-2-module-pr", output.read_text())
