# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import os
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from common import artifact, download_artifact  # noqa: E402
from openvino_plugin import build, tests  # noqa: E402


class ArtifactTests(unittest.TestCase):
    def test_safe_tar_extract_and_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "artifact.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                member = tarfile.TarInfo("dir/file.txt")
                member.size = 5
                output.addfile(member, io.BytesIO(b"12345"))
            artifact.extract_archive(archive, root / "out")
            self.assertEqual((root / "out/dir/file.txt").read_bytes(), b"12345")
            with self.assertRaisesRegex(ValueError, "expands beyond"):
                artifact.extract_archive(archive, root / "small", max_extracted_bytes=4)

    def test_tar_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "artifact.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                member = tarfile.TarInfo("../escape")
                member.size = 1
                output.addfile(member, io.BytesIO(b"x"))
            with self.assertRaises(tarfile.FilterError):
                artifact.extract_archive(archive, root / "out")

    def test_github_zip_must_contain_only_declared_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "artifact.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("expected.tar.gz", "payload")
                output.writestr("unexpected", "payload")
            with self.assertRaises(download_artifact.ArtifactDownloadError):
                download_artifact._extract(archive, root / "out", "expected.tar.gz")

    def test_short_range_is_retried(self):
        short = mock.MagicMock(status_code=206)
        short.__enter__.return_value = short
        short.iter_content.return_value = [b"short"]
        complete = mock.MagicMock(status_code=206)
        complete.__enter__.return_value = complete
        complete.iter_content.return_value = [b"payload"]
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(download_artifact.requests, "get", side_effect=[short, complete]),
            mock.patch.object(download_artifact._download_range.retry, "wait", return_value=0),
        ):
            destination = Path(temporary) / "range"
            with destination.open("w+b") as output:
                output.truncate(7)
                self.assertEqual(download_artifact._download_range("https://blob", output.fileno(), 0, 6), 7)
            self.assertEqual(destination.read_bytes(), b"payload")


class BuildTests(unittest.TestCase):
    def test_normalizes_restored_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            install = Path(temporary)
            package = install / "openvino_package"
            package.mkdir()
            (package / "setupvars.sh").touch()
            build.normalize_openvino_install(install)
            self.assertTrue((install / "setupvars.sh").is_file())
            self.assertFalse(package.exists())

    def test_release_output_is_selected_unambiguously(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in ("Debug", "Release"):
                directory = root / mode
                directory.mkdir()
                (directory / "tests").write_text(mode, encoding="utf-8")
            selected = build._find_output([root], "tests", "Release")
            self.assertEqual(selected.read_text(encoding="utf-8"), "Release")
            with self.assertRaises(ValueError):
                build._find_output([root], "missing", "Release")


class RuntimeTests(unittest.TestCase):
    def _runtime(self, workspace: Path) -> tests.Runtime:
        setupvars = workspace / "artifact/ov_install/setupvars.sh"
        setupvars.parent.mkdir(parents=True)
        setupvars.touch()
        (workspace / "artifact/module_bin").mkdir()
        return tests.Runtime(workspace, workspace / "results")

    def test_runtime_uses_direct_commands_and_requires_xml(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self._runtime(Path(temporary))
            runtime.results_dir.mkdir()
            with mock.patch.object(tests, "run") as run:
                run.return_value.returncode = 0
                self.assertFalse(runtime.run("unit", ("/test",)))
                (runtime.results_dir / "unit.xml").touch()
                self.assertTrue(runtime.run("unit", ("/test",)))
            self.assertEqual(
                run.call_args.args[0],
                ["/test", f"--gtest_output=xml:{runtime.results_dir / 'unit.xml'}"],
            )

    def test_prepare_sets_runtime_library_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self._runtime(Path(temporary))
            with (
                mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/image"}, clear=True),
                mock.patch.object(tests, "source_setupvars"),
            ):
                runtime.prepare()
                self.assertEqual(
                    os.environ["LD_LIBRARY_PATH"],
                    os.pathsep.join((str(runtime.module_bin), str(runtime.runtime_libs), "/image")),
                )

    def test_registration_replaces_existing_plugin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lib_dir = root / "ov/runtime/lib/intel64"
            lib_dir.mkdir(parents=True)
            (lib_dir / "libopenvino.so").touch()
            (lib_dir / "plugins.xml").write_text(
                '<ie><plugins><plugin name="EXAMPLE" location="old.so"/>'
                '<plugin name="EXAMPLE" location="older.so"/></plugins></ie>',
                encoding="utf-8",
            )
            module = root / "module"
            module.mkdir()
            (module / "libnew.so").touch()
            (module / "plugins.xml").write_text(
                '<ie><plugins><plugin name="EXAMPLE" location="libnew.so"/></plugins></ie>',
                encoding="utf-8",
            )
            tests.register_plugin(root / "ov", module, "plugins.xml")
            entries = tests._plugins(lib_dir / "plugins.xml")
            self.assertEqual(
                [(entry.get("name"), entry.get("location")) for entry in entries],
                [("EXAMPLE", "libnew.so")],
            )
