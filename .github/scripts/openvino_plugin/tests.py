# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test runtime and plugin registration for OpenVINO plugin modules."""

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

from common.core import required_env, run

from openvino_plugin import source_setupvars

MAX_PLUGIN_CONFIG_BYTES = 1024 * 1024


def _plugins(path: Path, required: bool = False) -> list[ET.Element]:
    if not path.is_file():
        if required:
            raise ValueError(f"plugin registry not found: {path}")
        return []
    if path.stat().st_size > MAX_PLUGIN_CONFIG_BYTES:
        raise ValueError(f"plugin registry is too large: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"invalid plugin registry '{path}': {error}") from None
    container = root if root.tag == "plugins" else root.find("plugins")
    result = [root] if root.tag == "plugin" else list(container) if container is not None else []
    if required and not result:
        raise ValueError(f"no plugin entries in {path}")
    return result


def register_plugin(ov_install: Path, module_bin: Path, config: str) -> None:
    if Path(config).name != config:
        raise ValueError(f"plugin config must be a filename: {config}")
    source_path = module_bin / config
    source = _plugins(source_path, required=True)
    names, locations = set(), []
    for plugin in source:
        name, location = plugin.get("name"), plugin.get("location")
        if (
            plugin.tag != "plugin"
            or not name
            or not location
            or Path(location).name != location
            or name in names
            or not (module_bin / location).is_file()
        ):
            raise ValueError(f"invalid plugin entry in {source_path}")
        names.add(name)
        locations.append(location)

    lib_dir = ov_install / "runtime/lib/intel64"
    if not any(lib_dir.glob("libopenvino.so*")):
        raise ValueError(f"OpenVINO runtime library directory is invalid: {lib_dir}")
    target = lib_dir / "plugins.xml"
    if target.is_file():
        _plugins(target)
        tree = ET.parse(target)
        root = tree.getroot()
        container = root if root.tag == "plugins" else root.find("plugins")
        if container is None:
            container = ET.SubElement(root, "plugins")
    else:
        root = ET.Element("ie")
        container = ET.SubElement(root, "plugins")
        tree = ET.ElementTree(root)
    for existing in list(container):
        if existing.tag == "plugin" and existing.get("name") in names:
            container.remove(existing)
    container.extend(source)
    tree.write(target, encoding="utf-8", xml_declaration=True)
    for location in locations:
        shutil.copy2(module_bin / location, lib_dir / location, follow_symlinks=True)


class Runtime:
    def __init__(self, workspace: Path, results_dir: Path) -> None:
        self.workspace = workspace.resolve()
        self.results_dir = results_dir.resolve()
        if self.workspace != self.results_dir and self.workspace not in self.results_dir.parents:
            raise ValueError(f"results directory is outside the workspace: {self.results_dir}")
        self.artifact = self.workspace / "artifact"
        self.ov_install = self.artifact / "ov_install"
        self.module_bin = self.artifact / "module_bin"
        self.runtime_libs = self.artifact / "runtime_libs"
        self.module_utils = self.artifact / "module_utils"
        self.setupvars = self.ov_install / "setupvars.sh"
        if not self.setupvars.is_file() or not self.module_bin.is_dir():
            raise ValueError(f"invalid plugin artifact: {self.artifact}")

    def prepare(self, plugin_config: str | None = None) -> None:
        if plugin_config:
            register_plugin(self.ov_install, self.module_bin, plugin_config)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        source_setupvars(self.setupvars)
        paths = [str(self.module_bin), str(self.runtime_libs), os.environ.get("LD_LIBRARY_PATH", "")]
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, paths))

    def executable(self, name: str) -> Path | None:
        path = self.module_bin / name
        if Path(name).name != name:
            raise ValueError(f"test executable must be a filename: {name}")
        if not path.is_file():
            print(f"::error::{name} not found in {self.module_bin}", file=sys.stderr)
            return None
        return path

    def run(self, name: str, command: tuple[str, ...]) -> bool:
        if Path(name).name != name or not name:
            raise ValueError(f"invalid test result name: {name}")
        result = self.results_dir / f"{name}.xml"
        print(f"::group::{name}")
        try:
            return_code = run([*command, f"--gtest_output=xml:{result}"], check=False).returncode
        finally:
            print("::endgroup::")
        if return_code == 0 and not result.is_file():
            print(f"::error::{name} did not produce {result}", file=sys.stderr)
            return False
        return return_code == 0


TestPolicy = Callable[[Runtime, str], list[str]]


def execute(
    policy: TestPolicy,
    *,
    before_prepare: Callable[[], None] | None = None,
    plugin_config: str | None = None,
) -> None:
    try:
        preset = required_env("CI_PRESET")
        runtime = Runtime(Path.cwd(), Path.cwd() / "test-results")
        if before_prepare:
            before_prepare()
        runtime.prepare(plugin_config)
        failures = policy(runtime, preset)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(f"::error::plugin tests failed: {error}") from None
    if failures:
        raise SystemExit(f"::error::failed test kinds: {', '.join(failures)}")
