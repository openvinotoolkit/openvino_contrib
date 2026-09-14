#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Collect the minimal runtime needed by the NVIDIA GPU test job."""

import argparse
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def plugin_container(tree: ET.ElementTree) -> ET.Element:
    root = tree.getroot()
    container = root if root.tag == "plugins" else root.find("plugins")
    if container is None:
        raise ValueError("plugin registry has no <plugins> element")
    return container


def copy_file(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    if not target.exists():
        shutil.copy2(source.resolve(), target)
    return target


def linked_libraries(binary: Path) -> list[tuple[str, Path]]:
    output = subprocess.run(
        ["ldd", str(binary)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    result = []
    for line in output.splitlines():
        name, separator, location = line.partition("=>")
        if not separator:
            continue
        location = location.strip()
        if location.startswith("not found"):
            raise RuntimeError(f"{name.strip()} required by {binary} was not found")
        path = Path(location.split(maxsplit=1)[0])
        if path.is_absolute():
            result.append((name.strip(), path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvino-package", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runtime-plugins", nargs="+", required=True)
    parser.add_argument("--frontends", nargs="+", required=True)
    parser.add_argument("--executables", nargs="+", required=True)
    parser.add_argument("--extra-library", type=Path, required=True)
    args = parser.parse_args()

    runtime = args.openvino_package.resolve() / "runtime"
    runtime_lib = runtime / "lib/intel64"
    artifact_bin = args.artifact.resolve() / "bin"
    artifact_lib = args.artifact.resolve() / "lib"

    target_plugins = plugin_container(ET.parse(artifact_lib / "plugins.xml"))

    objects = [artifact_bin / name for name in args.executables]
    for executable in objects:
        if not executable.is_file():
            raise FileNotFoundError(executable)
    for plugin in target_plugins:
        location = plugin.get("location", "")
        if Path(location).name != location or not (artifact_lib / location).is_file():
            raise ValueError(f"invalid NVIDIA plugin location: {location!r}")
        objects.append(artifact_lib / location)
    for location in args.runtime_plugins:
        if Path(location).name != location:
            raise ValueError(f"invalid OpenVINO plugin location: {location!r}")
        objects.append(copy_file(runtime_lib / location, artifact_lib))
    for name in args.frontends:
        if not name.isidentifier():
            raise ValueError(f"invalid OpenVINO frontend name: {name!r}")
        candidates = sorted(path for path in runtime_lib.glob(f"libopenvino_{name}_frontend.so*") if path.is_file())
        if not candidates:
            raise FileNotFoundError(f"OpenVINO {name} frontend")
        objects.extend(copy_file(path, artifact_lib) for path in candidates)

    objects.append(copy_file(args.extra_library.resolve(), artifact_bin))

    package_root = args.openvino_package.resolve()
    for binary in objects:
        for name, dependency in linked_libraries(binary):
            from_package = dependency.is_relative_to(package_root)
            if from_package or name.startswith("libcutensor.so"):
                copy_file(dependency, artifact_lib)


if __name__ == "__main__":
    main()
