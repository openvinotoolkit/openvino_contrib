# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build and package an OpenVINO C++ plugin module."""

import os
import re
import shutil
import tarfile
from pathlib import Path

from common.core import env_lines, relative_path, required_env, run, safe_name

from openvino_plugin import source_setupvars


def _module_path() -> Path:
    return Path("openvino_contrib") / relative_path(required_env("CI_MODULE_PATH"), "modules")


def normalize_openvino_install(install: Path) -> None:
    if (install / "setupvars.sh").is_file():
        return
    packages = [path for path in install.glob("openvino_package*") if path.is_dir()]
    if len(packages) != 1:
        raise ValueError(f"expected one unpacked OpenVINO package under {install}")
    package = packages[0]
    contents = list(package.iterdir())
    collisions = [path.name for path in contents if (install / path.name).exists()]
    if collisions:
        raise ValueError(f"OpenVINO artifact path collision: {', '.join(sorted(collisions))}")
    for path in contents:
        path.rename(install / path.name)
    package.rmdir()


def build(
    setupvars: Path,
    module_path: Path,
    build_dir: Path,
    developer_package_dir: Path,
    build_type: str,
    generator: str,
    targets: list[str],
    extra_cmake: list[str],
) -> None:
    source_setupvars(setupvars)
    module = module_path.resolve()
    build_dir = build_dir.resolve()
    dev_pkg = developer_package_dir.resolve()
    if not (module / "CMakeLists.txt").is_file():
        raise ValueError(f"No CMakeLists.txt under module path {module}")
    if not dev_pkg.is_dir():
        raise ValueError(f"Developer package cmake dir not found: {dev_pkg}")
    if not targets:
        raise ValueError("No build targets provided")

    run(
        [
            "cmake",
            "-G",
            generator,
            f"-DOpenVINODeveloperPackage_DIR={dev_pkg}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            *extra_cmake,
            "-S",
            str(module),
            "-B",
            str(build_dir),
        ]
    )
    run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--parallel",
            str(os.cpu_count() or 1),
            "--config",
            build_type,
            "--verbose",
            "--target",
            *targets,
        ]
    )


def _copy(source: Path, destination: Path, name: str | None = None) -> None:
    if not source.is_file():
        raise ValueError(f"required artifact file not found: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (name or source.name)
    if target.exists():
        raise ValueError(f"artifact basename collision: {target.name}")
    shutil.copy2(source, target, follow_symlinks=True)


def _find_output(roots: list[Path], name: str, build_type: str) -> Path:
    if Path(name).name != name:
        raise ValueError(f"build output must be a filename: {name}")
    matches = {
        path.resolve(): path for root in roots if root.is_dir() for path in root.rglob(name) if path.is_file()
    }
    configured = [
        path
        for path in matches.values()
        if any(part.casefold() == build_type.casefold() for part in path.parts)
    ]
    candidates = sorted(configured or matches.values())
    if len(candidates) != 1:
        found = "none" if not candidates else ", ".join(map(str, candidates))
        raise ValueError(f"expected one build output '{name}', found: {found}")
    return candidates[0]


def _copy_relative(root: Path, paths: list[str], destination: Path) -> None:
    root = root.resolve()
    for value in paths:
        source = (root / value).resolve()
        if Path(value).is_absolute() or (source != root and root not in source.parents):
            raise ValueError(f"artifact path must be relative to {root}: {value}")
        _copy(source, destination)


def _copy_runtime_libraries(entries: list[str], destination: Path) -> None:
    sonames: dict[str, Path] = {}
    if any("/" not in entry for entry in entries):
        for line in run(["ldconfig", "-p"], capture=True).stdout.splitlines():
            if match := re.match(r"^\s*(\S+)\s+.*=>\s+(\S+)$", line):
                sonames.setdefault(match.group(1), Path(match.group(2)))
    for entry in entries:
        source = Path(entry) if "/" in entry else sonames.get(entry)
        if source is None:
            raise ValueError(f"runtime library not found: {entry}")
        _copy(source.resolve(), destination, Path(entry).name)


def pack(
    staging: Path,
    ov_install: Path,
    build_dir: Path,
    build_type: str,
    module_path: Path,
    outputs: list[str],
    utils: list[str],
    test_support: list[str],
    runtime_libraries: list[str],
    archive: Path,
) -> None:
    staging = staging.resolve()
    staging.mkdir(parents=True, exist_ok=True)
    if any(staging.iterdir()):
        raise ValueError(f"staging directory must be empty: {staging}")

    ov_install = ov_install.resolve()
    runtime_out = staging / "ov_install"
    for name in ("runtime", "setupvars.sh"):
        source = ov_install / name
        if source.is_dir():
            shutil.copytree(source, runtime_out / name, symlinks=True)
        elif source.is_file() or source.is_symlink():
            runtime_out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, runtime_out / name, follow_symlinks=False)
        else:
            raise ValueError(f"OpenVINO runtime path not found: {source}")

    roots = [build_dir.resolve(), module_path.resolve() / "bin"]
    if not outputs:
        raise ValueError("no module outputs were declared")
    for name in outputs:
        _copy(_find_output(roots, name, build_type), staging / "module_bin")
    _copy_relative(module_path / "utils", utils, staging / "module_utils")
    _copy_relative(ov_install, test_support, staging / "module_bin")
    _copy_runtime_libraries(runtime_libraries, staging / "runtime_libs")

    with tarfile.open(archive.resolve(), "w:gz", compresslevel=1) as output:
        for item in sorted(staging.iterdir()):
            output.add(item, arcname=item.name)


def main() -> None:
    install = Path(required_env("OV_INSTALL_DIR"))
    normalize_openvino_install(install)
    safe_name(required_env("CI_MODULE_NAME"), "module name")
    build(
        install / "setupvars.sh",
        _module_path(),
        Path(required_env("MODULE_BUILD_DIR")),
        install / "developer_package/cmake",
        required_env("CMAKE_BUILD_TYPE"),
        required_env("CMAKE_GENERATOR"),
        env_lines("CI_CMAKE_TARGETS", required=True),
        env_lines("CI_EXTRA_CMAKE"),
    )
    pack(
        Path(required_env("STAGING_DIR")),
        Path(required_env("OV_INSTALL_DIR")),
        Path(required_env("MODULE_BUILD_DIR")),
        required_env("CMAKE_BUILD_TYPE"),
        _module_path(),
        env_lines("CI_ARTIFACT_FILES", required=True),
        env_lines("CI_UTILS"),
        env_lines("CI_TEST_SUPPORT_LIBRARIES"),
        env_lines("CI_RUNTIME_LIBRARIES"),
        Path(required_env("ARCHIVE_PATH")),
    )
