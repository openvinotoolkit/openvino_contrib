#!/usr/bin/env python3
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request


SCRIPT_DIR = Path(__file__).resolve().parent
PLUGIN_ROOT = SCRIPT_DIR.parent
LLVM_SRC_ROOT = PLUGIN_ROOT / "third_party" / "llvm-project" / "llvm"
VULKAN_HEADERS_ROOT = PLUGIN_ROOT / "third_party" / "Vulkan-Headers"
VULKAN_HEADERS_RELEASE = "v1.3.239"
DEFAULT_DEBIAN_MIRROR = "https://deb.debian.org/debian"
DEFAULT_SYSROOT_PROFILE = "rpi5-bookworm"

SYSROOT_PROFILES: dict[str, dict[str, object]] = {
    "rpi5-bookworm": {
        "description": "Generic Raspberry Pi 5 userspace profile based on Debian Bookworm arm64 packages",
        "suite": "bookworm",
        "arch": "arm64",
        "triple": "aarch64-linux-gnu",
        "cpu": "cortex-a76",
        "gcc_version": "12",
        "packages": [
            "gcc-12-base",
            "libatomic1",
            "libc6",
            "libc6-dev",
            "libgcc-12-dev",
            "libgcc-s1",
            "libstdc++-12-dev",
            "libstdc++6",
            "linux-libc-dev",
            "libvulkan1",
            "zlib1g",
            "zlib1g-dev",
        ],
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hermetic LLVM-based Raspberry Pi Vulkan cross-toolchain bundle for gfx_plugin. "
            "The bundle is created under --output-dir and includes a generated "
            "gfx-rpi-vulkan-aarch64.toolchain.cmake for direct use with cmake."
        )
    )
    parser.add_argument("--output-dir", required=True, help="Directory where the toolchain bundle is generated")
    parser.add_argument(
        "--sysroot-profile",
        default=DEFAULT_SYSROOT_PROFILE,
        choices=sorted(SYSROOT_PROFILES),
        help="Generic target sysroot profile to assemble",
    )
    parser.add_argument(
        "--sysroot-dir",
        help="Use an existing extracted target sysroot instead of downloading packages",
    )
    parser.add_argument(
        "--sysroot-tarball",
        help="Use an existing target sysroot archive instead of downloading packages",
    )
    parser.add_argument("--debian-mirror", default=DEFAULT_DEBIAN_MIRROR, help="Debian mirror used for generic package-based sysroot")
    parser.add_argument("--cmake-generator", default="Ninja", help="CMake generator for building host LLVM")
    parser.add_argument("--llvm-build-type", default="Release", help="Build type for host LLVM tools")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing them")
    return parser.parse_args()


def shlex_quote(value: str) -> str:
    if not value or any(ch in value for ch in " \t\n'\"$`\\"):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    return value


def run(cmd: list[str], *, dry_run: bool, cwd: Path | None = None) -> None:
    pretty = " ".join(shlex_quote(part) for part in cmd)
    prefix = f"[cwd={cwd}] " if cwd else ""
    print(f"{prefix}{pretty}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def require_command(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise RuntimeError(f"Required command was not found in PATH: {name}")
    return resolved


def ensure_dir(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"mkdir -p {path}")
        return
    path.mkdir(parents=True, exist_ok=True)


def normalize_absolute_sysroot_symlinks(sysroot_dir: Path, *, dry_run: bool) -> None:
    rewritten = 0
    for link_path in sysroot_dir.rglob("*"):
        if not link_path.is_symlink():
            continue
        target = os.readlink(link_path)
        if not target.startswith("/"):
            continue

        resolved_target = sysroot_dir / target.lstrip("/")
        if not resolved_target.exists():
            continue

        relative_target = os.path.relpath(resolved_target, start=link_path.parent)
        print(f"rewrite symlink {link_path} -> {relative_target} (was {target})")
        rewritten += 1
        if dry_run:
            continue
        link_path.unlink()
        link_path.symlink_to(relative_target)

    if rewritten:
        print(f"rewrote {rewritten} absolute sysroot symlink(s)")


def write_text(path: Path, content: str, *, executable: bool = False, dry_run: bool) -> None:
    print(f"write {path}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def download_file(url: str, destination: Path, *, dry_run: bool) -> None:
    print(f"download {url} -> {destination}")
    if dry_run:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def extract_archive(archive_path: Path, destination: Path, *, dry_run: bool) -> None:
    print(f"extract {archive_path} -> {destination}")
    if dry_run:
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        archive.extractall(destination)


def find_vulkan_include_root(source_root: Path) -> Path:
    for candidate in source_root.rglob("include/vulkan/vulkan.h"):
        return candidate.parent.parent
    raise RuntimeError(f"Failed to locate include/vulkan/vulkan.h under {source_root}")


def parse_ar_members(path: Path) -> dict[str, bytes]:
    data = path.read_bytes()
    if not data.startswith(b"!<arch>\n"):
        raise RuntimeError(f"{path} is not a valid ar archive")
    offset = 8
    members: dict[str, bytes] = {}
    while offset + 60 <= len(data):
        header = data[offset:offset + 60]
        offset += 60
        name = header[0:16].decode("utf-8").strip()
        if name.endswith("/"):
            name = name[:-1]
        size = int(header[48:58].decode("utf-8").strip())
        payload = data[offset:offset + size]
        offset += size
        if offset % 2 == 1:
            offset += 1
        members[name] = payload
    return members


def extract_deb_data_archive(deb_path: Path, destination: Path, *, dry_run: bool) -> None:
    print(f"extract deb {deb_path} -> {destination}")
    if dry_run:
        return
    members = parse_ar_members(deb_path)
    data_name = next((name for name in members if name.startswith("data.tar")), None)
    if not data_name:
        raise RuntimeError(f"Failed to find data.tar member in {deb_path}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(members[data_name]), mode="r:*") as archive:
        archive.extractall(destination)


def parse_packages_index(text: str) -> dict[str, dict[str, str]]:
    packages: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}
    for raw_line in text.splitlines():
        if not raw_line.strip():
            if "Package" in current:
                packages[current["Package"]] = current
            current = {}
            continue
        if raw_line.startswith(" "):
            continue
        key, value = raw_line.split(":", 1)
        current[key] = value.strip()
    if "Package" in current:
        packages[current["Package"]] = current
    return packages


def load_packages_index(mirror: str, suite: str, arch: str, downloads_dir: Path, *, dry_run: bool) -> dict[str, dict[str, str]]:
    index_dir = downloads_dir / "debian-index"
    index_path = index_dir / f"{suite}-{arch}-Packages.xz"
    ensure_dir(index_dir, dry_run=dry_run)
    index_url = f"{mirror}/dists/{suite}/main/binary-{arch}/Packages.xz"
    download_file(index_url, index_path, dry_run=dry_run)
    if dry_run:
        return {}
    raw = lzma.decompress(index_path.read_bytes()).decode("utf-8", errors="replace")
    return parse_packages_index(raw)


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected}, got {digest}")


def build_generic_sysroot_from_packages(
    output_dir: Path,
    sysroot_dir: Path,
    profile: dict[str, object],
    mirror: str,
    *,
    dry_run: bool,
) -> None:
    downloads_dir = output_dir / "_downloads"
    package_cache_dir = downloads_dir / "debs"
    suite = str(profile["suite"])
    arch = str(profile["arch"])
    packages = list(profile["packages"])

    index = load_packages_index(mirror, suite, arch, downloads_dir, dry_run=dry_run)
    ensure_dir(sysroot_dir, dry_run=dry_run)
    ensure_dir(package_cache_dir, dry_run=dry_run)

    for package_name in packages:
        if dry_run:
            print(f"resolve package {package_name} from {suite}/{arch}")
            continue
        if package_name not in index:
            raise RuntimeError(f"Package {package_name} was not found in Debian index {suite}/{arch}")
        entry = index[package_name]
        relative_filename = entry["Filename"]
        package_url = f"{mirror}/{relative_filename}"
        local_deb = package_cache_dir / Path(relative_filename).name
        download_file(package_url, local_deb, dry_run=False)
        if "SHA256" in entry:
            verify_sha256(local_deb, entry["SHA256"])
        extract_deb_data_archive(local_deb, sysroot_dir, dry_run=False)

    if dry_run:
        return

    # Normalized paths expected by clang/lld wrappers and CMake find logic.
    for path in [
        sysroot_dir / "usr" / "include",
        sysroot_dir / "usr" / "lib",
        sysroot_dir / "usr" / "lib" / str(profile["triple"]),
        sysroot_dir / "lib" / str(profile["triple"]),
        sysroot_dir / "usr" / "lib" / "gcc" / str(profile["triple"]) / str(profile["gcc_version"]),
    ]:
        path.mkdir(parents=True, exist_ok=True)

    # Generic compile-time symlink for libvulkan.so if only SONAME is present.
    vulkan_loader = sysroot_dir / "usr" / "lib" / str(profile["triple"]) / "libvulkan.so"
    vulkan_soname = sysroot_dir / "usr" / "lib" / str(profile["triple"]) / "libvulkan.so.1"
    if not vulkan_loader.exists() and vulkan_soname.exists():
        vulkan_loader.symlink_to("libvulkan.so.1")

    normalize_absolute_sysroot_symlinks(sysroot_dir, dry_run=dry_run)


def copy_user_sysroot(sysroot_source: Path, sysroot_dir: Path, *, dry_run: bool) -> None:
    print(f"copy sysroot {sysroot_source} -> {sysroot_dir}")
    if dry_run:
        return
    if sysroot_dir.exists():
        shutil.rmtree(sysroot_dir)
    shutil.copytree(sysroot_source, sysroot_dir, symlinks=True)
    normalize_absolute_sysroot_symlinks(sysroot_dir, dry_run=dry_run)


def materialize_sysroot(args: argparse.Namespace, output_dir: Path, profile: dict[str, object], *, dry_run: bool) -> None:
    sysroot_dir = output_dir / "sysroot"
    if args.sysroot_dir and args.sysroot_tarball:
        raise RuntimeError("Use only one of --sysroot-dir or --sysroot-tarball")

    if args.sysroot_dir:
        copy_user_sysroot(Path(args.sysroot_dir).resolve(), sysroot_dir, dry_run=dry_run)
        return

    if args.sysroot_tarball:
        archive_path = Path(args.sysroot_tarball).resolve()
        print(f"materialize sysroot from tarball {archive_path}")
        if dry_run:
            return
        if sysroot_dir.exists():
            shutil.rmtree(sysroot_dir)
        sysroot_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(sysroot_dir)
        normalize_absolute_sysroot_symlinks(sysroot_dir, dry_run=dry_run)
        return

    build_generic_sysroot_from_packages(output_dir, sysroot_dir, profile, args.debian_mirror, dry_run=dry_run)


def build_host_llvm_tools(output_dir: Path, generator: str, build_type: str, *, dry_run: bool) -> Path:
    llvm_build_dir = output_dir / "host-llvm-build"
    llvm_install_dir = output_dir / "llvm"
    ensure_dir(llvm_build_dir, dry_run=dry_run)
    configure_cmd = [
        "cmake",
        "-S",
        str(LLVM_SRC_ROOT),
        "-B",
        str(llvm_build_dir),
        "-G",
        generator,
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_INSTALL_PREFIX={llvm_install_dir}",
        "-DLLVM_ENABLE_PROJECTS=clang;lld",
        "-DLLVM_TARGETS_TO_BUILD=AArch64",
        "-DLLVM_BUILD_TOOLS=ON",
        "-DLLVM_ENABLE_RTTI=ON",
        "-DLLVM_INCLUDE_BENCHMARKS=OFF",
        "-DLLVM_INCLUDE_DOCS=OFF",
        "-DLLVM_INCLUDE_EXAMPLES=OFF",
        "-DLLVM_INCLUDE_TESTS=OFF",
        "-DLLVM_INCLUDE_UTILS=ON",
        "-DLLVM_ENABLE_TERMINFO=OFF",
        "-DLLVM_ENABLE_ZLIB=OFF",
        "-DLLVM_ENABLE_ZSTD=OFF",
        "-DLLVM_ENABLE_LIBEDIT=OFF",
        "-DLLVM_ENABLE_LIBXML2=OFF",
        "-DLLVM_BUILD_LLVM_DYLIB=OFF",
        "-DBUILD_SHARED_LIBS=OFF",
    ]
    run(configure_cmd, dry_run=dry_run)
    run(["cmake", "--build", str(llvm_build_dir), "--target", "install"], dry_run=dry_run)
    return llvm_install_dir / "bin"


def install_vulkan_headers(sysroot_dir: Path, *, dry_run: bool) -> None:
    if not VULKAN_HEADERS_ROOT.exists():
        raise RuntimeError(
            "Vendored Vulkan headers were not found. "
            f"Expected {VULKAN_HEADERS_ROOT} prepared from the upstream {VULKAN_HEADERS_RELEASE} release."
        )
    include_root = find_vulkan_include_root(VULKAN_HEADERS_ROOT)
    target_include_dir = sysroot_dir / "usr" / "include"
    print(f"copy Vulkan headers {include_root} -> {target_include_dir}")
    if dry_run:
        return
    target_include_dir.mkdir(parents=True, exist_ok=True)
    for name in ("vulkan", "vk_video"):
        source_dir = include_root / name
        if not source_dir.exists():
            continue
        destination_dir = target_include_dir / name
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        shutil.copytree(source_dir, destination_dir)


def create_python_wrappers(output_dir: Path, profile: dict[str, object], *, dry_run: bool) -> None:
    bin_dir = output_dir / "bin"
    ensure_dir(bin_dir, dry_run=dry_run)
    triple = str(profile["triple"])
    gcc_version = str(profile["gcc_version"])
    cpu = str(profile["cpu"])

    compiler_driver = f"""#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import platform
import subprocess
import sys


MODE = os.environ.get("GFX_TC_COMPILER_MODE", "c")
ROOT = Path(__file__).resolve().parents[1]
LLVM_BIN = ROOT / "llvm" / "bin"
SYSROOT = ROOT / "sysroot"
TRIPLE = "{triple}"
CPU = "{cpu}"
GCC_VER = "{gcc_version}"
EXE_SUFFIX = ".exe" if platform.system() == "Windows" else ""
CLANG = LLVM_BIN / ("clang++" if MODE == "cxx" else "clang")
GCC_LIB = SYSROOT / "usr" / "lib" / "gcc" / TRIPLE / GCC_VER
ALT_GCC_LIB = SYSROOT / "lib" / "gcc" / TRIPLE / GCC_VER
COMMON_ARGS = [
    str(CLANG) + EXE_SUFFIX,
    f"--target={{TRIPLE}}",
    f"--sysroot={{SYSROOT}}",
    f"-mcpu={{CPU}}",
    f"-B{{GCC_LIB}}",
    f"-B{{ALT_GCC_LIB}}",
    f"-isystem{{SYSROOT / 'usr' / 'include' / TRIPLE}}",
]
LINK_ARGS = [
    "-fuse-ld=lld",
    f"-L{{GCC_LIB}}",
    f"-L{{ALT_GCC_LIB}}",
    f"-L{{SYSROOT / 'usr' / 'lib' / TRIPLE}}",
    f"-L{{SYSROOT / 'lib' / TRIPLE}}",
]

if MODE == "cxx":
    cxx_versions = sorted((SYSROOT / "usr" / "include" / "c++").glob("*"))
    if not cxx_versions:
        raise SystemExit("No C++ headers were found in the generated sysroot")
    cxx_ver = cxx_versions[-1].name
    COMMON_ARGS.extend([
        "-stdlib=libstdc++",
        f"-isystem{{SYSROOT / 'usr' / 'include' / 'c++' / cxx_ver}}",
        f"-isystem{{SYSROOT / 'usr' / 'include' / TRIPLE / 'c++' / cxx_ver}}",
    ])

compile_only_flags = {{"-c", "-E", "-S", "-fsyntax-only"}}
user_args = sys.argv[1:]
is_compile_only = any(arg in compile_only_flags for arg in user_args)
args = COMMON_ARGS + ([] if is_compile_only else LINK_ARGS) + user_args
sys.exit(subprocess.call(args))
"""
    pkg_config_driver = f"""#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SYSROOT = ROOT / "sysroot"
pkg_config = shutil.which("pkg-config") or shutil.which("pkgconf")
if not pkg_config:
    raise SystemExit("pkg-config or pkgconf is required on the host")
user_args = sys.argv[1:]
pc_file_args = [
    Path(arg) for arg in user_args
    if not arg.startswith("-") and Path(arg).suffix == ".pc"
]

env = dict(os.environ)
if pc_file_args:
    # Direct validation of a concrete .pc file must not be sysroot-prefixed.
    env.pop("PKG_CONFIG_SYSROOT_DIR", None)
    env.pop("PKG_CONFIG_LIBDIR", None)
else:
    env["PKG_CONFIG_SYSROOT_DIR"] = str(SYSROOT)
    env["PKG_CONFIG_LIBDIR"] = ":".join([
        str(SYSROOT / "usr" / "lib" / "{triple}" / "pkgconfig"),
        str(SYSROOT / "usr" / "lib" / "pkgconfig"),
        str(SYSROOT / "usr" / "share" / "pkgconfig"),
    ])
sys.exit(subprocess.call([pkg_config] + user_args, env=env))
"""
    write_text(bin_dir / "compiler_driver.py", compiler_driver, executable=True, dry_run=dry_run)
    write_text(bin_dir / "pkg_config_driver.py", pkg_config_driver, executable=True, dry_run=dry_run)


def create_launchers(output_dir: Path, *, dry_run: bool) -> None:
    bin_dir = output_dir / "bin"
    unix_specs = {
        "aarch64-linux-gnu-clang": ('export GFX_TC_COMPILER_MODE="c"\nexec python3 "$ROOT/bin/compiler_driver.py" "$@"\n'),
        "aarch64-linux-gnu-clang++": ('export GFX_TC_COMPILER_MODE="cxx"\nexec python3 "$ROOT/bin/compiler_driver.py" "$@"\n'),
        "aarch64-linux-gnu-pkg-config": 'exec python3 "$ROOT/bin/pkg_config_driver.py" "$@"\n',
    }
    windows_specs = {
        "aarch64-linux-gnu-clang.cmd": ('set "GFX_TC_COMPILER_MODE=c"\r\npy -3 "%ROOT%\\bin\\compiler_driver.py" %*\r\n'),
        "aarch64-linux-gnu-clang++.cmd": ('set "GFX_TC_COMPILER_MODE=cxx"\r\npy -3 "%ROOT%\\bin\\compiler_driver.py" %*\r\n'),
        "aarch64-linux-gnu-pkg-config.cmd": 'py -3 "%ROOT%\\bin\\pkg_config_driver.py" %*\r\n',
    }
    for name, body in unix_specs.items():
        content = "#!/usr/bin/env bash\nset -euo pipefail\nROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\"\n" + body
        write_text(bin_dir / name, content, executable=True, dry_run=dry_run)
    for name, body in windows_specs.items():
        content = "@echo off\r\nsetlocal\r\nset \"ROOT=%~dp0..\"\r\n" + body
        write_text(bin_dir / name, content, executable=False, dry_run=dry_run)


def generate_toolchain_file(output_dir: Path, profile: dict[str, object], *, dry_run: bool) -> Path:
    cmake_dir = output_dir / "cmake"
    ensure_dir(cmake_dir, dry_run=dry_run)
    toolchain_path = cmake_dir / "gfx-rpi-vulkan-aarch64.toolchain.cmake"
    triple = str(profile["triple"])
    cpu = str(profile["cpu"])
    content = f"""# Generated by tools/gfx_rpi_vulkan_toolchain_builder.py
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_LIBRARY_ARCHITECTURE {triple})
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

get_filename_component(GFX_RPI_VULKAN_TC_ROOT "${{CMAKE_CURRENT_LIST_DIR}}/.." ABSOLUTE)
set(CMAKE_SYSROOT "${{GFX_RPI_VULKAN_TC_ROOT}}/sysroot" CACHE PATH "GFX RPi Vulkan sysroot" FORCE)

if(WIN32)
    set(_gfx_tc_cmd_ext ".cmd")
    set(_gfx_tc_host_exe ".exe")
else()
    set(_gfx_tc_cmd_ext "")
    set(_gfx_tc_host_exe "")
endif()

set(CMAKE_C_COMPILER "${{GFX_RPI_VULKAN_TC_ROOT}}/bin/aarch64-linux-gnu-clang${{_gfx_tc_cmd_ext}}")
set(CMAKE_CXX_COMPILER "${{GFX_RPI_VULKAN_TC_ROOT}}/bin/aarch64-linux-gnu-clang++${{_gfx_tc_cmd_ext}}")
set(CMAKE_AR "${{GFX_RPI_VULKAN_TC_ROOT}}/llvm/bin/llvm-ar${{_gfx_tc_host_exe}}")
set(CMAKE_RANLIB "${{GFX_RPI_VULKAN_TC_ROOT}}/llvm/bin/llvm-ranlib${{_gfx_tc_host_exe}}")
set(CMAKE_STRIP "${{GFX_RPI_VULKAN_TC_ROOT}}/llvm/bin/llvm-strip${{_gfx_tc_host_exe}}")
set(CMAKE_LINKER "${{GFX_RPI_VULKAN_TC_ROOT}}/llvm/bin/ld.lld${{_gfx_tc_host_exe}}")
set(PKG_CONFIG_EXECUTABLE "${{GFX_RPI_VULKAN_TC_ROOT}}/bin/aarch64-linux-gnu-pkg-config${{_gfx_tc_cmd_ext}}")

set(CMAKE_C_COMPILER_TARGET {triple})
set(CMAKE_CXX_COMPILER_TARGET {triple})

set(CMAKE_C_FLAGS_INIT "-mcpu={cpu}")
set(CMAKE_CXX_FLAGS_INIT "-mcpu={cpu}")

set(CMAKE_FIND_ROOT_PATH "${{CMAKE_SYSROOT}}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(ENV{{PKG_CONFIG_SYSROOT_DIR}} "${{CMAKE_SYSROOT}}")
set(ENV{{PKG_CONFIG_LIBDIR}}
    "${{CMAKE_SYSROOT}}/usr/lib/{triple}/pkgconfig:${{CMAKE_SYSROOT}}/usr/lib/pkgconfig:${{CMAKE_SYSROOT}}/usr/share/pkgconfig")
"""
    write_text(toolchain_path, content, dry_run=dry_run)
    return toolchain_path


def write_manifest(output_dir: Path, profile_name: str, profile: dict[str, object], toolchain_file: Path, *, dry_run: bool) -> None:
    manifest = {
        "bundle_root": str(output_dir),
        "generated_toolchain_file": str(toolchain_file),
        "plugin_root": str(PLUGIN_ROOT),
        "llvm_source_root": str(LLVM_SRC_ROOT),
        "vulkan_headers_root": str(VULKAN_HEADERS_ROOT),
        "vulkan_headers_release": VULKAN_HEADERS_RELEASE,
        "sysroot_profile": profile_name,
        "target": {
            "triple": profile["triple"],
            "cpu": profile["cpu"],
            "suite": profile["suite"],
            "arch": profile["arch"],
            "gcc_version": profile["gcc_version"],
        },
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
    }
    write_text(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n", dry_run=dry_run)


def validate_paths(output_dir: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    host_exe = ".exe" if platform.system() == "Windows" else ""
    required = [
        output_dir / "llvm" / "bin" / f"clang{host_exe}",
        output_dir / "llvm" / "bin" / f"clang++{host_exe}",
        output_dir / "llvm" / "bin" / f"ld.lld{host_exe}",
        output_dir / "sysroot" / "usr" / "include" / "vulkan" / "vulkan.h",
        output_dir / "cmake" / "gfx-rpi-vulkan-aarch64.toolchain.cmake",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Toolchain bundle is incomplete. Missing paths:\n" + "\n".join(missing))


def main() -> int:
    args = parse_args()
    require_command("cmake")
    output_dir = Path(args.output_dir).resolve()
    profile = SYSROOT_PROFILES[args.sysroot_profile]
    if not LLVM_SRC_ROOT.exists():
        raise RuntimeError(f"Vendored LLVM source tree was not found: {LLVM_SRC_ROOT}")
    if not VULKAN_HEADERS_ROOT.exists():
        raise RuntimeError(
            "Vendored Vulkan headers were not found. "
            f"Expected {VULKAN_HEADERS_ROOT} from the upstream {VULKAN_HEADERS_RELEASE} release."
        )

    print(f"toolchain bundle root: {output_dir}")
    print(f"sysroot profile: {args.sysroot_profile} - {profile['description']}")
    print(f"Vulkan-Headers submodule: {VULKAN_HEADERS_ROOT} ({VULKAN_HEADERS_RELEASE})")
    ensure_dir(output_dir, dry_run=args.dry_run)

    llvm_bin_dir = build_host_llvm_tools(output_dir, args.cmake_generator, args.llvm_build_type, dry_run=args.dry_run)
    print(f"host LLVM tools: {llvm_bin_dir}")
    materialize_sysroot(args, output_dir, profile, dry_run=args.dry_run)
    install_vulkan_headers(output_dir / "sysroot", dry_run=args.dry_run)
    create_python_wrappers(output_dir, profile, dry_run=args.dry_run)
    create_launchers(output_dir, dry_run=args.dry_run)
    toolchain_file = generate_toolchain_file(output_dir, profile, dry_run=args.dry_run)
    write_manifest(output_dir, args.sysroot_profile, profile, toolchain_file, dry_run=args.dry_run)
    validate_paths(output_dir, dry_run=args.dry_run)
    print(f"generated toolchain file: {toolchain_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
