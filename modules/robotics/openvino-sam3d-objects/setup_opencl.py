#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
SAM 3D Objects — Build OpenVINO Custom Extensions

Builds the C++/OpenCL custom extensions for sparse convolution and vox2seq
using CMake. Inspired by the BEVFusion setup_opencl.py pattern.

Usage:
    python setup_opencl.py              # Build all extensions
    python setup_opencl.py --clean      # Clean build directories first
    python setup_opencl.py --verify     # Build and verify loading
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ── OpenVINO extension definitions ────────────────────────────────────────────
OV_EXTENSIONS = [
    {
        "name": "SparseConv3d",
        "dir": str(SCRIPT_DIR / "openvino_extensions" / "sparse_conv_3d"),
        "output": "libopenvino_sparse_conv_3d_extension.so",
    },
    {
        "name": "Vox2Seq",
        "dir": str(SCRIPT_DIR / "openvino_extensions" / "vox2seq"),
        "output": "libov_vox2seq.so",
    },
]


def build_extension(ext_info: dict, clean: bool = False) -> bool:
    """Build a single OpenVINO custom extension via CMake."""
    ext_dir = ext_info["dir"]
    ext_name = ext_info["name"]
    build_dir = os.path.join(ext_dir, "build")
    output_path = os.path.join(build_dir, ext_info["output"])

    if not os.path.isdir(ext_dir):
        print(f"[build] Source directory not found: {ext_dir}, skipping {ext_name}")
        return False

    print(f"\n{'=' * 60}")
    print(f"Building OpenVINO {ext_name} extension")
    print(f"{'=' * 60}")

    if clean and os.path.isdir(build_dir):
        import shutil
        shutil.rmtree(build_dir)
        print(f"[build] Cleaned {build_dir}")

    os.makedirs(build_dir, exist_ok=True)

    # CMake configure
    cmake_cmd = ["cmake", ".."]
    print(f"[build] Running: {' '.join(cmake_cmd)}  (in {build_dir})")
    result = subprocess.run(
        cmake_cmd, cwd=build_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[build] cmake configure failed:\n{result.stdout}\n{result.stderr}")
        return False
    print("[build] CMake configure OK")

    # CMake build
    nproc = os.cpu_count() or 1
    make_cmd = ["cmake", "--build", ".", "-j", str(nproc)]
    print(f"[build] Running: {' '.join(make_cmd)}  (in {build_dir})")
    result = subprocess.run(
        make_cmd, cwd=build_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[build] Build failed:\n{result.stdout}\n{result.stderr}")
        return False

    if os.path.isfile(output_path):
        print(f"[build] ✓ Built: {output_path}")
        return True
    else:
        # Check if it was built with a different naming convention
        for f in Path(build_dir).glob("*.so"):
            print(f"[build] Found .so: {f}")
        print(f"[build] ✗ Expected output not found: {output_path}")
        return False


def verify_extension(ext_info: dict) -> bool:
    """Verify that an extension can be loaded by OpenVINO."""
    build_dir = os.path.join(ext_info["dir"], "build")
    lib_path = os.path.join(build_dir, ext_info["output"])

    if not os.path.isfile(lib_path):
        print(f"[verify] {ext_info['name']}: .so not found, skipping")
        return False

    try:
        import openvino as ov
        core = ov.Core()
        core.add_extension(lib_path)
        print(f"[verify] ✓ {ext_info['name']}: loaded successfully")
        return True
    except Exception as e:
        print(f"[verify] ✗ {ext_info['name']}: load failed — {e}")
        return False


def build_all(clean: bool = False, verify: bool = False):
    """Build all OpenVINO custom extensions."""
    # Check that OpenVINO is available
    try:
        import openvino
        print(f"OpenVINO version: {openvino.__version__}")
    except ImportError:
        print("ERROR: OpenVINO Python package not found.")
        print("Install with: pip install openvino>=2025.0")
        sys.exit(1)

    # Check cmake is available
    try:
        result = subprocess.run(
            ["cmake", "--version"], capture_output=True, text=True
        )
        cmake_version = result.stdout.split("\n")[0] if result.returncode == 0 else "unknown"
        print(f"CMake: {cmake_version}")
    except FileNotFoundError:
        print("ERROR: cmake not found. Install cmake first.")
        sys.exit(1)

    results = {}
    for ext_info in OV_EXTENSIONS:
        success = build_extension(ext_info, clean=clean)
        results[ext_info["name"]] = success

    if verify:
        print(f"\n{'=' * 60}")
        print("Verifying extensions...")
        print(f"{'=' * 60}")
        for ext_info in OV_EXTENSIONS:
            if results.get(ext_info["name"]):
                verify_extension(ext_info)

    # Summary
    print(f"\n{'=' * 60}")
    print("Build Summary")
    print(f"{'=' * 60}")
    all_ok = True
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
        if not success:
            all_ok = False

    if all_ok:
        print("\nAll extensions built successfully!")
    else:
        print("\nSome extensions failed to build. Check the output above.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Build SAM3D OpenVINO custom extensions"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Clean build directories before building"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify extensions can be loaded after building"
    )
    args = parser.parse_args()
    build_all(clean=args.clean, verify=args.verify)


if __name__ == "__main__":
    main()
