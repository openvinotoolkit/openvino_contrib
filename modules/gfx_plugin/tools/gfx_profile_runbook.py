#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULTS = {
    "macos_build": "/Users/anesterov/repos/openvino_contrib/build-gfx-plugin-macos",
    "android_build": "/Users/anesterov/repos/openvino_contrib/build-gfx-plugin-android",
    "rpi_build": "/Users/anesterov/repos/openvino_contrib/build-gfx-plugin-rpi",
    "model": "/Users/anesterov/repos/openvino_contrib/yolo12n_ir/yolov12n.xml",
    "android_dir": "/data/local/tmp/openvino_gfx_android",
    "rpi_dir": "/home/anesterov/gfx_eval",
}


def run(argv: List[str]) -> str:
    return subprocess.check_output(argv, text=True)


def detect_android_package() -> str | None:
    try:
        proc = subprocess.run(["adb", "shell", "pm", "list", "packages"],
                              text=True,
                              capture_output=True,
                              check=False)
    except Exception:
        return None
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if "SecurityException" in output:
        return None
    if proc.returncode != 0:
        return None
    packages = [line.split(":", 1)[1].strip() for line in output.splitlines() if line.startswith("package:")]
    preferred = [
        "com.intel.openvino.benchmark_app",
        "org.openvino.benchmark_app",
    ]
    for candidate in preferred:
        if candidate in packages:
            return candidate
    for package in packages:
        if "benchmark" in package and "openvino" in package:
            return package
    for package in packages:
        if "benchmark" in package:
            return package
    return None


def shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def macos_binary(name: str, build_dir: str) -> str:
    return f"{build_dir}/output/bin/arm64/RelWithDebInfo/{name}"


def android_binary(name: str, build_dir: str) -> str:
    return f"{build_dir}/output/bin/aarch64/Release/{name}"


def rpi_binary(name: str, build_dir: str) -> str:
    return f"{build_dir}/output/bin/aarch64/Release/{name}"


def build_plan(platform: str, args: argparse.Namespace) -> Dict[str, object]:
    if platform == "macos":
        build_dir = args.build_dir or DEFAULTS["macos_build"]
        microbench = macos_binary("ov_gfx_microbench", build_dir)
        benchmark = macos_binary("benchmark_app", build_dir)
        return {
            "platform": platform,
            "build_dir": build_dir,
            "commands": {
                "microbench": shell_join([
                    microbench,
                    "--backend", "metal",
                    "--warmup", str(args.warmup),
                    "--iterations", str(args.iterations),
                    "--output", args.microbench_output or "/tmp/gfx-microbench-macos.json",
                    "--calibration-output", args.calibration_output or "/tmp/gfx-calibration-macos.json",
                ]),
                "smoke_roundtrip": shell_join([
                    sys.executable,
                    str(Path(__file__).with_name("gfx_microbench_smoke.py")),
                    "--platform", "host",
                    "--binary", microbench,
                    "--backend", "metal",
                ]),
                "benchmark": (
                    "OV_GFX_PROFILE_TRACE=signpost "
                    "OV_GFX_PROFILE_TRACE_FILE=/tmp/gfx-trace-macos.json "
                    + shell_join([benchmark, "-m", args.model or DEFAULTS["model"], "-d", "GFX", "-pc", "-niter", "10"])
                ),
                "xctrace": (
                    "xcrun xctrace record --template 'Time Profiler' --time-limit 10s "
                    "--output /tmp/gfx-profile.trace --launch -- "
                    + shell_join([benchmark, "-m", args.model or DEFAULTS["model"], "-d", "GFX", "-pc", "-niter", "10"])
                ),
            },
        }

    if platform == "android":
        build_dir = args.build_dir or DEFAULTS["android_build"]
        remote_dir = args.android_dir or DEFAULTS["android_dir"]
        microbench = android_binary("ov_gfx_microbench", build_dir)
        benchmark = android_binary("benchmark_app", build_dir)
        plugin_path = args.plugin_path or f"{remote_dir}/libopenvino_gfx_plugin.so"
        package = args.android_package or detect_android_package()
        return {
            "platform": platform,
            "build_dir": build_dir,
            "android_package": package,
            "commands": {
                "push_microbench": shell_join(["adb", "push", microbench, f"{remote_dir}/ov_gfx_microbench"]),
                "microbench": (
                    "adb shell "
                    + shlex.quote(
                        f"cd {remote_dir} && "
                        f"GFX_PLUGIN_PATH={plugin_path} LD_LIBRARY_PATH={remote_dir} "
                        f"./ov_gfx_microbench --backend vulkan --warmup {args.warmup} --iterations {args.iterations} "
                        f"--output {remote_dir}/gfx-microbench-android.json "
                        f"--calibration-output {remote_dir}/gfx-calibration-android.json"
                    )
                ),
                "smoke_roundtrip": shell_join([
                    sys.executable,
                    str(Path(__file__).with_name("gfx_microbench_smoke.py")),
                    "--platform", "android",
                    "--binary", microbench,
                    "--backend", "vulkan",
                    "--plugin-path", plugin_path,
                    "--android-dir", remote_dir,
                ]),
                "benchmark": (
                    "adb shell "
                    + shlex.quote(
                        f"cd {remote_dir} && "
                        f"OV_GFX_PROFILE_TRACE=perfetto OV_GFX_PROFILE_TRACE_FILE={remote_dir}/gfx-trace-android.json "
                        f"GFX_PLUGIN_PATH={plugin_path} LD_LIBRARY_PATH={remote_dir} "
                        f"./benchmark_app -m {args.remote_model or f'{remote_dir}/yolov12n.xml'} -d GFX -pc -niter 10"
                    )
                ),
                "validation_enable": None if not package else "\n".join([
                    "adb shell settings put global enable_gpu_debug_layers 1",
                    f"adb shell settings put global gpu_debug_app {shlex.quote(package)}",
                    "adb shell settings put global gpu_debug_layers VK_LAYER_KHRONOS_validation",
                ]),
                "validation_disable": "\n".join([
                    "adb shell settings delete global enable_gpu_debug_layers",
                    "adb shell settings delete global gpu_debug_app",
                    "adb shell settings delete global gpu_debug_layers",
                ]),
            },
        }

    build_dir = args.build_dir or DEFAULTS["rpi_build"]
    remote_dir = args.rpi_dir or DEFAULTS["rpi_dir"]
    benchmark = f"{remote_dir}/benchmark_app"
    microbench = f"{remote_dir}/ov_gfx_microbench"
    return {
        "platform": platform,
        "build_dir": build_dir,
        "commands": {
            "microbench": "\n".join([
                f"cd {shlex.quote(remote_dir)}",
                f"LD_LIBRARY_PATH={shlex.quote(remote_dir + '/libs/Release')}:{shlex.quote(remote_dir)} "
                + shell_join([
                    microbench,
                    "--backend", "vulkan",
                    "--warmup", str(args.warmup),
                    "--iterations", str(args.iterations),
                    "--output", f"{remote_dir}/gfx-microbench-rpi.json",
                    "--calibration-output", f"{remote_dir}/gfx-calibration-rpi.json",
                ]),
            ]),
            "benchmark": "\n".join([
                f"cd {shlex.quote(remote_dir)}",
                f"LD_LIBRARY_PATH={shlex.quote(remote_dir + '/libs/Release')}:{shlex.quote(remote_dir)} "
                + shell_join([benchmark, "-m", args.remote_model or f"{remote_dir}/yolov12n.xml", "-d", "GFX", "-pc", "-niter", "10"]),
            ]),
            "perf_stat": shell_join([
                "perf", "stat", "-e", "cycles,instructions,cache-misses,branch-misses", "--",
                benchmark, "-m", args.remote_model or f"{remote_dir}/yolov12n.xml", "-d", "GFX", "-pc", "-niter", "10"
            ]),
            "perf_record": "\n".join([
                shell_join([
                    "perf", "record", "-g", "--",
                    benchmark, "-m", args.remote_model or f"{remote_dir}/yolov12n.xml", "-d", "GFX", "-pc", "-niter", "10"
                ]),
                "perf report",
            ]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ready-to-run GFX profiling commands for macOS, Android, and Raspberry Pi.")
    parser.add_argument("--platform", choices=["macos", "android", "rpi"], required=True)
    parser.add_argument("--build-dir", help="Override build directory for the selected platform.")
    parser.add_argument("--model", help="Host-side model path for macOS.")
    parser.add_argument("--remote-model", help="Remote model path for Android or Raspberry Pi.")
    parser.add_argument("--plugin-path", help="Override GFX plugin path for Android.")
    parser.add_argument("--android-dir", help="Override Android working directory.")
    parser.add_argument("--rpi-dir", help="Override Raspberry Pi working directory.")
    parser.add_argument("--android-package", help="Override Android package name used for validation layers.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--microbench-output")
    parser.add_argument("--calibration-output")
    parser.add_argument("--format", choices=["json", "shell"], default="json")
    args = parser.parse_args()

    plan = build_plan(args.platform, args)
    if args.format == "json":
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(f"# platform: {plan['platform']}")
        for name, command in plan["commands"].items():
            if not command:
                continue
            print(f"\n# {name}\n{command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
