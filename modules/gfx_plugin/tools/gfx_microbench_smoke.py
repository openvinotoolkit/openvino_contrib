#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List


def run(argv: List[str], *, env: Dict[str, str] | None = None) -> str:
    proc = subprocess.run(argv, check=True, text=True, capture_output=True, env=env)
    return proc.stdout


def adb_shell(command: str) -> str:
    return run(["adb", "shell", command])


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test ov_gfx_microbench and calibration roundtrip.")
    parser.add_argument("--binary", required=True, help="Path to ov_gfx_microbench binary.")
    parser.add_argument("--backend", required=True, help="Backend name passed to the tool.")
    parser.add_argument("--platform", choices=["host", "android"], default="host")
    parser.add_argument("--plugin-path", help="Required on Android or when the plugin path must be overridden.")
    parser.add_argument("--android-dir", default="/data/local/tmp/openvino_gfx_android",
                        help="Remote working directory for Android runs.")
    args = parser.parse_args()

    binary = Path(args.binary)
    if args.platform == "host":
        with tempfile.TemporaryDirectory(prefix="gfx_microbench_smoke_") as tmp_dir:
            report = Path(tmp_dir) / "report.json"
            calibration = Path(tmp_dir) / "calibration.json"
            cmd = [
                str(binary),
                "--backend", args.backend,
                "--warmup", "0",
                "--iterations", "1",
                "--output", str(report),
                "--calibration-output", str(calibration),
            ]
            env = os.environ.copy()
            if args.plugin_path:
                env["GFX_PLUGIN_PATH"] = args.plugin_path
            run(cmd, env=env)
            stdout = run([
                str(binary),
                "--backend", args.backend,
                "--warmup", "0",
                "--iterations", "1",
                "--calibration-input", str(calibration),
            ], env=env)
            payload = json.loads(stdout)
            loaded = payload.get("loaded_calibration", {})
            if not loaded.get("provided") or not loaded.get("device_key_match") or not loaded.get("schema_match"):
                raise RuntimeError("calibration roundtrip check failed")
            print(json.dumps({
                "platform": "host",
                "report": str(report),
                "calibration": str(calibration),
                "loaded_calibration": loaded,
            }, indent=2, sort_keys=True))
            return 0

    if not args.plugin_path:
        raise RuntimeError("--plugin-path is required for Android smoke runs")

    remote_dir = args.android_dir.rstrip("/")
    remote_binary = f"{remote_dir}/ov_gfx_microbench"
    remote_report = f"{remote_dir}/gfx-microbench-smoke.json"
    remote_calibration = f"{remote_dir}/gfx-calibration-smoke.json"
    run(["adb", "push", str(binary), remote_binary])
    adb_shell(f"chmod +x {shlex.quote(remote_binary)}")
    remote_cmd = (
        f"cd {shlex.quote(remote_dir)} && "
        f"GFX_PLUGIN_PATH={shlex.quote(args.plugin_path)} "
        f"LD_LIBRARY_PATH={shlex.quote(remote_dir)} "
        f"./ov_gfx_microbench --backend {shlex.quote(args.backend)} --warmup 0 --iterations 1 "
        f"--output {shlex.quote(remote_report)} --calibration-output {shlex.quote(remote_calibration)} >/dev/null"
    )
    adb_shell(remote_cmd)
    reload_cmd = (
        f"cd {shlex.quote(remote_dir)} && "
        f"GFX_PLUGIN_PATH={shlex.quote(args.plugin_path)} "
        f"LD_LIBRARY_PATH={shlex.quote(remote_dir)} "
        f"./ov_gfx_microbench --backend {shlex.quote(args.backend)} --warmup 0 --iterations 1 "
        f"--calibration-input {shlex.quote(remote_calibration)}"
    )
    payload = json.loads(adb_shell(reload_cmd))
    loaded = payload.get("loaded_calibration", {})
    if not loaded.get("provided") or not loaded.get("device_key_match") or not loaded.get("schema_match"):
        raise RuntimeError("Android calibration roundtrip check failed")
    print(json.dumps({
        "platform": "android",
        "remote_report": remote_report,
        "remote_calibration": remote_calibration,
        "loaded_calibration": loaded,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
