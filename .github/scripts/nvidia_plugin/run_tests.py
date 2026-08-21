#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run the nvidia_plugin test policy inside an isolated GPU container."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.core import run  # noqa: E402
from openvino_plugin.tests import (  # noqa: E402
    Runtime,
    execute,
)

FUNC_SMOKE_FILTER = "*smoke*:-*dynamic*:*Dynamic*"
SANITIZER_FILTER = (
    "*smoke*:-*dynamic*:*Dynamic*:smoke_GRU*:smoke_LSTM*:smoke_TensorIterator*"
    ":*ConvBiasFusion*:*smoke*OVExecGraphImportExportTest.importExportedIENetwork*"
    ":*smoke*OVClassBasicTestP.registerNewPluginNoThrows*:*smoke*OVHoldersTest.Orders*"
    ":*smoke*IEClassBasicTestP.registerNewPluginNoThrows*"
    ":*smoke*IEClassBasicTestP.smoke_registerPluginsXMLUnicodePath*"
)


def _show_hardware() -> None:
    run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv",
        ],
    )
    run(["lscpu"])


def _run_policy(runtime: Runtime, preset: str) -> list[str]:
    if preset not in {"pr", "nightly"}:
        raise ValueError(f"unsupported test preset: {preset}")
    failures = []
    unit = runtime.executable("ov_nvidia_unit_tests")
    functional = runtime.executable("ov_nvidia_func_tests")

    if unit is None or not runtime.run("unit", (str(unit),)):
        failures.append("unit")

    functional_args = () if preset == "nightly" else (f"--gtest_filter={FUNC_SMOKE_FILTER}",)
    if functional is None or not runtime.run("func", (str(functional), *functional_args)):
        failures.append("func")

    if preset == "nightly":
        sanitizer = runtime.module_utils / "cuda-sanitizer.sh"
        if functional is None:
            failures.append("sanitizer")
        elif not sanitizer.is_file():
            print(f"::error::sanitizer script is unavailable: {sanitizer}", file=sys.stderr)
            failures.append("sanitizer")
        else:
            command = ("bash", str(sanitizer), str(functional), f"--gtest_filter={SANITIZER_FILTER}")
            if not runtime.run("sanitizer", command):
                failures.append("sanitizer")
    return failures


def main() -> None:
    execute(
        _run_policy,
        before_prepare=_show_hardware,
        plugin_config="plugins.xml",
    )


if __name__ == "__main__":
    main()
