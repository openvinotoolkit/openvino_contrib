# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Numerical smoke test for the FlashOCC BEV pooling extension."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import openvino as ov
import pytest
from onnx import TensorProto, helper


MODULE_DIR = Path(__file__).resolve().parents[1]
EXTENSION_DIR = MODULE_DIR / "openvino_extensions" / "bev_pool"
GPU_CONFIG = EXTENSION_DIR / "bev_pool_gpu.xml"

NUM_CAMS = 6
DEPTH_BINS = 44
FEAT_H = 16
FEAT_W = 44
CHANNELS = 64
NX = 200
NY = 200
HW = FEAT_H * FEAT_W
DEPTH_HW = DEPTH_BINS * HW
TOTAL_POINTS = NUM_CAMS * DEPTH_HW


@pytest.fixture(scope="session")
def bevpool_extension(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the extension, unless CI explicitly supplies a prebuilt library."""
    supplied = os.environ.get("FLASHOCC_BEVPOOL_EXTENSION")
    if supplied:
        extension = Path(supplied).resolve()
        assert extension.is_file(), f"BEV pool extension does not exist: {extension}"
        return extension

    build_dir = tmp_path_factory.mktemp("bevpool_extension_build")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(EXTENSION_DIR),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", "2"],
        check=True,
    )
    extension = build_dir / "libopenvino_bevpool_extension.so"
    assert extension.is_file(), f"Build did not produce {extension}"
    return extension


def _build_bevpool_model(path: Path) -> Path:
    depth = helper.make_tensor_value_info(
        "depth", TensorProto.FLOAT, [NUM_CAMS, DEPTH_BINS, FEAT_H, FEAT_W]
    )
    feat = helper.make_tensor_value_info(
        "feat", TensorProto.FLOAT, [NUM_CAMS, CHANNELS, FEAT_H, FEAT_W]
    )
    geom = helper.make_tensor_value_info("geom", TensorProto.FLOAT, [TOTAL_POINTS, 3])
    bev = helper.make_tensor_value_info("bev", TensorProto.FLOAT, [1, CHANNELS, NY, NX])
    graph = helper.make_graph(
        [
            helper.make_node("BEVPoolBinSort", ["geom"], ["packed"], domain="bevfusion"),
            helper.make_node("BEVPoolV2", ["depth", "feat", "packed"], ["bev"], domain="bevfusion"),
        ],
        "flashocc_bevpool_smoke",
        [depth, feat, geom],
        [bev],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13), helper.make_operatorsetid("bevfusion", 1)],
        producer_name="openvino-flashocc-smoke-test",
    )
    model.ir_version = 7
    onnx.save(model, path)
    return path


def _fixed_inputs() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Create sparse deterministic data with collisions and asymmetric x/y cells."""
    depth = np.zeros((NUM_CAMS, DEPTH_BINS, FEAT_H, FEAT_W), dtype=np.float32)
    feat = np.zeros((NUM_CAMS, CHANNELS, FEAT_H, FEAT_W), dtype=np.float32)
    geom = np.full((TOTAL_POINTS, 3), -100.0, dtype=np.float32)
    expected = np.zeros((1, CHANNELS, NY, NX), dtype=np.float32)

    # (camera, depth bin, flattened feature pixel, ix, iy, depth probability)
    points = [
        (0, 0, 0, 2, 7, 0.25),
        (0, 1, 0, 2, 7, 0.75),  # same feature and cell: exercises accumulation
        (1, 2, 5, 2, 7, 0.50),
        (2, 3, 9, 11, 3, 0.20),  # asymmetric cell catches X/Y layout regressions
        (5, 43, HW - 1, 199, 198, 1.00),
    ]
    feature_values = {
        (0, 0): {0: 2.0, 3: -1.0},
        (1, 5): {1: 4.0, 7: 0.5},
        (2, 9): {2: -3.0, 63: 1.25},
        (5, HW - 1): {0: -0.5, 12: 8.0},
    }

    for (camera, hw), channels in feature_values.items():
        h, w = divmod(hw, FEAT_W)
        for channel, value in channels.items():
            feat[camera, channel, h, w] = value

    for camera, depth_bin, hw, ix, iy, probability in points:
        h, w = divmod(hw, FEAT_W)
        point_index = camera * DEPTH_HW + depth_bin * HW + hw
        depth[camera, depth_bin, h, w] = probability
        geom[point_index] = (-40.0 + (ix + 0.25) * 0.4, -40.0 + (iy + 0.25) * 0.4, 0.0)
        for channel, value in feature_values[(camera, hw)].items():
            expected[0, channel, iy, ix] += probability * value

    return {"depth": depth, "feat": feat, "geom": geom}, expected


def _test_devices(core: ov.Core) -> list[str]:
    forced = os.environ.get("FLASHOCC_BEVPOOL_TEST_DEVICE")
    if forced:
        return [forced]
    devices = ["CPU"]
    if any(device.upper().startswith("GPU") for device in core.available_devices):
        devices.append("GPU")
    return devices


def test_binsort_and_bevpool_v2_match_cpu_reference(
    bevpool_extension: Path, tmp_path: Path
) -> None:
    """Run BinSort+V2 and compare every BEV element with an independent reference."""
    core = ov.Core()
    core.add_extension(str(bevpool_extension))
    model = core.read_model(str(_build_bevpool_model(tmp_path / "bevpool_smoke.onnx")))
    inputs, expected = _fixed_inputs()

    for device in _test_devices(core):
        config = {"CONFIG_FILE": str(GPU_CONFIG)} if device.upper().startswith("GPU") else {}
        compiled = core.compile_model(model, device, config)
        actual = compiled(inputs)[compiled.output(0)]
        # The SimpleGPU layer may use FP16 storage according to the plugin's
        # selected execution precision. CPU remains a strict FP32 reference.
        atol = 5e-4 if device.upper().startswith("GPU") else 1e-6
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=atol,
            err_msg=f"BEVPoolBinSort + BEVPoolV2 mismatch on {device}",
        )
