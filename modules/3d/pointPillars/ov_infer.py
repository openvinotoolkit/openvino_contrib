#!/usr/bin/env python3

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Run the embedded or external PointPillars inference pipeline."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import openvino as ov

from ov_export import CHECKPOINT_BOX_VALUES, DETECTION_RECORD

ROOT = Path(__file__).resolve().parent

# Detection-record columns are defined by ov_export.py. A zero score ends the list.
COLUMN = {name: index for index, name in enumerate(DETECTION_RECORD)}
DETECTION_CHANNELS = len(DETECTION_RECORD)


def _to_result(rows):
    rows = np.asarray(rows, np.float32).reshape(-1, DETECTION_CHANNELS)
    rows = rows[rows[:, COLUMN["score"]] > 0.0]

    # Convert decoder records to the training model's box layout.
    bboxes = np.empty((len(rows), len(CHECKPOINT_BOX_VALUES)), np.float32)
    for index, name in enumerate(CHECKPOINT_BOX_VALUES):
        bboxes[:, index] = rows[:, COLUMN[name]]
    bboxes[:, CHECKPOINT_BOX_VALUES.index("z")] -= rows[:, COLUMN["height"]] / 2

    return {
        "lidar_bboxes": bboxes,
        "labels": np.ascontiguousarray(rows[:, COLUMN["class_id"]]).view(np.int32).astype(np.int64),
        "scores": np.ascontiguousarray(rows[:, COLUMN["score"]]),
    }


class EmbeddedInference:
    """Inference using an IR with custom operations embedded in the graph."""

    def __init__(self, model_xml=ROOT / "pretrained" / "ov_e2e" / "epoch_160.xml",
                 device="GPU", config_json=None,
                 extension=ROOT / "ov_plugins" / "build" / "pp_extension.so"):
        model_xml = Path(model_xml)
        if config_json is None:
            config_json = model_xml.parent / "bench_gpu_config.json"

        core = ov.Core()
        core.add_extension(str(extension))
        properties = json.loads(Path(config_json).read_text()).get(device, {})
        if properties:
            core.set_property(device, properties)

        compiled = core.compile_model(str(model_xml), device)
        self.request = compiled.create_infer_request()
        self.points = self.request.get_tensor(compiled.input("points"))
        self.num_points = self.request.get_tensor(compiled.input("num_points"))
        self.detections = compiled.output("detections")

    def infer(self, points):
        points = np.ascontiguousarray(points, np.float32).reshape(-1, 4)
        count = min(len(points), self.points.data.shape[0])
        self.points.data[:count] = points[:count]
        self.num_points.data[0] = count

        self.request.infer()
        return _to_result(self.request.get_tensor(self.detections).data)


class RuntimeInference:
    """Inference using a network-only IR and the external runtime library."""

    def __init__(self, model_xml=ROOT / "pretrained" / "ov" / "epoch_160.xml",
                 library=ROOT / "ov_plugins" / "build" / "libpp_runtime.so",
                 kernel_dir=ROOT / "ov_plugins" / "pp_gpu" / "src" / "kernel_selector" / "cl_kernels"):
        self.library = ctypes.CDLL(str(library))
        self.library.pp_runtime_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.library.pp_runtime_create.restype = ctypes.c_void_p
        self.library.pp_runtime_destroy.argtypes = [ctypes.c_void_p]
        self.library.pp_runtime_forward.argtypes = [ctypes.c_void_p,
                                                    np.ctypeslib.ndpointer(np.float32, ndim=2,
                                                                           flags="C_CONTIGUOUS"),
                                                    ctypes.c_int]
        self.library.pp_runtime_forward.restype = ctypes.POINTER(ctypes.c_float)

        self.runtime = self.library.pp_runtime_create(str(model_xml).encode(), str(kernel_dir).encode())
        if not self.runtime:
            raise RuntimeError(f"pp_runtime could not start on {model_xml}")
        self.shape = (self.library.pp_runtime_max_detections(),
                      self.library.pp_runtime_detection_stride())

    def infer(self, points):
        points = np.ascontiguousarray(points, np.float32).reshape(-1, 4)
        rows = self.library.pp_runtime_forward(self.runtime, points, len(points))
        if not rows:
            raise RuntimeError("pp_runtime failed on this frame")
        return _to_result(np.ctypeslib.as_array(rows, self.shape))

    def __del__(self):
        if getattr(self, "runtime", None):
            self.library.pp_runtime_destroy(self.runtime)
            self.runtime = None


def main():
    """Example: Run both inference wrappers on the same seeded random point cloud."""
    np.random.seed(0)
    points = np.random.random((100_000, 4)).astype(np.float32)

    for name, inference in (("embedded", EmbeddedInference()), ("runtime", RuntimeInference())):
        result = inference.infer(points)
        print(f"{name}: {len(result['scores'])} detections")


if __name__ == "__main__":
    main()
