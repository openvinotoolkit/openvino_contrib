<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# PointPillars OpenVINO

OpenVINO export and inference for the PointPillars LiDAR detector.

The exporter supports two execution modes:

- `EmbeddedInference`: voxelization and postprocessing are embedded in the IR.
- `RuntimeInference`: the IR contains the network; `libpp_runtime.so` runs the
  surrounding OpenCL stages.

## Python Dependencies

Install the packages used by export and inference:

```bash
python3 -m pip install --upgrade numpy openvino
python3 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/xpu
```

PyTorch is needed to load the checkpoint during export. The inference wrappers
use NumPy and OpenVINO.

## Export

From the repository root:

```bash
bash ov_contrib/ov_plugins/build.sh
python3 ov_contrib/ov_export.py
python3 ov_contrib/ov_export.py --embed-custom-ops
```

By default, the exporter reads `ov_contrib/pretrained/epoch_160.pth`. Use
`--ckpt /path/to/checkpoint.pth` to select a different checkpoint.

The build regenerates plugin constants and builds `pp_extension.so` and, when
OpenCL is available, `libpp_runtime.so`. The exporter writes the IR, weights,
generated kernel configuration, and GPU configuration files.

The default locations are:

- Network-only export: `ov_contrib/pretrained/ov/`
- Embedded export: `ov_contrib/pretrained/ov_e2e/`
- Generated plugin files: `ov_contrib/ov_plugins/`
- Built libraries: `ov_contrib/ov_plugins/build/`

Use `--output-dir /path/to/output` for model files and `--plugin-dir
/path/to/ov_plugins` for generated plugin files.

For a discrete GPU, prefer `RuntimeInference` and the custom runtime library.
It keeps the network and the surrounding OpenCL stages in the runtime path.

Model geometry, decoder settings, and device tuning are configured in
[ov_config.toml](ov_config.toml); regenerate the outputs after changing it.

## Inference

The minimal example runs both modes on the same seeded random point cloud:

```bash
python3 ov_contrib/ov_infer.py
```

The classes can also be used directly from `ov_contrib`:

```python
import numpy as np

from ov_infer import EmbeddedInference, RuntimeInference

# Each row is one point: [x, y, z, intensity].
# x, y, and z are coordinates in meters. The coordinate limits match
# [pillar].point_cloud_range in ov_config.toml.
points = np.random.default_rng(0).uniform(
    low=(0.0, -39.68, -3.0, 0.0),
    high=(69.12, 39.68, 1.0, 1.0),
    size=(100_000, 4),
).astype(np.float32)

embedded_result = EmbeddedInference().infer(points)
runtime_result = RuntimeInference().infer(points)
```

`uniform` applies the corresponding `low` and `high` values to each column:

| Column | Meaning | Generated range |
| --- | --- | --- |
| 0 | `x` coordinate | `[0.0, 69.12)` meters |
| 1 | `y` coordinate | `[-39.68, 39.68)` meters |
| 2 | `z` coordinate | `[-3.0, 1.0)` meters |
| 3 | `intensity` | `[0.0, 1.0)` |

`size=(100_000, 4)` creates 100,000 points with four values per point.
The fixed seed (`0`) makes this synthetic smoke-test input reproducible. It is
not a real LiDAR frame and does not describe actual objects.

Each result contains `lidar_bboxes`, `labels`, and `scores`.

## Configuration

`ov_config.toml` contains three configuration groups:

- `[pillar]`: point-cloud range, voxel dimensions, feature sizes, and tensor capacities.
- `[decoder]`: anchor geometry, score/NMS thresholds, direction offset, and output capacities.
- `[tuning]`: device-specific OpenCL work-group sizes and kernel packing settings.

For anchor geometry changes, edit `[decoder]` in [ov_config.toml](ov_config.toml):

- `anchor_sizes`: `[width, length, height]` per class.
- `anchor_bottom_heights`: bottom z coordinate per class.
- `rotations`: anchor yaw values in radians.

Regenerate all derived files after changing the configuration:

```bash
bash ov_contrib/ov_plugins/build.sh
python3 ov_contrib/ov_export.py
python3 ov_contrib/ov_export.py --embed-custom-ops
```

If the box or direction layout changes, update the layout tuples and generated
constant definitions in [ov_export.py](ov_export.py). Update the decode equations
in [pp_postprocess.cpp](ov_plugins/pp_cpu/src/pp_postprocess.cpp) and
[pp_postproc_common.cl](ov_plugins/pp_gpu/src/kernel_selector/cl_kernels/pp_postproc_common.cl)
together.

## References

- [zhulf0804/PointPillars: A Simple PointPillars PyTorch Implementation for 3D LiDAR (KITTI) Detection.](https://github.com/zhulf0804/PointPillars)
- [NVIDIA-AI-IOT/CUDA-PointPillars: A project demonstrating how to use CUDA-PointPillars to deal with cloud points data from lidar.](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)
