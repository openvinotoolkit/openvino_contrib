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

## Checkpoint

The exporter requires the trained PyTorch checkpoint at
`pretrained/epoch_160.pth`, relative to this PointPillars module directory.
Example: A sample checkpoint is available in the
[zhulf0804/PointPillars repository](https://github.com/zhulf0804/PointPillars/blob/main/pretrained/epoch_160.pth):

```bash
mkdir -p pretrained
curl -L \
  https://github.com/zhulf0804/PointPillars/raw/refs/heads/main/pretrained/epoch_160.pth \
  -o pretrained/epoch_160.pth
```

To use a checkpoint stored elsewhere, pass its path with `--ckpt` when running
`ov_export.py`.

## Export

From this PointPillars module directory:

### 1. Set the configuration

- Set [ov_config.toml](ov_config.toml) for your target environment; it controls
  the model geometry, decoder settings, and device tuning used by generated files.

### 2. Build the plugin libraries

```bash
bash ov_plugins/build.sh
```

This command generates the C++ headers, OpenCL configuration, and custom-kernel
configuration from `ov_config.toml`, then builds `pp_extension.so`. When OpenCL
is available, it also builds `libpp_runtime.so`. The build files will be in
`ov_plugins/build/`.

### 3. Export the models

For the network-only model:

```bash
python3 ov_export.py
```

This command reads `pretrained/epoch_160.pth`. Use
`--ckpt /path/to/checkpoint.pth` to select a different checkpoint path. It
writes the network-only IR, weights, and GPU configuration to `pretrained/ov/`.
Voxelization and postprocessing are provided by `libpp_runtime.so` at inference
time.

For the embedded model:

```bash
python3 ov_export.py --embed-custom-ops
```

This command writes an IR that includes voxelization and postprocessing to
`pretrained/ov_e2e/`. The `pp_extension.so` built in step 2 is required because
the embedded operations are registered while the model is exported.

Both export commands also refresh the generated plugin files in `ov_plugins/`.

The default locations are:

- Network-only export: `pretrained/ov/`
- Embedded export: `pretrained/ov_e2e/`
- Generated plugin files: `ov_plugins/`
- Built libraries: `ov_plugins/build/`

Use `--output-dir /path/to/output` for model files and `--plugin-dir
/path/to/ov_plugins` for generated plugin files.

For a discrete GPU, prefer `RuntimeInference` and the custom runtime library.

Model geometry, decoder settings, and device tuning are configured in
[ov_config.toml](ov_config.toml); regenerate the outputs after changing it.

## Inference

The minimal example runs both modes on the same seeded random point cloud:

```bash
python3 ov_infer.py
```

The classes can also be used directly from this module directory:

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
Modify or replace this generated input for your use case, while keeping the
`[x, y, z, intensity]` format and the configured point-cloud range.

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
bash ov_plugins/build.sh
python3 ov_export.py
python3 ov_export.py --embed-custom-ops
```

If the box or direction layout changes, update the layout tuples and generated
constant definitions in [ov_export.py](ov_export.py). Update the decode equations
in [pp_postprocess.cpp](ov_plugins/pp_cpu/src/pp_postprocess.cpp) and
[pp_postproc_common.cl](ov_plugins/pp_gpu/src/kernel_selector/cl_kernels/pp_postproc_common.cl)
together.

## References

- [zhulf0804/PointPillars: A Simple PointPillars PyTorch Implementation for 3D LiDAR (KITTI) Detection.](https://github.com/zhulf0804/PointPillars)
- [NVIDIA-AI-IOT/CUDA-PointPillars: A project demonstrating how to use CUDA-PointPillars to deal with cloud points data from lidar.](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)
