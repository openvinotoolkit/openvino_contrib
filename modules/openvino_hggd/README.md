<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# HGGD — OpenVINO iGPU Enablement

OpenVINO inference enablement for [HGGD](https://github.com/THU-VCLab/HGGD) (Hybrid Grasp Detection and Generation) on Intel integrated GPU.

## Prerequisites: Fetch Upstream HGGD Files

The `customgraspnetAPI`, `dataset`, and `models` directories are not included in this module — they are identical to the upstream repository and should be copied directly from there:

```bash
git clone https://github.com/THU-VCLab/HGGD /tmp/HGGD_upstream
cp -r /tmp/HGGD_upstream/customgraspnetAPI path_to_openvino_hggd/
cp -r /tmp/HGGD_upstream/dataset          path_to_openvino_hggd/
cp -r /tmp/HGGD_upstream/models           path_to_openvino_hggd/
```

## Setup

```bash
cd path_to_openvino_hggd
bash setup.sh
conda activate hggd_intel
```

## Export Models

```bash
cd path_to_openvino_hggd
conda activate hggd_intel

python export_models.py \
  --checkpoint-path /path/to/HGGD_realsense_checkpoint \
  --output-dir openvino_models \
  --ov-device GPU
```

## Run

```bash
cd path_to_openvino_hggd
conda activate hggd_intel

bash run.sh \
  /path/to/HGGD_realsense_checkpoint \
  /path/to/dataset/6dto2drefine_realsense \
  /path/to/graspnet \
  100 GPU
```

## Evaluation

Inference dumps per-frame grasp predictions under
`output/scene_<scene_id>/pred/`. To compute **Collision-Free AP** and
the collision rate, run the standalone evaluator:

```bash
cd path_to_openvino_hggd
conda activate hggd_intel

python evaluate.py \
  --scene-path /path/to/graspnet \
  --pred-dir output/scene_100/pred \
  --scene-l 100 --scene-r 101
```

The evaluator runs the standard GraspNet grasp pipeline (NMS, object
assignment, gripper-box collision and empty-grasp test, AP aggregation)
and counts a grasp as a success when it does not collide with any scene
point and is not "empty" (at least 10 object points held between the
fingers). It is fully standalone (numpy + open3d + transforms3d +
grasp-nms) and only needs `--scene-path` to contain `scenes/` and
`models/%03d/nontextured.ply` (no textured models required);
`--scene-path` is the same `/path/to/graspnet` passed to `run.sh`.
Results are printed to the console and saved to `eval_result_cf.npy`
next to the prediction directory.

Useful options: `--camera realsense` (default), `--top-k 50`,
`--proc N` (worker processes), `--result-path /custom/location.npy`.

Note: CF-AP is **not comparable** to the AP / AP@0.8 / AP@0.4 reported
in the HGGD paper (it has no friction-coefficient semantics) — treat it
as a separate metric.

To reproduce the AP / AP@0.8 / AP@0.4 as reported in the HGGD paper,
use the official evaluation toolkit from `graspnetAPI==1.2.11`
(`pip install --no-deps graspnetAPI==1.2.11`), which evaluates the
same prediction dumps from this module.

## Output

Results are written under `output/scene_<scene_id>/`.

```
output/scene_100/
├── inference.log
├── eval_result_cf.npy   # written by evaluate.py
├── logs/
└── pred/
```
