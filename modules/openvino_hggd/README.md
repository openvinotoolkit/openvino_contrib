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

## Output

Results are written under `output/scene_<scene_id>/`.

```
output/scene_100/
├── inference.log
├── eval_result.npy
├── logs/
└── pred/
```
