<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Note
_This work depends on [CDPN_ICCV2019_ZhigangLi](https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi) (commit 625f9a8)._

Can be cited as:

```bibtex
@inproceedings{li2019cdpn,
  title={CDPN: Coordinates-based disentangled pose network for real-time rgb-based 6-dof object pose estimation},
  author={Li, Zhigang and Wang, Gu and Ji, Xiangyang},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={7678--7687},
  year={2019}
}
```
-----------------------
Run the setup script from this directory to prepare the `cdpn_repo` for OpenVINO inference:

```bash
bash setup.sh
```

It clones the upstream CDPN repo (pinned to commit `625f9a8`), applies `cdpn_changes.patch`, and copies the OpenVINO-specific files from `copy_files_to_cdpn_repo.txt`.
<details>
<summary>Manual steps (equivalent to setup.sh)</summary>

```bash
# in openvino_contrib/modules/3d/CDPN/
git clone https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi cdpn_repo
cd cdpn_repo
git checkout 625f9a8
patch -p1 --forward < ../cdpn_changes.patch  # strip the top-level a/ and b/ prefixes from the diff paths
grep -v -E '^[[:space:]]*(#|$)' ../copy_files_to_cdpn_repo.txt | while IFS= read -r f; do mkdir -p "$(dirname "./$f")" && cp "../$f" "./$f"; done
python -m pip install --upgrade torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/xpu
python -m pip install --upgrade -r requirements.txt
```
</details>
-----------------------

<details>
<summary style="font-size:1.5em; font-weight:600">XPU Inference</summary>

To run the inference in intel xpu for lm_full dataset the following command can be used:
```bash
python xpu_infer.py \
   --cfg tools/exps_cfg/config_rot_trans.yaml \
   --load_model checkpoints/stage3.checkpoint \
   --xpu 0 \
   --dataset_dir dataset/lm_full \
   --batch_size 4
```
</details>

<details>
<summary style="font-size:1.5em; font-weight:600">OpenVINO Inference</summary>

Build the plugins:
```bash
bash ov_plugins/build.sh
```

Export the model to OpenVINO IR format:
```bash
python ov_export.py \
   --cfg tools/exps_cfg/config_rot_trans.yaml \
   --load_model /workspace/checkpoints/stage3.checkpoint \
   --output_dir checkpoints \
   --basename cdpn_stage3 \
   --verify
```

For extended nn (pre/post-processing till pnpsolve in OV graph), add `--extnn --extension ov_plugins/build/cdpn_extension.so` and for end-to-end (full graph with custom ops through OV), add `--e2e --extension ov_plugins/build/cdpn_extension.so`.

To run the inference through OpenVINO for lm_full dataset the following command can be used:
```bash
python ov_infer.py \
   --model checkpoints/<cdpn_stage3|cdpn_stage3_extnn|cdpn_stage3_e2e>.xml \
   --dataset_dir dataset/lm_full \
   --<cpu|gpu> \
   --batch_size 4
```

FP16 (mixed precision) keeps sensitive layers in FP32 while running the bulk of the network in FP16. Can be used when faster GPU inference with minimal accuracy loss is required.

Export:
```bash
python ov_export.py \
   --cfg tools/exps_cfg/config_rot_trans.yaml \
   --load_model checkpoints/stage3.checkpoint \
   --output_dir checkpoints \
   --basename cdpn_stage3 \
   --fp16_nn
```

Inference:
```bash
python ov_infer.py \
   --model checkpoints/cdpn_stage3_fp16.xml \
   --dataset_dir dataset/lm_full \
   --batch_size 4 \
   --infer_precision f16 \
   --gpu
```

INT8 (quantized) uses NNCF post-training quantization to shrink the model and maximize throughput. It requires a calibration dataset and is best suited for GPU or NPU deployments where throughput matters most.

Export:
```bash
python ov_export.py \
   --cfg tools/exps_cfg/config_rot_trans.yaml \
   --load_model checkpoints/stage3.checkpoint \
   --output_dir checkpoints \
   --basename cdpn_stage3 \
   --dataset_dir dataset/lm_full \
   --int8_nn \
   --int8_subset_size 300 \
   --int8_target_device GPU
```

Inference:
```bash
python ov_infer.py \
   --model checkpoints/cdpn_stage3_int8.xml \
   --dataset_dir dataset/lm_full \
   --int8_nn \
   --infer_precision none \
   --batch_size 4 \
   --gpu
```
</details>
