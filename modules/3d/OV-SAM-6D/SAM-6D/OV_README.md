# SAM-6D OpenVINO Setup and Inference Guide

This guide explains how to set up and run the OpenVINO-based SAM-6D pipeline on Ubuntu 24.

---

# Prerequisites

- Ubuntu 24.04
- Conda installed
- Intel GPU drivers installed (for GPU inference)
- Python 3.10+
- Internet connection (required for model and Blender downloads)

---

# 1. Create and Activate Conda Environment

```bash
cd JiehongLin-SAM-6D
cd SAM-6D

conda env create -f ov_environment_u24.yaml
conda activate ov_sam6d
```

---

# 2. Install OpenVINO Nightly Build

Download the OpenVINO nightly build package from:

https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/

Download:

```text
openvino_toolkit_ubuntu24_2026.3.0.dev20260519_x86_64.tgz
```

Extract the package:

```bash
tar -xvf openvino_toolkit_ubuntu24_2026.3.0.dev20260519_x86_64.tgz
```

Source the OpenVINO environment:

```bash
source openvino_toolkit_ubuntu24_2026.3.0.dev20260519_x86_64/setupvars.sh
```

> NOTE:
> You must source `setupvars.sh` every new terminal session before running OpenVINO commands.

---

# 3. Install Eigen Library

```bash
sudo apt update
sudo apt install -y libeigen3-dev
```

---

# 4. Render Templates

Navigate to the render directory:

```bash
cd openvino_contrib/modules/3d/OV-SAM-6D/SAM-6D/Render
```

## Render Templates Automatically

The first run automatically downloads Blender 3.3.1:

```bash
blenderproc run render_custom_templates.py \
    --output_dir ../Data/Example/outputs \
    --cad_path ../Data/Example/obj_000005.ply
```

---

## Optional: If Blender Download Failed

Install Blender manually and use:

```bash
wget https://download.blender.org/release/Blender3.3/blender-3.3.1-linux-x64.tar.xz
tar -xvf blender-3.3.1-linux-x64.tar.xz
blenderproc run \
    --custom-blender-path /path/to/blender-3.3.1-linux-x64 \
    render_custom_templates.py \
    --output_dir ../Data/Example/outputs \
    --cad_path ../Data/Example/obj_000005.ply
```

---

# 5. Instance Segmentation Model (ISM)

Navigate to the ISM directory:

```bash
cd openvino_contrib/modules/3d/OV-SAM-6D/SAM-6D/Instance_Segmentation_Model
```

## Download Required Models

```bash
python download_sam.py
python download_fastsam.py
python download_dinov2.py
```

---

## Export PyTorch Models to OpenVINO IR

```bash
python export_ism.py \
    --ext \
    --output_dir ./checkpoints/ov_models \
    --sam_checkpoint_dir ./checkpoints/segment-anything \
    --dinov2_checkpoint_dir ./checkpoints/dinov2 \
    --fastsam_checkpoint ./checkpoints/FastSAM/FastSAM-x.pt
```

---

## Run Instance Segmentation Inference

## Run ISM, FP32 baseline

```bash
python infer_ism_ov.py \
    --ov_model_dir ./checkpoints/ov_models \
    --ov_device GPU \
    --precision fp32 \
    --ext \
    --segmentor_model fastsam \
    --image ../Data/Example/rgb.png \
    --cad ../Data/Example/obj_000005.ply \
    --templates_dir ../Data/Example/outputs/templates \
    --output_dir ../Data/Example/outputs/sam6d_results \
    --gt_mask ../Data/Example/mask_visib
```
## Run ISM, FP16 recommended mode

```bash
python infer_ism_ov.py \
    --ov_model_dir ./checkpoints/ov_models \
    --ov_device GPU \
    --precision fp16 \
    --ext \
    --segmentor_model fastsam \
    --image ../Data/Example/rgb.png \
    --cad ../Data/Example/obj_000005.ply \
    --templates_dir ../Data/Example/outputs/templates \
    --output_dir ../Data/Example/outputs/sam6d_results \
    --gt_mask ../Data/Example/mask_visib
```


---

# 6. Download Pose Estimation Model

```bash
cd openvino_contrib/modules/3d/OV-SAM-6D/SAM-6D/Pose_Estimation_Model

python download_sam6d-pem.py
```

---

# 7. Build OpenVINO PointNet2 Operator

Navigate to the OpenVINO PointNet2 operator directory:

```bash
cd openvino_contrib/modules/3d/OV-SAM-6D/SAM-6D/Pose_Estimation_Model/model/ov_pointnet2_op
```

Create build directory:

```bash
mkdir build
cd build
```

Run CMake:

```bash
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-fPIC" \
    -DOpenVINO_DIR=$(python3 -c "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')") \
    ../
```

Build the project:

```bash
make
```

---

# 8. Install PointNet2

```bash
cd openvino_contrib/modules/3d/OV-SAM-6D/SAM-6D/Pose_Estimation_Model/model/pointnet2

python setup.py install
```

---

# 9. Convert ONNX Models to OpenVINO IR

Navigate to Pose Estimation Model directory:

```bash
cd openvino_contrib/modules/3d/OV-SAM-6D/SAM-6D/Pose_Estimation_Model
```

Run conversion:

```bash
python pem_model_convert_ov_ir.py
```

---

# 10. Run Pose Estimation Inference

# FP32 baseline:

```bash
cd openvino_contrib/modules/3d/OV-SAM-6D/SAM-6D/Pose_Estimation_Model

python run_inference_custom_openvino.py \
    --device GPU \
    --precision fp32 \
    --topk_ism_score 1
```
# FP16 recommended mode:

```bash
cd openvino_contrib/modules/3d/OV-SAM-6D/SAM-6D/Pose_Estimation_Model

python run_inference_custom_openvino.py \
    --device GPU \
    --precision fp16 \
    --topk_ism_score 1
```


---

# Expected Output

Inference outputs will be generated inside:

```text
SAM-6D/Data/Example/outputs/
```

Results may include:

- Rendered templates
- Segmentation masks
- Pose estimation outputs
- Visualization images

---


# Troubleshooting

## OpenVINO Environment Not Found

Re-source the environment:

```bash
source openvino_toolkit_ubuntu24_2026.3.0.dev20260519_x86_64/setupvars.sh
```

---

## GPU Device Not Detected

Verify OpenVINO GPU runtime installation:

```bash
python -c "from openvino import Core; print(Core().available_devices)"
```

Expected output example:

```text
['CPU', 'GPU.0']
```

---

# Notes

- The generated templates are required before running inference.
- OpenVINO GPU inference requires Intel GPU support.
- BlenderProc downloads Blender only during the first run.
- Ensure all checkpoints are downloaded successfully before exporting models.


