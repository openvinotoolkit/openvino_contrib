# BEVFusion Export and Inference Guide

This guide explains how to export BEVFusion models to OpenVINO and run inference with OpenCL acceleration on Intel GPUs.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Model Export](#model-export)
- [Building OpenCL Extensions](#building-opencl-extensions)
- [Running Inference](#running-inference)

---

## Prerequisites

### Software Requirements
```bash
# Python environment
Python 3.13 (or 3.10+)
torch>=2.0.0
openvino>=2024.0
numpy>=1.24.0

# System libraries
OpenCL (Intel Compute Runtime)
OpenMP
```

### Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Intel Compute Runtime for OpenCL
# On Ubuntu:
sudo apt-get install intel-opencl-icd
```

---

## Building OpenVino Extensions

### Overview
The pipeline uses custom OpenCL kernels for:
1. **Voxelization**: Convert point cloud to voxel grid
2. **BEV Pool**: Camera-to-BEV transformation
3. **Sparse Convolution**: 3D sparse encoder

### Build Commands
```bash
# Build all OpenCL extensions
python setup_opencl.py build_ext --inplace

# Verify build
ls openvino_extensions/<kernel>/build/*
# Should see .so files.

### Build Troubleshooting
```bash
# If build fails, check:
# 1. OpenCL headers
sudo apt-get install opencl-headers ocl-icd-opencl-dev

# 2. Verify Intel GPU
clinfo | grep "Device Name"
# Should show: Intel(R) Graphics [0xb0b0] or similar
```


## Model Export

### Step 1: Prepare Pre-trained Weights
Download the BEVFusion pre-trained model:
```bash
# Place the checkpoint at:
pretrained_models/
```

### Step 2: Export OpenVino Model
```bash
# Export with SwinT backbone
python3 export.py --checkpoint pretrained_models/bevfusion-det.pth --output-dir openvino_model --data-pkl /path/to/dataset_infos_val.pkl --data-root /path/to/dataset_root/

# Export with ResNet backbone
python3 export.py --checkpoint pretrained_models/bevfusion-det.pth --output-dir openvino_model --data-pkl /path/to/dataset_infos_val.pkl --data-root /path/to/dataset_root/ --resnet
```

**Directory structure after export:**
```
openvino_models/
├── camera_backbone_neck.onnx / .xml / .bin
├── vtransform_depthnet.onnx / .xml / .bin
├── vtransform_dtransform.onnx / .xml / .bin
├── vtransform_downsample.onnx / .xml / .bin
├── fuser.onnx / .xml / .bin
├── bev_decoder.onnx / .xml / .bin
├── transfusion_head.onnx / .xml / .bin
├── full_detection.onnx / .xml / .bin
├── bev_pool.xml
├── sparse_encoder.xml / .bin
├── voxelize.xml
└── sparse_encoder_weights/
    ├── conv_input_0_weight.npy
    ├── conv_input_1_*.npy
    ├── encoder_layers_*.npy
    └── conv_out_*.npy
    └── ....
    
```

---

## Running Inference

### Quick Inference (Modular Pipeline)
```bash
python3 run_inference_standalone.py --model-dir openvino_model --num-samples 5 --data-pkl /path/to/dataset_infos_val.pkl --data-root /path/to/dataset_root

# Adjust detection threshold
python3 run_inference_standalone.py --model-dir openvino_model --num-samples 5 --data-pkl /path/to/dataset_infos_val.pkl --data-root /path/to/dataset_root --score-threshold 0.2
```

### Inference with Evaluation Metrics (mAP + Throughput)
```bash
# Full evaluation on validation set
python evaluate_map_throughput.py --num-samples 0 --device GPU --model-dir <model_output_path> --data-pkl /path/to/dataset_infos_val.pkl --data-root /path/to/dataset_root

# Quick 5-sample test
python evaluate_map_throughput.py \
    --num-samples 5 \
    --score-threshold 0.2 \
    --data-pkl /path/to/dataset_infos_val.pkl \
    --data-root /path/to/dataset_root
```

## References
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [Intel Compute Runtime](https://github.com/intel/compute-runtime)
- [BEVFusion Paper](https://arxiv.org/abs/2205.13542)