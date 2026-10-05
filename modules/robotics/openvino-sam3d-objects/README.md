<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# SAM 3D Objects — OpenVINO Enablement

OpenVINO-accelerated inference for **SAM 3D Objects** (Single-image 3D reconstruction from Meta).

## Architecture Overview

SAM3D generates 3D objects from a single RGBA image via a two-stage flow-matching approach:

1. **Sparse Structure (SS)**: DINOv2 image encoding → Dense DiT denoising (25 Euler steps with CFG) → Conv3D VAE decoder → 64³ occupancy grid

2. **Structured Latent (SLAT)**: Sparse DiT denoising on occupied voxels → Per-voxel 8-dim latent features → Gaussian/Mesh decoder

### Custom OpenVINO Extensions

- **SparseConv3dEngine** (`openvino_extensions/sparse_conv_3d/`):
  Monolithic C++/OpenCL extension handling all sparse 3D convolution operations (SubMConv, strided down, inverse up, linear, residual blocks).

- **Vox2Seq** (`openvino_extensions/vox2seq/`):
  Space-filling curve serialization (Z-order/Morton, Hilbert) for serialized sparse attention modes.

## Directory Structure

```
openvino-sam3d-objects/
├── export.py                       # Export PyTorch → OV IR
├── run_inference_standalone.py     # Standalone inference 
├── eval_accuracy.py                # Evaluate accuracy of OV models
├── setup_opencl.py                 # Build custom OpenCL extensions
├── requirements.txt
├── EXPORT_AND_INFERENCE_GUIDE.md   # Step-by-step export & inference guide
├── sam3d/
│   ├── __init__.py
│   └── config.py                   # All constants and helpers
├── openvino_extensions/
│   ├── sparse_conv_3d/
│   └── vox2seq/
├── kid_box/                        # Input data
```

## References

- [Export & Inference Guide](EXPORT_AND_INFERENCE_GUIDE.md)
- [SAM 3D Objects (Meta)](https://github.com/facebookresearch/sam-3d-objects)
- [OpenVINO Custom Operations Guide](https://docs.openvino.ai/2024/documentation/openvino-extensibility.html)
