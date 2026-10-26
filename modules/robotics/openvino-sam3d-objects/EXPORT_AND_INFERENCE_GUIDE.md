<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# SAM 3D Objects — Export and Inference Guide

This guide explains how to export SAM 3D Objects models to OpenVINO IR and run
inference with OpenCL acceleration on Intel GPUs.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Building Custom Extensions](#building-custom-extensions)
- [Model Export](#model-export)
- [Running Inference](#running-inference)

---

## Prerequisites

### Software Requirements
```
Python >= 3.10
OpenVINO >= 2025.0  (target: 2026.1.0)
CMake >= 3.18
OpenCL development headers + Intel GPU driver

# Export only (not needed for standalone inference)
torch >= 2.0
safetensors
omegaconf
hydra-core
onnx
onnxscript
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
sudo apt-get install intel-opencl-icd opencl-headers ocl-icd-opencl-dev
```

### Verify Intel GPU
```bash
clinfo | grep "Device Name"
# Should show: Intel(R) Graphics [...] or similar
```

---

## Building Custom Extensions

The pipeline uses two custom OpenVINO C++/OpenCL extensions:

| Extension | Library | Purpose |
|---|---|---|
| **SparseConv3d** | `libopenvino_sparse_conv_3d_extension.so` | All sparse 3D convolution ops (SubMConv, strided down/up, residual blocks) used in the SLAT generator |
| **Vox2Seq** | `libov_vox2seq.so` | Space-filling curve serialisation (Z-order/Morton, Hilbert) for serialised sparse attention |

### Build All Extensions
```bash
# Build and verify both extensions
python setup_opencl.py --verify

# Build only (no verification)
python setup_opencl.py

# Clean rebuild
python setup_opencl.py --clean --verify
```

Built libraries land in:
```
openvino_extensions/sparse_conv_3d/build/libopenvino_sparse_conv_3d_extension.so
openvino_extensions/vox2seq/build/libov_vox2seq.so
```

### Manual CMake Build (if needed)
```bash
for EXT in sparse_conv_3d vox2seq; do
    cd openvino_extensions/$EXT
    rm -rf build && mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd ../../..
done
```

### Build Troubleshooting
```bash
# Missing OpenCL headers
sudo apt-get install opencl-headers ocl-icd-opencl-dev

# Verify Intel GPU
clinfo | grep "Device Name"
# Should show: Intel(R) Graphics [...] or similar

# Verify extension loads correctly
python -c "
import openvino as ov
ov.Core().add_extension('openvino_extensions/sparse_conv_3d/build/libopenvino_sparse_conv_3d_extension.so')
print('OK')
"
```

---

## Model Export

### Step 1: Prepare Checkpoints

Download or place the SAM3D checkpoints in a directory. The export script
expects the following files under `<workspace>/checkpoints/` (or a path
specified with `--ckpt-dir`):

```
checkpoints/
├── ss_generator.ckpt
├── slat_generator.ckpt
└── ss_decoder.ckpt
```

HuggingFace weights can be placed under `checkpoints/hf/`:
```
checkpoints/hf/
```

### Step 2: Export to OpenVINO IR
```bash
python export.py \
    --workspace ./checkpoints/hf \
    --output-dir ./exported_models
```

**Optional flags:**

| Flag | Description |
|---|---|
| `--ckpt-dir <path>` | Override checkpoint directory (default: `<workspace>/checkpoints`) |
| `--output-dir <path>` | Output directory for IR files (default: `./exported_models`) |
| `--skip-dino` | Skip DINOv2 condition embedder export |
| `--skip-decoders` | Skip SLAT decoder export (Gaussian Splatting + mesh) |
| `--skip-moge` | Skip MoGe pointmap model export (requires `moge` package) |
| `--hf-cache <path>` | HuggingFace cache directory for MoGe download |

**Directory structure after export:**
```
exported_models/
├── config.json
├── moge.xml / .bin                          # MoGe pointmap model
│
│   # ── Sparse-structure (SS) stage ──
├── ss_dino_image.xml / .bin                 # DINOv2 (RGB stream)
├── ss_dino_mask.xml / .bin                  # DINOv2 (mask stream)
├── ss_pointpatch.xml / .bin                 # PointPatchEmbed (inner ViT)
├── ss_pointpatch_outer.xml / .bin           # PointPatchEmbed (outer proj)
├── ss_pointpatch_outer_weights.npz
├── ss_embedder_proj_{0,1,2}.xml / .bin      # Condition embedder projection nets
├── ss_idx_emb.npy                           # Positional embedding lookup table
├── ss_backbone.xml / .bin                   # SS dense DiT backbone
├── ss_latent_in_all.xml / .bin              # SS latent input projections
├── ss_latent_out_all.xml / .bin             # SS latent output projections
├── ss_decoder.xml / .bin                    # SS Conv3D VAE decoder
│
│   # ── Structured-latent (SLAT) stage ──
├── slat_dino_{0,1}.xml / .bin               # SLAT DINOv2 embedders
├── slat_embedder_proj_{0,1}.xml / .bin      # SLAT condition projections
├── slat_idx_emb.npy                         # SLAT positional embeddings
├── slat_input_layer.xml / .bin              # SLAT sparse input/output layers
├── slat_out_layer.xml / .bin
├── slat_t_emb_all.xml / .bin                # Timestep + IO embedding nets
├── slat_attn_blocks.xml / .bin              # SLAT dense attention blocks
├── slat_downsample.xml / .bin               # Sparse 2x down/up sampling
├── slat_upsample.xml / .bin
├── slat_io_{input,out}_blocks_{0,1}_fused_resblock.xml / .bin
├── slat_sparse_io_manifest.json             # Sparse IO extension manifest
├── slat_sparse_io_weights/                  # Packed sparse-conv weights
│
│   # ── Decoders ──
├── slat_dec_gs.xml / .bin                   # Gaussian Splatting decoder
├── slat_decoder_gs_weights/
├── slat_decoder_mesh_attn.xml / .bin        # Mesh decoder (attention)
├── slat_mesh_upsample.xml / .bin            # Sparse subdivide upsample
├── slat_decoder_mesh_upsample_manifest.json
├── slat_decoder_mesh_upsample_weights/
└── slat_flexicubes.xml / .bin               # FlexiCubes surface extraction
```

> A `model_cache/` directory is also created next to the models on first run.
> It holds OpenVINO's compiled-blob cache, which makes subsequent runs start
> much faster. It is safe to delete.

---

## Running Inference

Standalone inference requires only **OpenVINO + NumPy + Pillow** — no PyTorch
or CUDA is needed at inference time.

### Basic Usage
```bash
python run_inference_standalone.py \
    --model-dir ./exported_models \
    --image kid_box/image.png \
    --mask  kid_box/0.png \
    --output kid_box.ply
```

The output extension selects the format:

| `--output` | Produces |
|---|---|
| `out.ply` | Gaussian splat PLY, plus `out.glb` if the mesh models are present |
| `out.glb` | Textured mesh GLB only |

### RGBA Input (mask embedded in alpha channel)
```bash
python run_inference_standalone.py \
    --model-dir ./exported_models \
    --image /path/to/input_rgba.png \
    --output output.ply
```

> **Note:** Always supply `--mask` or use an RGBA image that has the object mask
> in the alpha channel. Without a mask the entire image is treated as the object,
> producing significantly fewer occupied voxels compared to the reference baseline.

### Full Options
```bash
python run_inference_standalone.py \
    --model-dir ./exported_models \
    --ext-dir   ./openvino_extensions \
    --image     kid_box/image.png \
    --mask      kid_box/0.png \
    --output    reconstruction.ply \
    --device    GPU.0 \
    --seed      42
```

**Argument reference:**

| Argument | Default | Description |
|---|---|---|
| `--model-dir` | *(required)* | Directory with exported OV IR models and `config.json` |
| `--ext-dir` | `<model-dir>/../openvino_extensions` | Directory containing extension `.so` libraries |
| `--image` | *(required)* | Input image path (RGBA PNG or RGB JPEG) |
| `--mask` | auto-detected | Mask image path; if omitted, `0.png` next to `--image` is used |
| `--output` | `output.ply` | Output path; extension selects PLY or GLB (see above) |
| `--device` | `GPU` | OpenVINO inference device (`GPU`, `GPU.0`, `CPU`, `AUTO`) |
| `--moge-dir` | `None` | Path to MoGe checkpoint directory (optional) |
| `--seed` | `42` | Random seed for reproducibility |
| `--quiet` | `False` | Suppress verbose progress output |
| `--slat-steps` | from `config.json` (25) | SLAT ODE step count. Leave at the default for baseline parity |

> On a multi-GPU host, `--device GPU` picks the first enumerated device, which
> may not be the one you want. Use an explicit index such as `--device GPU.0`.
> `python -c "import openvino as ov; print(ov.Core().available_devices)"` lists
> the options.

### Pipeline Summary

The inference script runs the following stages without PyTorch:

1. **Load image** → extract RGB + alpha mask
2. **MoGe** (`moge.xml`) → pointmap `(3, H, W)`
3. **Preprocess** → crop around mask, pad to square, resize to 518×518
4. **SS condition embedding** → DINOv2 × 4 + PointPatchEmbed × 2 + projection
   nets → `(1, 7528, 1024)` tokens
5. **SS sampling** → ShortCut ODE, 25 steps → `(1, 4096, 8)` shape latent (16³
   tokens) plus pose latents
6. **SS decode** (`ss_decoder.xml`) → `(1, 1, 64, 64, 64)` occupancy → active
   voxel coordinates
7. **SLAT condition embedding** → `(1, 5496, 1024)` tokens
8. **SLAT sampling** → FlowMatching ODE, 25 steps, using the SparseConv3d
   extension + dense attention blocks → `(N_vox, 8)` latents
9. **Decode** → Gaussian Splatting parameters and mesh features (in parallel)
10. **Mesh extraction** → sparse subdivide upsample + FlexiCubes (skipped when
    the mesh models are absent)
11. **Save** → `.ply` (Gaussian splat) and/or `.glb` (textured mesh)

> **Step counts are not tuning parameters.** 25 SS steps and 25 SLAT steps are
> what the reference PyTorch/CUDA baseline uses; lowering either will diverge
> from the baseline output.

---

## References

- [SAM 3D Objects (Meta)](https://github.com/facebookresearch/sam-3d-objects)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [Intel Compute Runtime](https://github.com/intel/compute-runtime)
- [OpenVINO Custom Operations Guide](https://docs.openvino.ai/2024/documentation/openvino-extensibility.html)