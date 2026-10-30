<!-- Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0 -->

# OpenVINO FlashOCC Export and Inference Guide

This guide explains how to export FlashOCC to OpenVINO IR and run OpenVINO
inference/benchmarking from the `openvino_contrib/modules/openvino_flashocc`
module.

Optimized end-to-end inference for **FlashOCC** (3D occupancy prediction) uses
**OpenVINO 2026.3** and custom BEV pool extensions.

## Prerequisites

| Tool | Version |
|---|---|
| Conda | Miniconda/Mambaforge or Anaconda |
| CMake | ≥ 3.16 |
| Ninja | any (recommended) |
| OpenCL runtime | Intel NEO driver |
| GCC/Clang | C++17 |

> **Note:** `setup.sh` creates one Python 3.10 Conda environment under `.conda/flashocc_ws` for model conversion and inference.

```bash
sudo apt install cmake ninja-build build-essential git
```

## OpenVINO IR Models

The following models are required (not included in the repo — generate from PyTorch checkpoint):

```
split_f16out/
  image_encoder.xml / .bin        (~48 MB)  — ResNet50 image backbone, 6 cameras, F16
  bev_trunk.xml / .bin            (~29 MB)  — BEV neck + head, F16 output
  bev_trunk_argmax_only.xml / .bin (~58 MB) — Fused bev_trunk + ArgMax (i32 class output)
```

Generate these models with `convert_to_openvino.py`.

## Quick Start

### 1. Go to the module directory

```bash
cd <openvino_contrib>/modules/openvino_flashocc
```

### 2. Run setup

`setup.sh` creates a shared conversion/runtime Conda environment, installs
OpenVINO and the required dependencies, and builds the bev_pool extension.
When requested, it also converts a user-supplied checkpoint in that environment.

**First-time setup** (uses a user-supplied checkpoint to generate IR models):

```bash
bash setup.sh \
    --prepare-models \
    --model-variant m0 \
     --checkpoint /path/to/your/checkpoint.pth \
    --jobs $(nproc)
```

**With existing IR models** (skips conversion, fastest path):

```bash
bash setup.sh \
    --model-dir  /path/to/split_f16out \
    --jobs $(nproc)
```

Options:
| Flag | Description |
|---|---|
| `--model-dir PATH` | Path to pre-built `split_f16out/` IR directory. If omitted, defaults to `./split_f16out` |
| `--model-variant m0\|m1` | Architecture/configuration matching the checkpoint. Default: `m0` |
| `--checkpoint PATH` | Compatible user-supplied PyTorch checkpoint used for conversion |
| `--prepare-models` | Force IR generation even if IR files already exist |
| `--jobs N` | Parallel build jobs (default: `nproc`) |
| `--run-test` | Run E2E benchmark immediately after setup |
| `--num-samples N` | Samples for `--run-test` (default: 80) |

### 3. Run E2E inference

```bash
bash run_flashocc_ov_ws.sh [--num-samples 80] [--ov-device GPU]
```

This is the main entry point for the full benchmark workflow. It calls
`run_flashocc_ov.py` and runs the optimized OpenVINO deployment pipeline.
Use it for end-to-end inference, profiling, tensor dumps, and other deployment
debugging tasks.

### 4. Run the Python entry directly (optional)

For explicit provider selection or direct script control, run `run_flashocc_ov.py`:

```bash
python run_flashocc_ov.py \
     --model-dir /path/to/split_f16out \
     --sample-provider random \
     --num-samples 20 \
     --ov-device GPU
```

## Directory Layout (after setup)

```
openvino_flashocc/
├── run_flashocc_ov.py                    # E2E inference script (all OV optimizations)
├── run_flashocc_ov_ws.sh                 # Benchmark runner (reads setup.env)
├── setup.sh                              # Model conversion and runtime environment setup
├── setup.env                             # Auto-generated paths (written by setup.sh)
├── requirements.txt                      # Python deps for this module
├── openvino_extensions/
│   └── bev_pool/                         # Custom BEVPool OV C++ extension
│       ├── bev_pool_op.cpp / .hpp
│       ├── ov_extension.cpp
│       ├── CMakeLists.txt
│       ├── bev_pool_gpu.xml               # GPU kernel config (Arrow Lake + Panther Lake)
│       └── *.cl                          # OpenCL kernels
├── .conda/flashocc_ws/                   # Python 3.10 Conda env (OV 2026.3 + deps)
└── split_f16out/                         # Generated or user-provided IR models
```

## Architecture Overview

```
Camera frames (6×)
      │
      ▼
 image_encoder.xml        ResNet50 backbone  (F16, static B=6, ~17ms)
      │  async start_async()
      │                ──┬──  parallel: _compute_geom() ~1.2ms
      │  wait()          │
      ▼                  ▼
 bev_pool (C++ ext)   BEVPool scatter onto BEV grid  (~2.7ms)
      │
      ▼
 bev_trunk_argmax_only.xml   BEV neck + head + ArgMax fused  (~10ms)
      │
      ▼
 occ_label [B,200,200,16] i32   3D occupancy class prediction
```

## Notes

- The bev_pool extension (`.so`) must be compiled against the same OpenVINO build used for inference.
- The `bev_pool_gpu.xml` config is the active shared profile validated for both Arrow Lake and Panther Lake iGPU.
- `setup.env` is written by `setup.sh` and sourced by `run_flashocc_ov_ws.sh` — do not delete it.
- Model `.bin/.xml` files are gitignored. They must be obtained separately.

## Contrib Integration Notes

- This module is intended to run in-place under `openvino_contrib/modules/openvino_flashocc`.
- Top-level module overview is listed in `openvino_contrib/README.md`.
- Keep module docs and paths relative to this folder so they remain portable across environments.
