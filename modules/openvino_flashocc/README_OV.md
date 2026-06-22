<!-- Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0 -->

# FlashOCC — OpenVINO Inference on Intel iGPU

Optimized end-to-end inference for **FlashOCC** (3D occupancy prediction) using
**OpenVINO 2026.3** on Intel Xe3 Pantherlake iGPU.

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.12 |
| CMake | ≥ 3.16 |
| Ninja | any (recommended) |
| OpenCL runtime | Intel NEO driver |
| GCC/Clang | C++17 |

```bash
sudo apt install python3.12 python3.12-venv python3.12-dev cmake ninja-build build-essential

python3 -m venv venv_ov2026_ws
source venv_ov2026_ws/bin/activate
pip install packaging
pip install --upgrade wheel setuptools

```

## OpenVINO IR Models

The runtime uses the following IR files:

```
split_f16out/
  image_encoder.xml / .bin        (~48 MB)  — ResNet50 image backbone, 6 cameras, F16
  bev_trunk.xml / .bin            (~29 MB)  — BEV neck + head, F16 output
  bev_trunk_argmax_only.xml / .bin (~58 MB) — Fused bev_trunk + ArgMax (i32 class output)
```

`setup.sh` now supports both workflows:

- If `--model-dir` already contains these files, setup uses them directly.
- If files are missing, setup can auto-download checkpoint + convert for you.

## First-Time Setup (from scratch)

If you do not yet have IR files, `setup.sh` can prepare them automatically.

Requirements for auto-prepare path:

- `python3.10` and `python3.10-venv` (used for a temporary conversion venv)
- internet access (to download checkpoint and Python packages)

Install Python 3.10 once:

```bash
sudo apt install -y python3.10 python3.10-venv
```

Then run setup with auto-prepare enabled:

```bash
bash setup.sh \
          --prepare-models \
          --model-variant m0 \
          --model-dir $(pwd)/split_f16out \
          --jobs $(nproc)
```

What `--prepare-models` does:

- creates/uses `venv_convert` (Python 3.10)
- installs CPU-only conversion dependencies
- downloads the checkpoint (`m0` or `m1`)
- runs `convert_to_openvino.py --export-mode split`
- writes required IR files into `--model-dir`

Notes:

- conversion env uses `numba>=0.56,<0.60` (compatible with Python 3.10)

## Quick Start

### 1. Clone the repository

```bash
git clone -b openvino-optimized https://github.com/<your-fork>/FlashOCC.git
cd FlashOCC
```

### 2. Run setup

`setup.sh` handles everything: clones + builds OpenVINO, creates the venv,
installs all dependencies, builds the bev_pool extension.

```bash
bash setup.sh \
    --model-dir  $(pwd)/split_f16out \
    --jobs $(nproc)
```

Options:
| Flag | Description |
|---|---|
| `--model-dir PATH` | Path to IR model directory (default: `./split_f16out`) |
| `--prepare-models` | Force model download+conversion before setup |
| `--model-variant m0\|m1` | Variant used for auto-prepare (default: `m0`) |
| `--jobs N` | Parallel build jobs (default: `nproc`) |
| `--skip-ov-build` | Reuse existing `ov_build/` clone (skip clone + build) |
| `--skip-bevpool-build` | Skip bev_pool extension build |
| `--run-test` | Run 80-frame E2E benchmark immediately after setup |
| `--num-samples N` | Frames for `--run-test` (default: 80) |

### 3. Run E2E inference

```bash
bash run_flashocc_ov_ws.sh [--num-samples 80] [--ov-device GPU]
```

This is the main entry point for the full benchmark workflow. It calls
`run_flashocc_ov.py` and runs the optimized OpenVINO deployment pipeline.

By default the pipeline uses **synthetic random inputs** via `RandomSampleProvider`.
For explicit provider control, run the Python entry directly:

```bash
python run_flashocc_ov.py \
     --model-dir /path/to/split_f16out \
     --sample-provider random \
     --num-samples 80 \
     --ov-device GPU
```

To plug in your own data source, use:

```bash
python run_flashocc_ov.py \
     --model-dir /path/to/split_f16out \
     --sample-provider custom \
     --num-samples 80 \
     --ov-device GPU
```

Then implement `CustomSampleProvider.__call__()` in `run_flashocc_ov.py`.
See `make_random_sample()` for the exact dict format every provider must return:

```python
{
    'images':      np.ndarray  # float32 (6, 256, 704, 3)  — 6 cameras, BGR, 0–255
    'post_rots':   np.ndarray  # float32 (6, 3, 3)          — image-space augmentation rotation
    'post_trans':  np.ndarray  # float32 (6, 3)             — image-space augmentation translation
    'sensor2egos': np.ndarray  # float32 (6, 4, 4)          — camera→ego SE3 transform
    'ego2globals': np.ndarray  # float32 (6, 4, 4)          — ego→global SE3 transform
    'intrins':     np.ndarray  # float32 (6, 3, 3)          — camera intrinsic matrix K
}
```

Current provider options:

- `--sample-provider random`: synthetic inputs for smoke tests and FPS measurement
- `--sample-provider custom`: calls `CustomSampleProvider` template for real data integration

## Directory Layout (after setup)

```
FlashOCC/
├── run_flashocc_ov.py                    # E2E inference script (all OV optimizations)
├── run_flashocc_ov_ws.sh                 # Benchmark runner (reads setup.env)
├── setup.sh                              # Full setup: clone OV, build, create venv
├── setup.env                             # Auto-generated paths (written by setup.sh)
├── requirements.txt                      # Python deps (excluding OV — built from source)
├── openvino_extensions/
│   └── bev_pool/                         # Custom BEVPool OV C++ extension
│       ├── bev_pool_op.cpp / .hpp
│       ├── ov_extension.cpp
│       ├── CMakeLists.txt
│       ├── bev_pool_gpu_panterlake.xml   # GPU kernel config for Panter Lake iGPU
│       └── *.cl                          # OpenCL kernels
├── ov_build/
│   └── openvino/                         # Cloned + built deepaks2/openvino
│       ├── build/                        # CMake build dir
│       └── bin/intel64/Release/          # GPU plugin .so
├── venv_ov2026_ws/                       # Python 3.12 venv (OV 2026.3 + deps)
└── work_dirs/flashocc-r50-m0/openvino/
    └── split_f16out/                     # IR models (gitignored — provide externally)
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
- The `bev_pool_gpu_panterlake.xml` config is tuned for Intel Xe3 Pantherlake iGPU tensor shapes.
- `setup.env` is written by `setup.sh` and sourced by `run_flashocc_ov_ws.sh` — do not delete it.
- Model `.bin/.xml` files are gitignored. They must be obtained separately.
