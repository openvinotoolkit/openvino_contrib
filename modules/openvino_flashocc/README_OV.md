<!-- Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0 -->

# FlashOCC — OpenVINO Inference on Intel iGPU

Optimized end-to-end inference for **FlashOCC** (3D occupancy prediction) using
**OpenVINO 2026+** on Intel Xe3 Pantherlake iGPU.

## Prerequisites

| Tool | Version |
|---|---|
| Conda | Miniconda/Mambaforge or Anaconda |
| CMake | ≥ 3.16 |
| Ninja | any (recommended) or Unix Makefiles |
| OpenCL runtime | Intel NEO driver |
| GCC/Clang | C++17 |
| Git | any |

Install required system packages and ensure `conda` is on `PATH`. `setup.sh` creates Python 3.10 Conda envs under `.conda/`, so no host Python version switching is required.

```bash
sudo apt install cmake build-essential git
# Optional: ninja for faster builds
sudo apt install ninja-build
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

- Conda on `PATH` (used for the conversion env)
- internet access (to download checkpoint and Python packages)

Then run setup with auto-prepare enabled:

```bash
bash setup.sh \
          --prepare-models \
          --model-variant m0 \
          --model-dir $(pwd)/split_f16out \
          --jobs $(nproc)
```

What `--prepare-models` does:

- creates/uses `.conda/convert` (Conda Python 3.10)
- installs CPU-only conversion dependencies
- downloads the checkpoint (`m0` or `m1`)
- runs `convert_to_openvino.py --export-mode split`
- writes required IR files into `--model-dir`

## Quick Start

### Single Unified Command (from scratch):

```bash
bash setup.sh \
    --prepare-models \
    --run-test \
    --model-variant m0 \
    --model-dir "$(pwd)/split_f16out" \
    --jobs "$(nproc)"
```

This exports models, creates the runtime Conda env with pip-installed OpenVINO, builds bev_pool, and runs the benchmark in ~5–10 minutes.

### Two-Step Setup (if you prefer to separate conversion from runtime):

**Step 1: Export models**
```bash
bash setup.sh --prepare-models --model-variant m0 --model-dir "$(pwd)/split_f16out" --jobs "$(nproc)"
```

**Step 2: Create runtime and run benchmark**
```bash
bash setup.sh --run-test --model-dir "$(pwd)/split_f16out" --jobs "$(nproc)"
```

### Then anytime: Run inference

```bash
bash run_flashocc_ov_ws.sh --num-samples 80
```

## BEVPool Correctness Smoke Test

The `--run-test` setup option runs the end-to-end random-input benchmark. For
numerical correctness, the dedicated smoke test builds the BEVPool extension,
runs `BEVPoolBinSort` followed by `BEVPoolV2` on deterministic inputs, and
compares the complete BEV output with an independent NumPy reference.

From this module directory, using the runtime environment created by
`setup.sh`:

```bash
.conda/flashocc_ws/bin/python -m pip install -r tests/requirements.txt
.conda/flashocc_ws/bin/python -m pytest -v tests/test_bevpool_smoke.py
```

The test always runs on CPU and also runs on GPU when one is available. To
select one device explicitly:

```bash
FLASHOCC_BEVPOOL_TEST_DEVICE=CPU \
     .conda/flashocc_ws/bin/python -m pytest -v tests/test_bevpool_smoke.py

FLASHOCC_BEVPOOL_TEST_DEVICE=GPU \
     .conda/flashocc_ws/bin/python -m pytest -v tests/test_bevpool_smoke.py
```

The extension is built automatically in a temporary directory. A successful
run reports `1 passed`.

## Setup and Inference

### Run setup

`setup.sh` handles model preparation (if needed), creates the Python Conda env with pip-installed OpenVINO,
installs dependencies, and builds the bev_pool C++ extension.

**Option A: Complete setup in one command**

```bash
bash setup.sh \
    --prepare-models \
    --run-test \
    --model-variant m0 \
    --model-dir "$(pwd)/split_f16out" \
    --jobs "$(nproc)"
```

**Option B: Two-step (export models first, then runtime)**

```bash
# Export models
bash setup.sh --prepare-models --model-variant m0 --model-dir "$(pwd)/split_f16out"

# Create runtime environment
bash setup.sh --run-test --model-dir "$(pwd)/split_f16out" --jobs "$(nproc)"
```

**Option C: If you already have IR files**

```bash
bash setup.sh --run-test --model-dir "$(pwd)/split_f16out" --jobs "$(nproc)"
```

Options:

| Flag | Description |
|---|---|
| `--model-dir PATH` | Path to IR model directory (default: `./split_f16out`) |
| `--prepare-models` | Force model download+conversion (creates `.conda/convert`) |
| `--model-variant m0\|m1` | Variant for auto-prepare: M0 (14M params) or M1 (larger) (default: `m0`) |
| `--jobs N` | Parallel BEVPool extension build jobs (default: `nproc`) |
| `--run-test` | Build Conda env + extension + run 80-frame benchmark |
| `--num-samples N` | Frames for `--run-test` (default: 80) |

### Run E2E inference

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
├── setup.sh                              # Full setup: model export → Conda env + OV install → bev_pool build
├── setup.env                             # Auto-generated paths (written by setup.sh)
├── requirements.txt                      # Python deps for inference runtime
├── convert_to_openvino.py                # Model export script (ONNX → OpenVINO IR)
├── openvino_extensions/
│   └── bev_pool/                         # Custom BEVPool OV C++ extension
│       ├── bev_pool_op.cpp / .hpp
│       ├── ov_extension.cpp
│       ├── CMakeLists.txt
│       ├── bev_pool_gpu.xml               # GPU kernel config (Arrow Lake + Panther Lake)
│       ├── build_ws/                     # CMake build artifacts (git-ignored)
│       └── *.cl                          # OpenCL kernels
├── .conda/
│   ├── flashocc_ws/                      # Python 3.10 Conda env for inference (pip OpenVINO + deps)
│   └── convert/                          # Python 3.10 Conda env for model conversion (created by --prepare-models)
├── split_f16out/                         # IR models (gitignored — generated by --prepare-models)
│   ├── image_encoder.xml / .bin
│   ├── bev_trunk.xml / .bin
│   └── bev_trunk_argmax_only.xml / .bin
├── checkpoints/                          # Model weights (downloaded by --prepare-models)
│   └── flashocc-r50.pth or flashocc-r50-m1.pth
└── ov_build/                             # Build logs (git-ignored)
     └── logs/
```

**Key paths in `setup.env`** (sourced by `run_flashocc_ov_ws.sh`):
```bash
FLASHOCC_CONDA_ENV="/path/to/.conda/flashocc_ws"
FLASHOCC_MODEL_DIR="/path/to/split_f16out"
FLASHOCC_BEV_SO="/path/to/openvino_extensions/bev_pool/build_ws/libopenvino_bevpool_extension.so"
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
