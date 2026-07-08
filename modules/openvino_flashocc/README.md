<!-- Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0 -->

# OpenVINO FlashOCC

OpenVINO-enabled FlashOCC workflow for 3D occupancy prediction.

## Upstream Attribution

This standalone implementation is based on concepts and architecture from the original FlashOCC project.
Thanks to the FlashOCC authors and contributors:

- Upstream project: https://github.com/Yzichen/FlashOCC

This module provides:

- Model export from PyTorch to OpenVINO IR via [convert_to_openvino.py](./convert_to_openvino.py)
- OpenVINO inference pipeline via [run_flashocc_ov.py](./run_flashocc_ov.py)
- Workspace setup and benchmark runner scripts ([setup.sh](./setup.sh), [run_flashocc_ov_ws.sh](./run_flashocc_ov_ws.sh))
- Custom OpenVINO extensions for BEV pool and related kernels under [openvino_extensions](./openvino_extensions)

## Quick Setup

Complete setup from scratch (model export → Conda env → OpenVINO pip install → bev_pool build → benchmark):

```bash
bash setup.sh --prepare-models --run-test --model-variant m0 --model-dir "$(pwd)/split_f16out" --jobs "$(nproc)"
```

This takes **~5–10 minutes** and produces:
- `.conda/flashocc_ws/` with pip-installed OpenVINO ≥2024.0
- `split_f16out/` with converted F16 IR models
- `openvino_extensions/bev_pool/build_ws/libopenvino_bevpool_extension.so` (C++ extension)
- `setup.env` with paths to reuse later

Then run inference anytime:

```bash
bash run_flashocc_ov_ws.sh --num-samples 80
```

For full details and alternative workflows, see [README_OV.md](./README_OV.md).

## Running inference

The pipeline uses a `SampleProvider` interface to decouple data loading from inference.
By default it runs with **synthetic random inputs** — no dataset required:

```bash
python run_flashocc_ov.py \
    --model-dir /path/to/split_f16out \
    --sample-provider random \
    --num-samples 20 \
    --ov-device GPU
```

### Connecting your own data

Use `--sample-provider custom` and implement `CustomSampleProvider.__call__()` in
`run_flashocc_ov.py`:

```bash
python run_flashocc_ov.py \
    --model-dir /path/to/split_f16out \
    --sample-provider custom \
    --num-samples 20 \
    --ov-device GPU
```

`CustomSampleProvider` should return the same dict contract as below:

```python
class MyCameraProvider:
    def __init__(self, frames, calibration):
        self._frames = frames
        self._calib = calibration

    def __len__(self) -> int:
        return len(self._frames)

    def __call__(self, index: int) -> dict:
        return {
            'images':      load_and_resize_6_cameras(self._frames[index]),  # float32 (6,256,704,3)
            'post_rots':   self._calib[index]['post_rots'],    # float32 (6,3,3)
            'post_trans':  self._calib[index]['post_trans'],   # float32 (6,3)
            'sensor2egos': self._calib[index]['sensor2egos'],  # float32 (6,4,4)
            'ego2globals': self._calib[index]['ego2globals'],  # float32 (6,4,4)
            'intrins':     self._calib[index]['intrins'],      # float32 (6,3,3)
        }
```

See `make_random_sample()` in `run_flashocc_ov.py` for the full field descriptions and
exact array shapes and dtypes every provider must return.

## CLI arguments

| Argument | Description |
|---|---|
| `--model-dir PATH` | IR model directory (`*.image_encoder.xml`, `*.bev_trunk.xml`) |
| `--ov-device DEV` | OpenVINO device for encoder/trunk (default: `GPU`) |
| `--ov-bevpool-device DEV` | Device for BEV pool extension (default: `--ov-device`) |
| `--ov-gpu-precision` | GPU precision hint: `auto`/`f16`/`f32` (default: `f32`) |
| `--ov-inference-precision` | Override INFERENCE_PRECISION_HINT for all models |
| `--ov-enc-inference-precision` | Precision hint for `image_encoder` only |
| `--ov-trk-inference-precision` | Precision hint for `bev_trunk` only |
| `--ov-bevpool-inference-precision` | Precision hint for BEV pool extension only |
| `--ov-extension-so PATH` | Path to `libopenvino_bevpool_extension.so` |
| `--ov-gpu-config-xml PATH` | GPU custom layers XML for the BEV pool extension |
| `--num-samples N` | Number of frames to run (default: `10`) |
| `--run-duration SECS` | Run for at least N seconds, recycling frames |
| `--warmup-frames N` | Exclude first N frames from latency reporting (default: `5`) |
| `--sample-provider random\|custom` | Choose synthetic test data (`random`) or your custom provider integration (`custom`) |

## Quick Links

- Full instructions: [README_OV.md](./README_OV.md)
- Python dependencies: [requirements.txt](./requirements.txt)

## Notes

- OpenVINO is installed via pip (`openvino>=2024.0`). The `bev_pool` extension is compiled against the pip-installed OpenVINO using its CMake config discovered at setup time.
- This module focuses on optimized OpenVINO deployment; model conversion is handled separately via `convert_to_openvino.py`.
- The setup workflow exports models to OpenVINO F16 IR format for GPU inference; see [README_OV.md](./README_OV.md) for troubleshooting and advanced flags.
