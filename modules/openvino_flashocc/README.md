# OpenVINO FlashOCC

OpenVINO-enabled FlashOCC workflow for 3D occupancy prediction.

This module provides:

- Model export from PyTorch to OpenVINO IR via [convert_to_openvino.py](./convert_to_openvino.py)
- OpenVINO inference and profiling pipeline via [run_compare_flashocc_pt_ov.py](./run_compare_flashocc_pt_ov.py)
- Workspace setup and benchmark runner scripts ([setup.sh](./setup.sh), [run_flashocc_ov_ws.sh](./run_flashocc_ov_ws.sh))
- A lightweight OpenVINO-only latency runner via [run_flashocc_ov_latency.py](./run_flashocc_ov_latency.py)
- Custom OpenVINO extensions for BEV pool and related kernels under [openvino_extensions](./openvino_extensions)

## Quick Links

- Full instructions: [EXPORT_AND_INFERENCE_GUIDE.md](./EXPORT_AND_INFERENCE_GUIDE.md)
- Python dependencies: [requirements.txt](./requirements.txt)

## Notes

- This module keeps both conversion and comparison workflows, so users can generate their own OpenVINO models and validate PyTorch vs OpenVINO behavior.
- The inference guide includes setup, benchmarking, and deployment usage details.
