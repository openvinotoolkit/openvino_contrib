# OpenVINO Metal Plugin (Experimental)

## 1. Overview
- Experimental Metal GPU backend for OpenVINO, built on top of Apple **MPSGraph**.
- Target platform: Apple Silicon / macOS with Metal enabled.
- Status: work‑in‑progress; already executes small OpenVINO models on the Apple GPU through MPSGraph.

## 2. Supported OpenVINO version
- The plugin currently targets **OpenVINO 2026.0.0** (Developer Package).
- It is intended to be built against an OpenVINO Developer Package produced from a build tree, e.g.
  `<openvino-build>/install/cmake` (replace with your local path).

## 3. Architecture (short)
- **Plugin (`src/plugin.cpp`)** implements `ov::IPlugin`, registers device name `"METAL"`, exposes device properties and `query_model`.
- **CompiledModel / InferRequest (`src/compiled_model.*`, `src/infer_request.*`)**
  - `CompiledModel` keeps the MPSGraph build result (graph handle + IO tensors).
  - `InferRequest` allocates public `ov::Tensor` inputs/outputs, wraps them via `ov::make_tensor(SoPtr)`, and calls the runtime executor.
- **MPSGraphBuilder (`src/mps_graph_builder.mm/.hpp`)**
  - Lowers `ov::Model` to an MPSGraph, creating placeholders/constants and mapping supported ops.
  - Execution path: `MTLCreateSystemDefaultDevice()` → `id<MTLCommandQueue>` → `runWithMTLCommandQueue:...` → outputs read back from `MPSGraphTensorData`/`MPSNDArray`.

## 4. Execution really runs on GPU
- In `mps_execute` a Metal device is obtained via `MTLCreateSystemDefaultDevice()` and a command queue via `[device newCommandQueue]`; MPSGraph execution is submitted to that queue with `runWithMTLCommandQueue:...`.
- MPSGraph has no CPU fallback backend; the ops are recorded into Metal command buffers and executed on the GPU.
- Outputs are retrieved from `MPSGraphTensorData` as `MPSNDArray`, which wraps GPU buffers; data is copied back to host only at the end.

**How to observe GPU load**
```bash
# Terminal 1: watch GPU power/load
sudo powermetrics --samplers gpu_power -i 500 -n 10

# Terminal 2: run the METAL tests
DYLD_LIBRARY_PATH=<path-to-openvino-runtime-libs> \
  <build-dir>/modules/metal_plugin/tests/ov_metal_basic_tests --gtest_filter=MetalBasicOps.*
```
You should see GPU power/load spikes while the tests are running.

## 5. Supported ops (current subset)
- Parameter (v0)
- Constant (v0)
- Result (v0)
- Add (v1) — no broadcasting yet
- Relu (v0)
- MatMul (v0) — only rank‑2 or rank‑3 (batched) inputs; transpose flags not supported
- Convolution (v1) — 2D, rank‑4 NCHW input and OIHW weights, `groups == 1`, basic strides/pads/dilations

### MLIR/Metal path (experimental, active)
- MatMul (rank‑2/3, f32, simple batch broadcast)
- Add (equal-shape + scalar/channel broadcast)
- Unary activations: Relu / Sigmoid / Tanh / Elu / PRelu (scalar) / Gelu
- Softmax (last axis, rank ≥ 2)
- MaxPool2D / AvgPool2D (NCHW, rank‑4, standard strides/pads, exclude_pad honored for AvgPool)

**Known limitations**
- No dynamic shapes.
- No grouped or depthwise convolutions yet.
- No general broadcasting or transposed matmuls.
- Only basic layouts (NCHW / OIHW internally).

## 6. Building the module
Assuming OpenVINO 2026 DevPackage is already built and installed to `<openvino-build>/install/cmake`:
```bash
# 1) (example) build OpenVINO DevPackage
cd /path/to/openvino
cmake -S . -B build-ninja -G Ninja -DENABLE_TESTS=OFF
cmake --build build-ninja --target install

# 2) configure metal_plugin
cd /path/to/openvino_contrib
mkdir -p cmake-build-metal && cd cmake-build-metal
cmake -S ../modules/metal_plugin -B . \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINODeveloperPackage_DIR=/path/to/openvino/build-ninja/install/cmake \
  -DENABLE_TESTS=ON

# 3) build plugin + tests
cmake --build . --target ov_metal_basic_tests -- -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
```
Adjust paths to match your local build directories.

## 7. Running tests
Available gtest suite: `ov_metal_basic_tests`, covering:
- MetalBasicOps.Add / Relu / MatMul2D / Conv2D
- MetalBasicOps.DevicePropertiesAndQuery
- MetalBasicOps.AutoMetalCpuHybrid (AUTO:METAL,CPU sanity)

Run all:
```bash
DYLD_LIBRARY_PATH=<path-to-openvino-runtime-libs> \
  ./modules/metal_plugin/tests/ov_metal_basic_tests --gtest_filter=MetalBasicOps.*
```
If the METAL plugin cannot be registered (e.g., missing Metal device), tests will gracefully `GTEST_SKIP`.

## 8. Current limitations / roadmap
- Apple Metal GPU only (Apple Silicon / macOS).
- Limited op coverage as listed above; no dynamic shapes.
- Performance tuning is minimal; `ov::hint::performance_mode` is stored but not yet exploited.
- TODO / roadmap:
  - Add pooling (MaxPool/AvgPool), Softmax and more elementwise ops.
  - Implement broadcasting for Add/MatMul.
  - Support grouped/depthwise convolutions.
  - Improve performance-mode handling and tighter AUTO/HETERO integration.
