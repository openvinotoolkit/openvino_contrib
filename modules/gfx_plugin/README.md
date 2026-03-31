# OpenVINO GFX Plugin

Experimental OpenVINO device plugin registered as `"GFX"`.

This directory is intended to be self-contained. Start here, then read:
- `docs/ARCHITECTURE.md` for the current code-truth architecture
- `docs/DEVELOPMENT.md` for build, extension, and debugging guidance
- `docs/TESTING.md` for test targets and workflows
- `docs/USAGE.md` for integration examples

## What The Plugin Does
GFX is an OpenVINO runtime plugin that compiles an `ov::Model` into a backend-specific GPU execution pipeline. The current implementation supports:
- Metal on macOS
- Vulkan on non-Apple builds when Vulkan is available at configure time

The codebase uses a shared frontend and a stage-based runtime:
1. OpenVINO graph transformations run in `src/transforms/`
2. Backend-aware support probing is driven by MLIR builders in `src/mlir/`
3. `CompiledModel` builds a pipeline of `GpuStage` objects
4. Backend-specific infer requests bind tensors and execute the stage pipeline

Recent runtime work extends this model in two directions:
- compile-time stage planning now picks layout, fusion, and execution policy per stage
- backend runtimes, especially Vulkan, can choose specialized direct or chunked execution routes for selected ops
- infer execution can batch stage recording into submission windows and reuse prepared bindings or immutable device buffers across requests
- device-aware scheduling now uses backend-reported execution limits through shared `gfx_parallelism.*` and `gfx_partitioning.*` helpers

This is not the old monolithic `MlirBackend` architecture that earlier design notes experimented with.

## Current Status
- Device name: `GFX`
- Backends: `metal`, `vulkan`
- Supported OpenVINO version: `2026.0.0` Developer Package builds
- No partial CPU fallback: unsupported models fail during `compile_model()`
- `query_model()` is backend-aware and follows the same support probing path as compilation
- `import_model()` reloads an OpenVINO model and recompiles it
- `export_model()` serializes the OpenVINO model, not a backend pipeline cache
- `GfxRemoteContext` and `GfxRemoteTensor` are implemented, but capabilities still depend on the compiled backend

## Source Layout
- `include/openvino/gfx_plugin/`: public plugin headers
- `src/plugin/`: `Plugin`, `CompiledModel`, shared property handling, pipeline construction
- `src/runtime/`: backend-neutral runtime abstractions and helpers
- `src/mlir/`: MLIR support probes, builders, and shared codegen helpers
- `src/backends/metal/`: Metal-specific plugin glue, runtime, memory, profiling, MSL compilation
- `src/backends/vulkan/`: Vulkan-specific plugin glue, runtime, buffers, profiling, SPIR-V/Vulkan execution
- `src/transforms/`: OpenVINO graph passes and fusion logic
- `src/kernel_ir/`: shared kernel metadata and planning structures
- `tests/`: unit, integration, backend, and tooling coverage
- `tools/`: developer scripts for profiling workflows, microbench smoke checks, and report post-processing
- `docs/`: local module docs, including profiling and microbench references
- `third_party/llvm-project/`: vendored LLVM/MLIR used by the build
- `third_party/Vulkan-Headers/`: vendored Vulkan headers used by the Raspberry Pi Vulkan toolchain flow

## Main Runtime Components
### Plugin
`src/plugin/plugin.cpp` owns:
- property parsing and validation
- backend resolution through `GFX_BACKEND`
- `query_model()`
- remote context creation
- model transformation before compilation

### CompiledModel
`src/plugin/compiled_model.cpp` owns:
- compiled-model properties and profiling configuration
- backend state creation
- stage pipeline construction in `build_op_pipeline()`
- optional fusion of compatible stages through `FusedSequenceStage`
- absorption of selected transpose inputs into downstream stages through `GfxInputTransform`

### InferRequest
`include/openvino/gfx_plugin/infer_request.hpp` plus backend-specific implementation files own:
- host and remote tensor binding
- per-request backend state
- command submission
- profiling collection

The current infer path is not a naive "execute one stage, submit immediately" loop. `src/plugin/infer_submission.*` and `src/plugin/infer_pipeline.*` now provide:
- reusable bound pipelines and prepared input-resolution plans
- prepared output-resolution plans for stage outputs, passthrough parameters, and materialized constant outputs
- reusable host output tensors for static output signatures when the user does not bind explicit output storage
- submission windows driven by stage submit policy, stage count, and output-byte thresholds
- backend-specific submission sessions for Metal and Vulkan

### Backend-neutral runtime
`src/runtime/` contains:
- `GpuStage`: execution-stage interface
- `GpuStageFactory` / `ExecutionDispatcher`: backend-specific stage dispatch
- `gfx_stage_policy.*`: runtime route, fusion, and submit-policy selection
- `gfx_parallelism.*` and `gfx_partitioning.*`: backend-neutral device-capability and workgroup planning helpers
- `immutable_gpu_buffer_cache.*`: backend-neutral cache for immutable device buffers
- shared remote context/tensor abstractions
- common tensor, buffer, logging, kernel-binding, and parallelism helpers

`GpuStage` now exposes two hooks that affect real runtime behavior:
- `set_input_transform()`: lets a stage consume an absorbed transpose as metadata instead of materializing a separate runtime stage
- `submit_policy()`: lets a stage communicate scheduling weight or isolation requirements to the infer pipeline

The runtime also has explicit reuse layers:
- immutable constant payloads can be cached as device buffers through backend const-cache implementations
- compiled kernels can reuse prepared binding tables through shared backend-neutral cache helpers in `gpu_backend_base.hpp`
- infer requests can reuse prepared output bindings and preallocated host output tensors across repeated executions

Profiling now also has two layers:
- compile-time tracing stored as a JSON `compile` section inside `GFX_PROFILING_REPORT`
- infer-time node, segment, transfer, allocation, and counter reporting through `gfx_profiling_report.*`

Backend-neutral planning now consumes device info exported by the active buffer manager:
- Metal and Vulkan buffer managers report subgroup width and workgroup limits through `GpuExecutionDeviceInfo`
- `gfx_parallelism.*` converts that into execution-policy caps
- `gfx_partitioning.*` derives 1D and 2D workgroup shapes from the same data

## Backend Selection
The plugin has two layers of backend choice:
- Build-time availability via CMake:
  - `GFX_ENABLE_METAL`
  - `GFX_ENABLE_VULKAN`
  - `GFX_DEFAULT_BACKEND`
- Runtime selection via property:
  - `GFX_BACKEND=metal`
  - `GFX_BACKEND=vulkan`

On macOS, CMake disables Vulkan and Metal becomes the only runtime backend.

## Supported Ops
Support is driven by MLIR builders in `src/mlir/` and backend runtime implementations. The active set includes:
- MatMul, Conv2D, Conv3D, GroupConv
- Add, Sub, Mul, Div, Pow, Mod, FloorMod
- compare, logical, and select operations
- unary activations and elementwise transforms
- MaxPool, AvgPool, Softmax, BatchNormInference
- Concat, Split, Slice, Transpose, Reshape, Convert, Interpolate
- Gather, GatherND, GatherElements, Scatter updates, ShapeOf, Range, Tile, TopK, SpaceToDepth, DepthToSpace

Important constraints:
- many paths require static rank or static shape
- many ops require constant attributes
- backend parity is not guaranteed between Metal and Vulkan

Current lowering/runtime special cases:
- selected transpose inputs can be absorbed into Add, Conv2D, GroupConv2D, and Split lowering instead of staying as standalone runtime stages
- Vulkan contains specialized direct or chunked paths for unary, binary, softmax, split/concat, transpose, convert, Conv2D, and GroupConv2D cases
- some Conv2D shapes may be lowered through an explicit MLIR `im2col + matmul` route when the selected execution policy prefers it
- Softmax lowering now supports arbitrary normalized axes instead of only the last axis
- Slice lowering now prefers `tensor.extract_slice`; generic slice metadata extraction still accepts the older generic form when needed

## Public And Internal Properties
Commonly used properties:
- `GFX_BACKEND`
- `GFX_ENABLE_FUSION`
- `GFX_PROFILING_LEVEL`
- `GFX_PROFILING_REPORT`
- `GFX_MEM_STATS`
- `ov::device::id`
- `ov::cache_dir`
- `ov::enable_profiling`
- `ov::loaded_from_cache`
- legacy `PERF_COUNT`

See `src/plugin/gfx_property_lists.cpp` for the exact supported property sets exposed by the plugin and by compiled models.

Practical meanings:
- `GFX_BACKEND`: request `metal` or `vulkan`
- `GFX_ENABLE_FUSION`: enable stage fusion during pipeline construction
- `GFX_PROFILING_LEVEL`: control profiling detail level
- `GFX_PROFILING_REPORT`: fetch the latest profiling report, including compile and infer sections when profiling is enabled
- `GFX_MEM_STATS`: fetch backend memory statistics from a compiled model
- `ov::device::id`: select a device index when the active backend supports it
- `ov::cache_dir`: reuse the standard OpenVINO cache directory for Vulkan pipeline-cache persistence
- `ov::loaded_from_cache`: report whether the OpenVINO model-cache path loaded this compiled model

## Build
Assuming an OpenVINO Developer Package is available:

```bash
cd /path/to/gfx_plugin
cmake -S . -B build-gfx-plugin -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINODeveloperPackage_DIR=/path/to/openvino/install/cmake \
  -DENABLE_TESTS=ON \
  -DGFX_DEFAULT_BACKEND=auto
cmake --build build-gfx-plugin --target openvino_gfx_plugin ov_gfx_func_tests
cmake --build build-gfx-plugin --target ov_gfx_unit_tests ov_gfx_runtime_micro_tests ov_gfx_microbench
```

Useful options:
- `-DGFX_ENABLE_METAL=ON|OFF`
- `-DGFX_ENABLE_VULKAN=ON|OFF`
- `-DGFX_DEFAULT_BACKEND=auto|metal|vulkan`

Build notes:
- vendored LLVM/MLIR is now built as a static external toolchain under `third_party/llvm-project`
- the external LLVM bootstrap now injects a tiny local dummy fuzzing-engine archive so `mlir-parser-fuzzer` configure paths do not break the bundled llvmorg-22.1.2 flow
- Android and generic cross-compiling flows forward toolchain settings into that external LLVM/MLIR build
- the module build treats compiler warnings as errors by default through `-Werror` on Clang/GCC and `/WX` on MSVC
- `cmake/GfxAndroidRuntimeBundle.cmake.in` provides helper copy logic for Android-side runtime dependency bundling
- `tools/gfx_rpi_vulkan_toolchain_builder.py` can assemble a hermetic Raspberry Pi Vulkan cross-toolchain bundle for `aarch64` Bookworm-style targets, normalize absolute sysroot symlinks, and install both `vulkan/` and `vk_video/` headers into the generated sysroot

The build produces the `openvino_gfx_plugin` shared library. On Unix-like builds this is typically emitted as `libopenvino_gfx_plugin.so`; the `.so` suffix is also forced on macOS for OpenVINO plugin loading compatibility.

## Quick Start
For a local development build, register the plugin explicitly through `ov::Core`.

### Register the plugin
```cpp
#include <openvino/openvino.hpp>

ov::Core core;
core.register_plugin("/path/to/libopenvino_gfx_plugin.so", "GFX");
```

### Inspect the active backend
```cpp
std::string backend = core.get_property("GFX", "GFX_BACKEND").as<std::string>();
std::string full_name = core.get_property("GFX", ov::device::full_name).as<std::string>();
```

### Compile a model for GFX
```cpp
auto model = core.read_model("/path/to/model.xml");

ov::AnyMap config = {
    {"GFX_BACKEND", "metal"},
    {"GFX_ENABLE_FUSION", true},
    {ov::enable_profiling.name(), true},
};

ov::CompiledModel compiled = core.compile_model(model, "GFX", config);
```

### Run inference
```cpp
ov::InferRequest request = compiled.create_infer_request();
request.set_input_tensor(input_tensor);
request.infer();
ov::Tensor output = request.get_output_tensor();
```

For longer examples and property-oriented workflows, see `docs/USAGE.md`.

## Running Tests
Run the labeled suite:

```bash
ctest --test-dir build-gfx-plugin --output-on-failure -L GFX
```

Locate and run the gtest binaries directly:

```bash
find build-gfx-plugin -name ov_gfx_func_tests -type f
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_func_tests> --gtest_filter=MetalBasicOps.*

find build-gfx-plugin -name ov_gfx_unit_tests -type f
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_unit_tests> --gtest_filter=GfxMlirTransforms.*
```

`ov_gfx_func_tests` covers plugin-facing and functional behavior, `ov_gfx_unit_tests` carries focused runtime, MLIR, backend, and cache regressions, and `ov_gfx_runtime_micro_tests` is used for smaller runtime subgraph checks. Helper tools `ov_gfx_compare_runner` and `ov_gfx_microbench` are also built from `tests/tools/`.

Recent regression coverage includes:
- canonical Conv2D MLIR lowering checks
- im2col rewrite coverage, including the batch-1 plain-matmul route
- linear matmul parallel-lowering coverage
- absorbed input-transform tests for Add, Conv2D, GroupConv2D, and Split
- Vulkan runtime regression coverage in `tests/backends/vulkan/`
- infer submission, prepared-pipeline reuse, immutable-const-cache reuse, and shared kernel-binding reuse tests
- reusable output-resolution and reusable host-output coverage in `tests/unit/infer_pipeline_reuse_test.cpp`
- internal transform and plugin coverage in `tests/unit/basic_ops_internal_test.cpp`
- backend memory/device integration coverage in `tests/unit/memory_device_integration_test.mm`

## Debugging And Instrumentation
Useful environment variables from the current codebase:
- `OV_GFX_TRACE`: trace logging
- `OV_GFX_DEBUG`: debug logging
- `OV_GFX_TEST_DEBUG`: extra test-side logging
- `OV_GFX_DEBUG_MSL`: dump generated Metal shader sources
- `OV_GFX_SAFE_DEBUG`: enable additional Metal memory safety checks
- `OV_GFX_DUMP_SPIRV_BINDINGS`: dump Vulkan binding information
- `OV_GFX_DUMP_SPIRV_MLIR`, `OV_GFX_DUMP_SPIRV_MLIR_FILTER`, `OV_GFX_DUMP_MLIR_PRE_SPIRV`: Vulkan/MLIR dump controls

For output-quality checks against a reference backend, use `ov_gfx_compare_runner`. It is an accuracy-only helper: it registers local plugin builds, compares tensor diffs, can run per-op windows or full-graph per-op output scans, and can also emit `GFX`-only output summaries for quick debugging. For performance numbers, use `benchmark_app` instead of the compare tool.

For profiling-driven triage, use `ov_gfx_microbench` plus the local references in `docs/MICROBENCH_SCHEMA.md` and `docs/PROFILING_RUNBOOK.md`.

## Integration Notes
- `query_model()` follows the same backend-specific support checks as compilation, so external schedulers see actual backend capability rather than an optimistic superset.
- Unsupported models fail during compilation instead of silently mixing CPU execution into a request.
- `import_model()` re-creates an OpenVINO model and recompiles it for the resolved backend.
- `export_model()` writes model data, not a serialized GPU pipeline cache.
- Remote contexts and remote tensors are available, but practical capabilities still depend on the active backend implementation.
- Runtime submissions and constant-buffer reuse are backend-aware internals; they improve execution behavior but do not change the plugin's external OpenVINO contract.
- Output binding now has a reusable internal planning layer for stage outputs, passthrough outputs, and constant outputs, but the public infer-request API remains standard OpenVINO.

## Developer Documentation
All developer-facing documentation intended for publication from this directory lives under `docs/`:
- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT.md`
- `docs/TESTING.md`
- `docs/USAGE.md`

If you add new documentation meant for published consumers of this module, keep it in English and keep it inside `modules/gfx_plugin/`.

## Current Limitations
- The plugin is still experimental.
- Dynamic-shape support is partial.
- There is no partial CPU fallback path.
- Remote tensor support exists but is not yet a fully polished cross-platform feature set.
- Backend coverage and runtime maturity differ between Metal and Vulkan.
- Some optimized Vulkan paths depend on device capabilities such as subgroup size and compute workgroup limits.
