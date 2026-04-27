# Development Guide

This document is for contributors working inside `modules/gfx_plugin`.

## Prerequisites
- CMake 3.13 or newer
- an OpenVINO Developer Package build
- Ninja recommended
- Metal toolchain on macOS for the Metal backend
- Vulkan SDK or system Vulkan development files for the Vulkan backend

The module vendors LLVM/MLIR under `third_party/llvm-project` and can build the required MLIR pieces as part of the CMake flow.
`third_party/Vulkan-Headers` is tracked separately as a git submodule for the Raspberry Pi Vulkan flow.

## Configure And Build
Example configuration:

```bash
cd /path/to/gfx_plugin
cmake -S . -B build-gfx-plugin -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINODeveloperPackage_DIR=/path/to/openvino/install/cmake \
  -DENABLE_TESTS=ON \
  -DGFX_DEFAULT_BACKEND=auto
```

Build:

```bash
cmake --build build-gfx-plugin --target openvino_gfx_plugin ov_gfx_func_tests
cmake --build build-gfx-plugin --target ov_gfx_unit_tests ov_gfx_runtime_micro_tests ov_gfx_microbench
```

Useful CMake options:
- `GFX_ENABLE_METAL`
- `GFX_ENABLE_VULKAN`
- `GFX_DEFAULT_BACKEND`
- `ENABLE_TESTS`

On macOS, Vulkan is disabled by `cmake/GfxBackendConfig.cmake`.

Current build-system notes:
- vendored LLVM/MLIR is built as a static external toolchain under `third_party/llvm-project`
- the external LLVM bootstrap injects a tiny local dummy fuzzing-engine archive so the bundled llvmorg-22.1.2 MLIR configure path does not fail in `mlir-parser-fuzzer`
- the module now reuses installed package exports when present and otherwise falls back to build-tree exports during bootstrap
- Android and generic cross-compiling flows forward toolchain settings into the vendored LLVM/MLIR configure step
- that bootstrap path also forwards `CMAKE_C_FLAGS`, `CMAKE_CXX_FLAGS`, and the executable/shared/module linker flag families into the nested LLVM configure step
- the module build treats warnings as errors by default through `-Werror` on Clang/GCC and `/WX` on MSVC
- `cmake/GfxAndroidRuntimeBundle.cmake.in` is used to copy Android runtime-side dependencies next to deployed plugin artifacts

## Hermetic RPi Vulkan Toolchain
To build a Raspberry Pi Vulkan cross-toolchain bundle for this module, use:

```bash
python3 modules/gfx_plugin/tools/gfx_rpi_vulkan_toolchain_builder.py \
  --output-dir /path/to/build-gfx-plugin-rpi-toolchain
```

The script builds host LLVM tools from the vendored
`modules/gfx_plugin/third_party/llvm-project`, assembles a generic
Raspberry Pi 4/5 Bookworm arm64 sysroot, normalizes absolute sysroot symlinks,
copies both `vulkan/` and `vk_video/` headers from the vendored
`modules/gfx_plugin/third_party/Vulkan-Headers` tree, and generates:

```text
/path/to/build-gfx-plugin-rpi-toolchain/cmake/gfx-rpi-vulkan-aarch64.toolchain.cmake
```

Use that generated file directly with a normal CMake configure command:

```bash
cmake -S /path/to/openvino -B /path/to/build-gfx-plugin-rpi \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/build-gfx-plugin-rpi-toolchain/cmake/gfx-rpi-vulkan-aarch64.toolchain.cmake \
  -DOPENVINO_EXTRA_MODULES=/path/to/openvino_contrib/modules/gfx_plugin \
  -DGFX_ENABLE_METAL=OFF \
  -DGFX_ENABLE_VULKAN=ON \
  -DGFX_DEFAULT_BACKEND=vulkan
```

The builder is host-generic and is intended to work on Linux, macOS, and
Windows. If needed, you can override the generic sysroot source with:
- `--sysroot-dir /path/to/extracted-rootfs`
- `--sysroot-tarball /path/to/rootfs.tar.xz`

The generated compiler wrappers and toolchain file now use portable
`-march=armv8-a` flags instead of a Raspberry Pi 5-only `-mcpu=` default, so
the same bundle can be reused more safely across Raspberry Pi 4/5 class Vulkan
targets.

Before running the builder, prepare the Vulkan-Headers submodule dependency in:

```text
modules/gfx_plugin/third_party/Vulkan-Headers
```

Initialize that dependency with git submodules, for example:

```bash
git submodule update --init modules/gfx_plugin/third_party/Vulkan-Headers
```

The toolchain builder expects that submodule to be checked out at the upstream
`Vulkan-Headers` release currently pinned by the module for Raspberry Pi 4/5
Bookworm compatibility: `v1.3.239`.

## Where To Start Reading
Read in this order:
1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `src/plugin/plugin.cpp`
4. `src/plugin/compiled_model.cpp`
5. `include/openvino/gfx_plugin/infer_request.hpp`
6. the backend directory you are changing

For runtime execution, follow:
- `Plugin::compile_model()`
- `CompiledModel::build_op_pipeline()`
- backend-specific `infer_*_impl()`
- backend stage factory and executor code

If the behavior depends on route or scheduling selection, also read:
- `src/runtime/gfx_stage_policy.*`
- `src/runtime/gfx_parallelism.*`

`tests/unit/gfx_parallelism_test.cpp` now covers Broadcom-oriented matmul
selection plus dense stride-1, huge-spatial, and ultra-dense convolution
threadgroup decisions.
- `src/runtime/gfx_partitioning.*`
- `src/runtime/gpu_buffer_manager.hpp` for `GpuDeviceFamily` and backend-reported execution-device info
- the active backend executor, especially under `src/backends/vulkan/runtime/`
- `src/runtime/gfx_profiling_report.*` when the change affects counters, trace sinks, or JSON report shape

The current planning path is no longer just backend-wide. It includes family-specific tuning hooks, especially for:
- Broadcom V3D Vulkan devices
- Qualcomm Adreno Vulkan devices
- explicit convolution dispatch attrs forwarded into MLIR lowering

For current convolution work, there are now two important lowering details to keep in mind:
- full interior tiles in conv parallel lowering can skip lane-level bounds guards on the fast path
- the interior-window decision is now split into reusable height and width checks before the combined 2D helper decides whether the tile is fully interior
- Vulkan specialized kernel compilation may re-resolve effective argument count from final SPIR-V bindings instead of trusting only pre-SPIR-V metadata
- manual Vulkan Conv2D building can emit `gpu.func` batch-1 parallel entry points with explicit `gfx.dispatch_*` attrs and falls back to a serial path for larger batches

If the change touches infer-request throughput or resource reuse, also read:
- `src/plugin/infer_submission.*`
- `src/plugin/infer_pipeline.*`
- `src/plugin/stateful_execution.*`
- `src/plugin/stateful_stage.*`
- `src/runtime/immutable_gpu_buffer_cache.*`
- `src/runtime/gpu_backend_base.hpp`
- `src/runtime/gpu_buffer_manager.hpp`
- `src/plugin/infer_io_utils.*`
- `src/backends/vulkan/runtime/vulkan_buffer_manager.*` when Vulkan const-upload batching or shared upload-recording behavior changes

For stage-output reuse changes, also inspect:
- `StageOutputBufferWorkspace` in `src/plugin/infer_pipeline.hpp`
- `GpuStage::describe_output_lifetimes()` in `src/runtime/gpu_stage.hpp`
- `src/runtime/fused_sequence_stage.*` for fused-stage lifetime propagation

If the change touches stateful graphs, read `src/plugin/infer_request_state.hpp` as well. `ReadValue` and `Assign` are no longer just generic ops flowing through the normal stateless path: infer-request state now owns persistent variable buffers keyed by OpenVINO variable id.

## Adding Or Extending An Op
Typical path:
1. Add or extend MLIR support in `src/mlir/`
2. Ensure support probing succeeds through `mlir_supports_node()`
3. Implement or update backend runtime/codegen handling
4. Make sure the relevant backend stage can be created
5. Add tests in `tests/unit/` and the relevant backend test directory

For the current codebase, also check whether the op should participate in:
- absorbed input transforms through `GfxInputTransform`
- stage policy selection in `src/runtime/gfx_stage_policy.*`
- MLIR parallel lowering, cleanup passes, or public-signature rewrites in the pass pipeline
- backend-specialized fast paths such as Vulkan direct or chunked routes
- reusable bytes-arg materialization or immutable const-buffer caching
- stateful execution interception through `src/plugin/stateful_execution.*` when the graph uses `ReadValue` / `Assign`

For Reduce-like work, prefer the existing typed reduction extraction in `src/mlir/mlir_builder_reduce.cpp`. The current builder reads axes and `keep_dims` from concrete Reduce op classes such as `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`, `ReduceProd`, `ReduceL1`, and `ReduceL2` instead of relying on a looser generic reduction-base path.

For `MatMul`, keep the compile-time const-buffer story aligned with runtime codegen. The current Metal path may repack a constant RHS from `f32` to `f16` for dynamic-shape `MatMul` stages and then derive kernel input element types from the effective runtime tensors instead of only the original node input types.
If the change touches compressed or quantized `MatMul` weights, also inspect `src/transforms/pipeline.cpp`. The transform pipeline is now backend-aware and can protect decompression subgraphs on Metal so backend-specific compressed-weight handling survives generic optimization passes. It can also horizontally regroup compatible compressed `MatMul` nodes that share one data input into a fused `MatMul` plus `VariadicSplit`, and stage compilation may repack concatenated quantized weight/scales into backend const buffers.

For RMSNorm-style work, remember that `src/transforms/pipeline.cpp` now runs OpenVINO `RMSFusion` before plugin-local cleanup. The current intended path is fused RMSNorm graph patterns lowering into the dedicated `RMS` builder and backend codegen, not preserving only the unfused arithmetic tail.

For `ScatterUpdate`, use the dedicated builder in `src/mlir/mlir_builder_scatter_update.cpp`. The current path expects a constant scalar `axis`, static ranks, and normalized negative indices in the generated kernel path.

For `RoPE`, use the dedicated builder in `src/mlir/mlir_builder_rope.cpp` and the Metal codegen path in `src/mlir/rope_codegen.cpp`. The current native Metal route expects rank-3 or rank-4 data, rank-2/3/4 cos/sin inputs, no `input_trans0213` / `output_trans0213`, no ChatGLM or Qwen-special layouts, and no sliced input layout. Supported LLaMA rotate-half arithmetic may be rewritten into native `RoPE` in `src/transforms/pipeline.cpp` before stage compilation.

For fusion work, note that current support is no longer limited to output post-ops. `Multiply` can now absorb selected activations into one chosen input through the fusion plan and backend `fuse_input_activation()` hooks. Keep the transform-side fusion pattern, compiled-model fusion bookkeeping, and backend runtime/codegen support aligned.

For `ScaledDotProductAttention`, the current native path is backend-specific: Metal can keep a rank-4 FP16/FP32 SDPA node and compile a dedicated kernel, while Vulkan still rejects native SDPA and expects other lowering paths.

For Slice-like work, note that the current lowering prefers `tensor.extract_slice` instead of synthesizing a `linalg.generic` copy. Shared metadata extraction in `src/mlir/slice_generic_codegen.cpp` still accepts both forms so older paths and debug flows remain readable.
At runtime, contiguous `Split` and `VariadicSplit` outputs may also alias slices of the input buffer instead of materializing new output allocations. Keep split-plan inference in `src/mlir/mlir_stage.cpp` aligned with any change that affects byte layout or view eligibility.

For layout-cleanup work around DFL-style postprocessing tails, the current rewrite target is a value-preserving `Softmax -> MatMul -> Reshape/Transpose` form. Do not describe the older synthetic 1x1 convolution rewrite as the active implementation.

If the op needs fusion support, inspect:
- `src/transforms/`
- `src/runtime/fused_sequence_stage.*`
- fusion planning inside `src/plugin/compiled_model.cpp`

## Properties
Supported property lists are defined in:
- `src/plugin/gfx_property_lists.cpp`
- `include/openvino/gfx_plugin/properties.hpp`

If you add a property:
1. define the property key if needed
2. add it to the supported property list
3. parse and validate it in plugin or compiled-model code
4. cover it with tests

Current device-selection contract:
- `ov::available_devices` exposes numeric ids such as `"0"` instead of backend-reported human-readable names
- `ov::device::full_name` remains the right property for user-facing device naming
- `ov::device::id` should be validated against those numeric ids

## Debugging
Useful environment variables:
- `OV_GFX_TRACE`
- `OV_GFX_DEBUG`
- `OV_GFX_TEST_DEBUG`
- `OV_GFX_DEBUG_MSL`
- `OV_GFX_SAFE_DEBUG`
- `OV_GFX_DUMP_SPIRV_BINDINGS`
- `OV_GFX_DUMP_SPIRV_MLIR`
- `OV_GFX_DUMP_SPIRV_MLIR_FILTER`
- `OV_GFX_DUMP_MLIR_PRE_SPIRV`

These are implemented directly in the runtime and codegen sources; grep for `OV_GFX_` when adding new diagnostics.

For functional comparison against a reference backend, build and run `tests/tools/ov_gfx_compare_runner.cpp`. It is an accuracy-only tool for numeric diffs, per-op narrowing, full-graph per-op output scans, and `GFX`-only output summaries. Useful switches now include `--reference-device`, `--reference-plugin`, `--per-op`, `--per-op-all`, `--single-op-output`, `--tinyllama-prompt-inputs`, and `--gfx-only`. The current tool also understands boolean tensors and prints an extra mismatch probe for `Select` failures. Do not use it for performance numbers; use `benchmark_app` for that.

For profiling workflows, calibration artifacts, and cross-device trace correlation, use:
- `tests/tools/ov_gfx_microbench.cpp`
- `docs/MICROBENCH_SCHEMA.md`
- `docs/PROFILING_RUNBOOK.md`
- `tools/gfx_profile_runbook.py`
- `tools/gfx_microbench_smoke.py`
- `tools/gfx_calibration_diff.py`
- `tools/gfx_external_trace_summary.py`

For reuse and submission changes, prefer the focused unit tests under:
- `tests/unit/infer_submission_test.cpp`
- `tests/unit/infer_pipeline_reuse_test.cpp`
- `tests/unit/gpu_const_cache_test.cpp`
- `tests/unit/kernel_arg_reuse_test.cpp`
- `tests/unit/gpu_backend_base_test.cpp`
- `tests/unit/gfx_parallelism_test.cpp`
- `tests/unit/mlir_matmul_parallel_test.cpp`
- `tests/unit/basic_ops_internal_test.cpp`
- `tests/unit/gfx_profiling_report_test.cpp`
- `tests/unit/gfx_stage_policy_test.cpp`
- `tests/unit/runtime_subgraph_test.cpp`

`tests/unit/gfx_parallelism_test.cpp` now also covers Broadcom-oriented matmul and convolution tuning choices, while `tests/unit/gfx_stage_policy_test.cpp` continues to cover submit-window policy decisions against synthetic device-info snapshots.

`tests/unit/infer_pipeline_reuse_test.cpp` now also covers:
- prepared output-resolution plans
- reusable host output tensors for static outputs
- reuse of pre-resolved input vectors across repeated executes

## Documentation Rules For This Module
- Keep published module docs in English
- Keep module docs inside `modules/gfx_plugin/`
- Update `README.md` when user-visible behavior changes
- Update `docs/ARCHITECTURE.md` when the code-truth architecture changes
- Update `docs/TESTING.md` when test layout or commands change

## What Not To Do
- do not rely on silent CPU fallback
- do not document removed architectures as current behavior
- do not put primary module documentation only in repository-level files outside this directory
- do not leave backend-specific route changes undocumented when they affect supported shapes, layout assumptions, or profiling behavior
- do not add ad-hoc backend caches before checking whether `ImmutableGpuBufferCache` or the shared prepared-binding cache already solves the problem
- do not bypass reusable output planning when changing infer I/O paths; keep stage outputs, passthrough outputs, and constant outputs on the same documented resolution path
