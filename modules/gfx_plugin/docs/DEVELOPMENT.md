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
Raspberry Pi 5 Bookworm arm64 sysroot, normalizes absolute sysroot symlinks,
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

Before running the builder, prepare the Vulkan-Headers submodule dependency in:

```text
modules/gfx_plugin/third_party/Vulkan-Headers
```

Initialize that dependency with git submodules, for example:

```bash
git submodule update --init modules/gfx_plugin/third_party/Vulkan-Headers
```

The toolchain builder expects that submodule to be checked out at the upstream
`Vulkan-Headers` release currently pinned by the module for Raspberry Pi 5
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
- `src/runtime/gfx_partitioning.*`
- `src/runtime/gpu_buffer_manager.hpp` for `GpuDeviceFamily` and backend-reported execution-device info
- the active backend executor, especially under `src/backends/vulkan/runtime/`
- `src/runtime/gfx_profiling_report.*` when the change affects counters, trace sinks, or JSON report shape

The current planning path is no longer just backend-wide. It includes family-specific tuning hooks, especially for:
- Broadcom V3D Vulkan devices
- Qualcomm Adreno Vulkan devices
- explicit convolution dispatch attrs forwarded into MLIR lowering

If the change touches infer-request throughput or resource reuse, also read:
- `src/plugin/infer_submission.*`
- `src/plugin/infer_pipeline.*`
- `src/runtime/immutable_gpu_buffer_cache.*`
- `src/runtime/gpu_backend_base.hpp`
- `src/runtime/gpu_buffer_manager.hpp`
- `src/plugin/infer_io_utils.*`
- `src/backends/vulkan/runtime/vulkan_buffer_manager.*` when Vulkan const-upload batching or shared upload-recording behavior changes

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

For Slice-like work, note that the current lowering prefers `tensor.extract_slice` instead of synthesizing a `linalg.generic` copy. Shared metadata extraction in `src/mlir/slice_generic_codegen.cpp` still accepts both forms so older paths and debug flows remain readable.

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

For functional comparison against a reference backend, build and run `tests/tools/ov_gfx_compare_runner.cpp`. It is an accuracy-only tool for numeric diffs, per-op narrowing, full-graph per-op output scans, and `GFX`-only output summaries. Useful switches now include `--reference-device`, `--reference-plugin`, `--per-op`, `--per-op-all`, and `--gfx-only`. Do not use it for performance numbers; use `benchmark_app` for that.

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
