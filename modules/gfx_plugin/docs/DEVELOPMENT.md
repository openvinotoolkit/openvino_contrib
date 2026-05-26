# Development Guide

This guide is for contributors working inside `modules/gfx_plugin`.

## Prerequisites

- CMake 3.13 or newer
- Ninja recommended
- OpenVINO Developer Package
- Metal SDK/frameworks on macOS for the Metal backend
- OpenCL runtime on non-Apple targets for the OpenCL source-kernel backend

The module vendors LLVM/MLIR in `third_party/llvm-project` and builds the
required MLIR components as part of the CMake flow. Do not modify vendored LLVM
unless a task explicitly requires it.

## Configure And Build

Configure from the module root:

```bash
cmake -S . -B build-gfx-plugin -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINODeveloperPackage_DIR=/path/to/openvino/install/cmake \
  -DENABLE_TESTS=ON \
  -DGFX_DEFAULT_BACKEND=auto
```

Build common targets:

```bash
cmake --build build-gfx-plugin --target openvino_gfx_plugin
cmake --build build-gfx-plugin --target ov_gfx_func_tests ov_gfx_unit_tests ov_gfx_runtime_micro_tests
cmake --build build-gfx-plugin --target ov_gfx_compare_runner ov_gfx_microbench
```

Useful CMake options:

- `GFX_ENABLE_METAL`
- `GFX_ENABLE_OPENCL`
- `GFX_DEFAULT_BACKEND=auto|metal|opencl`
- `ENABLE_TESTS`

On macOS, `cmake/GfxBackendConfig.cmake` disables OpenCL and resolves the Apple
route to Metal/MPS/MPSRT/MSL. On non-Apple builds, the OpenCL source backend is
the runtime route when it is enabled in the build.

Build-system notes:

- `cmake/GfxSources.cmake` is the source list used by both the module root and
  `src/CMakeLists.txt`.
- `gfx_plugin_core`, `gfx_runtime_common`, and `gfx_runtime_mlir` contain shared
  code.
- `gfx_plugin_metal` / `gfx_runtime_metal` and
  `gfx_plugin_opencl` / `gfx_runtime_opencl` contain backend-specific code.
- `src/plugin/gfx_backend_config.hpp.in` is configured into the build tree with
  backend availability booleans and the resolved default backend.
- The OpenCL backend dynamically loads the target OpenCL runtime; it does not
  require a compile-time OpenCL SDK link.
- Android and generic cross builds forward toolchain settings into the vendored
  LLVM/MLIR configure step.
- The build treats warnings as errors through `-Werror` on Clang/GCC and `/WX`
  on MSVC.

## Where To Start Reading

Read in this order:

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `src/plugin/plugin.cpp`
4. `src/plugin/compiled_model.cpp`
5. `src/plugin/infer_pipeline.*`
6. `src/plugin/infer_submission.*`
7. the backend directory you are changing

For runtime planning, also inspect:

- `src/runtime/gfx_stage_policy.*`
- `src/runtime/gfx_parallelism.*`
- `src/runtime/gfx_partitioning.*`
- `src/runtime/gfx_target_profile.*`
- `src/kernel_ir/gfx_kernel_manifest.hpp`
- `src/kernel_ir/gfx_custom_kernel_families.*`
- `src/kernel_ir/gfx_opencl_source_artifacts.*`
- `src/mlir/gfx_stage_runtime_values.*`
- `src/mlir/gfx_backend_custom_kernel_adapter.*`

For Metal placement, MPSRT, or MSL source planning, also inspect:

- `src/runtime/gfx_mpsrt_abi.hpp`
- `src/runtime/gfx_mpsrt_model.*`
- `src/runtime/gfx_mpsrt_plan.hpp`
- `src/runtime/gfx_mpsrt_program.hpp`
- `src/runtime/gfx_mpsrt_kernel_manifest_adapter.hpp`
- `src/mlir/gfx_apple_stage_pipeline.*`
- `src/mlir/gfx_apple_vendor_descriptors.*`
- `src/mlir/gfx_mpsrt_dialect.*`
- `src/mlir/gfx_mpsrt_ops.*`
- `src/mlir/gfx_mpsrt_source_plan.hpp`
- `src/mlir/msl_codegen_apple_msl*.{cpp,hpp}`
- `src/mlir/msl_codegen_apple_mps.*`
- `src/mlir/msl_codegen_matmul_*`
- `src/mlir/msl_codegen_attention.*`
- `src/mlir/msl_codegen_compressed_matmul.*`
- `src/backends/metal/runtime/mpsrt/`

For OpenCL source execution, start with:

- `src/backends/opencl/plugin/`
- `src/backends/opencl/runtime/opencl_api.*`
- `src/backends/opencl/runtime/opencl_buffer_manager.*`
- `src/backends/opencl/runtime/opencl_program_cache.*`
- `src/backends/opencl/runtime/opencl_source_stage.*`
- `tests/unit/gfx_opencl_source_artifacts_test.cpp`

## Adding Or Changing An Operation

1. Update support probing in `src/mlir/`.
2. Add or adjust lowering in the relevant `mlir_builder_*.cpp`,
   source-plan helper, or transform.
3. Decide the shared runtime contract before adding backend code:
   - stage policy for placement/fusion/submission
   - kernel manifest for custom-kernel ABI
   - runtime-value payloads for dynamic metadata
   - OpenCL source artifact for source-kernel execution
   - MPSRT/Apple MSL source plan for Metal execution
4. Add backend code only where the shared contract crosses into Metal or OpenCL.
5. Add focused unit tests first.
6. Add backend or functional tests when behavior is externally visible.
7. Update docs when public properties, supported shapes, route selection,
   profiling, or test workflow changes.

Common operation families that need extra care:

- dynamic-shape data movement: `ShapeOf`, `Concat`, `Broadcast`, `Select`,
  `Slice`, `Range`, and `Tile`
- stateful `ReadValue` / `Assign`
- view-style `Split` / `VariadicSplit` aliases
- Metal MPS/MPSGraph vendor stages and MPSRT storage bridges
- Metal custom MSL source plans with explicit kernel-buffer order
- OpenCL source artifacts with scalar ABI, constants, chunking, and boolean
  output padding
- LLM-oriented fusions such as `RoPE`, compressed `MatMul`, and SDPA variants

## Shared Versus Backend-Specific Code

Prefer these shared locations:

- graph rewrites: `src/transforms/`
- support probing and lowering: `src/mlir/`
- stage policy and parallelism: `src/runtime/gfx_stage_policy.*`,
  `gfx_parallelism.*`, and `gfx_partitioning.*`
- binding and manifest contracts: `src/kernel_ir/` and
  `src/mlir/gfx_backend_custom_kernel_adapter.*`
- infer planning and submission: `src/plugin/infer_pipeline.*` and
  `src/plugin/infer_submission.*`

Use backend directories only for real backend boundaries:

- Metal Objective-C++ APIs, MTL resources, MPS/MPSGraph setup, MSL compilation,
  and command encoding belong under `src/backends/metal/`.
- OpenCL platform/device discovery, program compilation, buffer management, and
  kernel enqueue code belong under `src/backends/opencl/`.

Do not duplicate shared ABI or shape rules in backend request code.

## OpenCL Source Artifacts

`src/kernel_ir/gfx_opencl_source_artifacts.*` is the source of truth for:

- source id and entry point
- tensor role order
- scalar ABI
- source-static scalar values
- local size
- element-count source
- dynamic-shape scalar metadata
- direct constant materialization
- boolean buffer padding
- generated Concat/Split chunk artifacts

`src/backends/opencl/runtime/opencl_source_stage.*` should stay a generic
artifact executor. Add metadata to artifacts rather than adding op-specific
branches to the executor.

## Metal MPSRT And MSL

Metal placement must stay coordinated across:

- `gfx_stage_policy.*`
- `GfxKernelStageManifest`
- custom-kernel family metadata
- Apple source-plan helpers
- typed `GfxMpsrtProgram` / generated `gfx_mpsrt_ops`
- `GfxMpsrtBuilderPlan`
- `src/runtime/gfx_mpsrt_model.*`
- `src/backends/metal/runtime/mpsrt/`

When a manifest supplies explicit external-buffer roles, those roles define the
runtime ABI. MSL buffer scans and signature hints are diagnostics/fallback
inputs only; they must not widen or shrink a typed MPSRT binding contract.

## Properties

Public GFX properties are declared in
`include/openvino/gfx_plugin/properties.hpp` and exposed through
`src/plugin/gfx_property_lists.cpp`.

Before changing property behavior, check:

- `src/plugin/gfx_property_utils.*`
- `src/plugin/plugin.cpp`
- `src/plugin/compiled_model.cpp`
- `tests/unit/plugin_tests.cpp`
- `docs/USAGE.md`

## Testing Expectations

Use the narrowest relevant checks first:

```bash
cmake --build build-gfx-plugin --target ov_gfx_unit_tests
find build-gfx-plugin -name ov_gfx_unit_tests -type f
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_unit_tests> --gtest_filter=GfxStagePolicy.*
```

Before commit, run at least:

```bash
git diff --check
ctest --test-dir build-gfx-plugin --output-on-failure -L GFX
```

For documentation-only changes, `git diff --check` and targeted grep checks are
usually enough unless build files, public properties, or source lists changed.

## Public Repository Hygiene

Keep these out of public commits:

- `AGENTS.md` and local agent notes
- build directories and generated reports
- `__pycache__`, `.DS_Store`, IDE metadata, temporary logs, and local profiles
- sensitive access material, machine names, local absolute paths, and device configs
- stale architecture notes for removed routes

Do not use `git add .`. Stage only reviewed files.

## Files To Avoid Without Explicit Reason

- `third_party/llvm-project/`
- local ignored artifacts
- root-level technical notes outside this module
- backend source files unrelated to the requested behavior
- generated build-tree files
