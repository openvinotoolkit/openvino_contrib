---
name: gfx-plugin-dev
description: Use when working on OpenVINO GFX plugin code in modules/gfx_plugin, especially MLIR lowering, runtime stages, backend routing, plugin properties, or architecture-sensitive refactors across Metal and OpenCL backends.
---

# GFX Plugin Development

This skill is for implementing or refactoring the `GFX` OpenVINO plugin in
`modules/gfx_plugin/`.

## Use This Skill When

- The task touches `src/`, `include/`, `tests/`, CMake, or backend-specific
  plugin code.
- The user asks for architecture changes, new ops, runtime fixes, property
  behavior, backend routing, or MLIR/codegen work.
- The task mentions Metal, OpenCL, `GFX`, `CompiledModel`, `InferRequest`,
  remote tensors, MPSRT, MSL, source artifacts, or MLIR lowering.

## Read First

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/DEVELOPMENT.md`
4. `docs/TESTING.md`

Then inspect the relevant code path:

- plugin contract: `src/plugin/`
- compiler contracts: `src/compiler/`
- backend-neutral runtime: `src/runtime/`
- kernel manifests and source artifacts: `src/kernel_ir/`
- MLIR builders and source planning: `src/mlir/`
- graph rewrites: `src/transforms/`
- Metal backend: `src/backends/metal/`
- OpenCL backend: `src/backends/opencl/`
- public API surface: `include/openvino/gfx_plugin/`

## Development Rules

- Treat module docs as the public source of truth.
- Keep shared behavior in `src/plugin/`, `src/runtime/`, `src/kernel_ir/`, or
  `src/mlir/` unless the code is truly backend-specific.
- Keep backend target, operation support, lowering plans, manifests,
  executable bundles, and artifact descriptors in `src/compiler/`.
- Keep Metal-specific code under `src/backends/metal/` and OpenCL-specific code
  under `src/backends/opencl/`.
- Do not add CPU fallback for unsupported GPU stages.
- Do not reintroduce removed backend routes or runtime defines to keep alternate
  paths alive.
- Keep support probing, lowering, runtime binding, and tests aligned for each
  operation.
- Do not add plugin-side support tables that bypass `GfxCompilerService` or
  `BackendRegistry`.
- When changing plugin-visible behavior, check properties, `query_model()`,
  compiled-model properties, and docs.
- Do not modify `third_party/llvm-project/` unless the task explicitly requires
  vendored LLVM changes.

## Metal Work

For Metal placement, MPSRT, or MSL source changes, keep these aligned:

- `src/compiler/*`
- `src/backends/metal/compiler/`
- `src/runtime/gfx_stage_policy.*`
- `src/kernel_ir/gfx_kernel_manifest.hpp`
- `src/kernel_ir/gfx_custom_kernel_families.*`
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
- `src/backends/metal/runtime/metal_runtime_kernel_loader.*`
- `src/backends/metal/runtime/mpsrt_vendor_primitive_stage.*`
- `src/backends/metal/runtime/mpsrt/`

Manifest external-buffer roles are the semantic ABI. Do not let MSL buffer
scans or stale signature hints widen or shrink a typed MPSRT runtime contract.
MPS/MPSGraph vendor primitive routes must flow through the compiler
`VendorDescriptor` payload and the MPSRT vendor primitive stage; do not recreate
vendor descriptors from request-time node checks.

## OpenCL Work

For OpenCL source-artifact work:

1. Inspect `src/kernel_ir/gfx_opencl_source_artifacts.*` first.
2. Check `src/backends/opencl/compiler/` for route selection and kernel-unit
   registration.
3. Keep `src/backends/opencl/runtime/opencl_source_stage.*` generic.
4. Add source id, entry point, role ABI, scalar ABI, dynamic-shape metadata,
   constant materialization, chunk helpers, local size, and boolean-buffer rules
   to the artifact contract.
5. For embedded `.cl` units, add the wrapper under
   `src/kernel_ir/opencl_kernels/`, the `gfx_embed_kernel_source()` entry in
   `src/CMakeLists.txt`, and the source/header entries in
   `cmake/GfxSources.cmake`.
6. Update `tests/unit/gfx_opencl_source_artifacts_test.cpp` and
   `tests/unit/gpu_backend_base_test.cpp` when compiler payload routing changes.
7. Add runtime coverage only when dynamic OpenCL loading, memory, command
   enqueue, or runtime-shape behavior changed.

Standalone OpenCL Conv2D microbench tools are experiments. A result there is
not plugin support until it is promoted through support probing, source
artifacts, runtime binding, and tests.

## Common Workflows

### New Or Changed Op

1. Update support probing in `src/mlir/`.
2. Update compiler operation support, kernel registry, and artifact routing in
   `src/compiler/` or `src/backends/*/compiler/`.
3. Add lowering/source planning in `src/mlir/` or a transform in
   `src/transforms/`.
4. Express the shared contract through compiler manifest/executable records,
   stage policy, kernel manifests, runtime-value payloads, OpenCL artifacts, or
   MPSRT/Apple MSL plans.
5. Add backend code only at a real Metal/OpenCL boundary.
6. Add focused unit tests, then backend/functional tests if externally visible.
7. Update docs if supported shapes, route selection, properties, profiling, or
   test workflows changed.

### Runtime Or Scheduling Change

1. Inspect `gfx_stage_policy.*`, `gfx_parallelism.*`, and `gfx_partitioning.*`.
2. Check infer submission, immutable const caches, prepared binding reuse, and
   workspace allocation.
3. Add or update `tests/unit/gfx_stage_policy_test.cpp`,
   `tests/unit/gfx_parallelism_test.cpp`, `tests/unit/infer_submission_test.cpp`,
   or cache/reuse tests as appropriate.
4. For backend-visible behavior, add Metal or OpenCL runtime coverage.

### Property Or Device Selection

1. Update property parsing and property lists in `src/plugin/`.
2. Check `ov::available_devices`, `ov::device::id`, `ov::device::full_name`,
   `GFX_BACKEND`, and cache-related properties.
3. Add or update `tests/unit/plugin_tests.cpp`.
4. Update `README.md` and `docs/USAGE.md`.

## Validation

Use the smallest credible test set first:

```bash
cmake --build build-gfx-plugin --target ov_gfx_unit_tests
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_unit_tests> --gtest_filter=GfxStagePolicy.*
```

Before publication, run `git diff --check` and the relevant GFX tests. For
source or build-file changes, prefer a real build/test target over documentation
inspection only.

## Output Expectations

- Keep changes focused by subsystem.
- State which tests were run and which were skipped.
- Update local docs in the same pass when architecture, runtime semantics, or
  public properties changed.
- If the task reaches commit/push stage, apply the same plugin change set to the
  mirrored `ov-ext-labs/gfx-plugin` repository unless the user explicitly says
  otherwise.
