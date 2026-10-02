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
- common backend/device value types: `src/common/`
- stage-placement contracts: `src/compiler/stage_placement.*`
- shared stage policy: `src/compiler/stage_policy.*`
- stage compiler policy: `src/compiler/stage_compiler_policy.*`
- pipeline-stage descriptor builder: `src/compiler/pipeline_stage_builder.*`
- pipeline-stage fusion selection: `src/compiler/pipeline_stage_fusion.*`
- compiler-side pipeline-stage I/O contracts:
  `src/compiler/pipeline_stage_plan.*`
- runtime executable descriptor builder:
  `src/compiler/runtime_executable_descriptor_builder.*`
- memory/cache compiler contracts: `src/compiler/memory_plan.*` and
  `src/compiler/cache_envelope.*`, `src/compiler/cache_import.*`, and
  `src/compiler/cache_repository.*`
- tensor-layout contracts: `src/compiler/tensor_layout.*`
- backend-neutral runtime: `src/runtime/`
- backend runtime/provider interfaces:
  `src/runtime/backend_runtime.*`,
  `src/runtime/backend_runtime_provider.*`, and
  `src/runtime/backend_request_state.hpp`
- backend stage factory and runtime materialization:
  `src/runtime/backend_stage_factory.hpp`,
  `src/runtime/runtime_materialized_stage.hpp`,
  `src/runtime/pipeline_stage_plan.hpp`, and
  `src/runtime/runtime_execution_plan.*`,
  `src/runtime/runtime_stage_materializer.*`
- kernel manifests and shared source-artifact records: `src/kernel_ir/`
- MLIR builders and backend hooks: `src/mlir/`
- graph rewrites: `src/transforms/`
- Metal backend: `src/backends/metal/`
- OpenCL backend: `src/backends/opencl/`
- public API surface: `include/openvino/gfx_plugin/`
  (`plugin.hpp`, `profiling.hpp`, and `properties.hpp` only; compiled-model and
  infer-request headers are internal under `src/plugin/`)

## Development Rules

- Treat module docs as the public source of truth.
- Keep shared behavior in `src/plugin/`, `src/runtime/`, `src/kernel_ir/`, or
  `src/mlir/` unless the code is truly backend-specific.
- Keep backend target, operation support, lowering plans, compiler-side
  pipeline-stage I/O plans, pipeline-stage builders, fusion selection,
  manifests, executable bundles, runtime executable descriptor
  build/verification, tensor-layout plans, stage-placement value objects, and
  artifact descriptors in `src/compiler/`.
- Keep descriptor-backed stage creation, `RuntimeMaterializedStage`, vendor
  primitive artifact materialization, and fused sequence materialization in
  `src/runtime/backend_stage_factory.hpp`,
  `src/runtime/runtime_materialized_stage.hpp`, and
  `src/runtime/runtime_stage_materializer.*`.
- Keep descriptor element-type/static-shape parsing and generated source
  `RuntimeParams` ownership rules in `src/runtime/tensor_binding_contract.*`;
  do not duplicate them in Metal/OpenCL request-time code.
- Do not add `ov::Node` handoff routes to backend stages. Metadata needed by
  vendor primitives, runtime-shape handling, `ConstTensor` inputs, or generated
  `RuntimeParams` must be frozen into descriptor-owned payloads, or descriptor
  verification must fail closed.
- Keep compiler memory regions, alias groups, lifetimes, transient arenas,
  cache-envelope fingerprints, cache import contracts, and cache repository
  request keys in `src/compiler/`; request code consumes runtime descriptors,
  `RuntimeExecutionPlan`, and `RuntimeSession`.
- Keep fused-output lifetime and alias-storage planning in
  `src/runtime/fused_output_lifetime_plan.*`; do not reintroduce runtime-stage
  type-name checks for view or lifetime classification.
- Keep configured backend availability in CMake-selected source files and
  backend registration/stub translation units. Do not reintroduce generated
  `backend_config.hpp` headers or source-level macro branches for backend
  routing. The default `BackendRegistry` contains only backend modules
  registered by the configured build.
- Keep Metal-specific code under `src/backends/metal/` and OpenCL-specific code
  under `src/backends/opencl/`.
- Do not add CPU fallback for unsupported GPU stages.
- Do not reintroduce removed backend routes or runtime defines to keep alternate
  paths alive.
- Do not move pipeline construction/fusion logic back into `CompiledModel` or
  backend request code.
- Do not reintroduce the deleted standalone
  `src/backends/metal/runtime/mps_graph_attention_stage.*`; MPSGraph attention
  must flow through compiler-owned vendor artifacts and MPSRT vendor primitive
  execution.
- Do not reintroduce `BackendLowering`, `metal_lowering`, or source-signature
  scanning as ABI fallback. Generated/prebuilt executable routes must carry
  compiler-owned manifest ABI metadata.
- Keep support probing, lowering, runtime binding, and tests aligned for each
  operation.
- Do not add plugin-side support tables that bypass `GfxCompilerService` or
  `BackendRegistry`.
- Do not make runtime or MLIR stages resolve `BackendRegistry::default_registry()`
  to discover placement or post-op capabilities. Pass the selected backend
  policy through `GpuStageRuntimeOptions` and `compiler::StageCompilerPolicy`.
- When changing plugin-visible behavior, check properties, `query_model()`,
  compiled-model properties, and docs.
- Keep public compiled-model cache behavior explicit: `ov::cache_dir`,
  `export_model()`, and `import_model()` use the GFX `CacheEnvelope` contract
  when `compiled_model_cache_roundtrip_supported()` is true. Update
  `gfx_cache_public_contract_test.cpp` and docs whenever cache wire format,
  repository keys, import behavior, or backend payload codecs change.
- Do not modify `third_party/llvm-project/` unless the task explicitly requires
  vendored LLVM changes.
- Keep `third_party/clvk` and `third_party/clspv` as submodule gitlinks for the
  optional Raspberry/OpenCL bundle; do not vendor-copy their contents into a
  plugin commit.

## Metal Work

For Metal placement, MPSRT, or MSL source changes, keep these aligned:

- `src/compiler/*`
- `src/compiler/stage_compiler_policy.*`
- `src/compiler/pipeline_stage_builder.*`
- `src/compiler/pipeline_stage_fusion.*`
- `src/compiler/pipeline_stage_plan.*`
- `src/compiler/runtime_executable_descriptor_builder.*`
- `src/compiler/memory_plan.*`
- `src/runtime/fused_output_lifetime_plan.*`
- `src/runtime/backend_stage_factory.hpp`
- `src/runtime/runtime_materialized_stage.hpp`
- `src/runtime/pipeline_stage_plan.hpp`
- `src/runtime/runtime_execution_plan.*`
- `src/runtime/runtime_stage_materializer.*`
- `src/runtime/runtime_session.*`
- `src/backends/metal/compiler/`
- `src/backends/metal/compiler/metal_stage_placement.*`
- `src/backends/metal/compiler/metal_conv_post_op_fusion_contract.*`
- `src/backends/metal/compiler/metal_mpsrt_program_cache_payload_codec.*`
- `src/compiler/stage_policy.*`
- `src/runtime/view_only_stage.*`
- `src/kernel_ir/gfx_kernel_manifest.hpp`
- `src/kernel_ir/gfx_custom_kernel_families.*`
- `src/backends/metal/common/mpsrt/gfx_mpsrt_abi.hpp`
- `src/backends/metal/common/mpsrt/gfx_mpsrt_builder_plan.hpp`
- `src/backends/metal/common/mpsrt/gfx_mpsrt_plan.hpp`
- `src/backends/metal/common/mpsrt/gfx_mpsrt_program.hpp`
- `src/backends/metal/common/mpsrt/gfx_mpsrt_program_artifact_payload.hpp`
- `src/backends/metal/common/mpsrt/gfx_mpsrt_kernel_manifest_adapter.hpp`
- `src/backends/metal/runtime/mpsrt/gfx_mpsrt_model.*`
- `src/backends/metal/compiler/apple_mlir_stage_hooks.*`
- `src/backends/metal/compiler/apple_stage_pipeline.*`
- `src/backends/metal/compiler/apple_vendor_descriptors.*`
- `src/mlir/gfx_mpsrt_dialect.*`
- `src/mlir/gfx_mpsrt_ops.*`
- `src/backends/metal/compiler/apple_mpsrt_source_plan.hpp`
- `src/backends/metal/compiler/msl_codegen_apple_msl*.{cpp,hpp}`
- `src/backends/metal/compiler/msl_codegen_apple_mps.*`
- `src/backends/metal/compiler/msl_codegen_matmul_*`
- `src/backends/metal/compiler/msl_codegen_attention.*`
- `src/backends/metal/compiler/msl_codegen_compressed_matmul.*`
- `src/backends/metal/common/mpsrt/gfx_mpsrt_vendor_contract.hpp`
- `src/backends/metal/common/mpsrt/gfx_mpsrt_vendor_artifact_payload.hpp`
- `src/backends/metal/runtime/metal_runtime_kernel_loader.*`
- `src/backends/metal/runtime/mpsrt_program_stage.*`
- `src/backends/metal/runtime/mpsrt_vendor_primitive_stage.*`
- `src/backends/metal/runtime/mpsrt/`

Manifest external-buffer roles are the semantic ABI. Do not let MSL buffer
scans or stale signature hints widen, shrink, or replace a typed MPSRT runtime
contract.
MPS/MPSGraph vendor primitive routes must flow through the compiler
`VendorDescriptor` payload and the MPSRT vendor primitive stage; do not recreate
vendor descriptors from request-time node checks.
Generated activation and elementwise MSL routes must stay aligned across the
Metal operation policy, kernel registry, artifact materialization, and
`src/backends/metal/compiler/msl_codegen_apple_msl_activation.*` /
`src/backends/metal/compiler/msl_codegen_apple_msl_eltwise.*`. For `Swish`,
keep static-beta and runtime scalar-beta binding roles aligned with
`src/mlir/mlir_builder_unary.cpp` and the OpenCL artifact ABI.
Generated reduction MSL routes must stay aligned across
`src/backends/metal/compiler/msl_codegen_apple_msl_reduction.*`, embedded
sources under `src/kernel_ir/metal_kernels/reduction_*`,
`metal_kernel_registry.cpp`, and `metal_kernel_artifacts.cpp`.
Generated Softmax/LogSoftmax MSL routes must stay aligned across
`src/backends/metal/compiler/msl_codegen_apple_msl_softmax.*`, embedded sources
under `src/kernel_ir/metal_kernels/softmax_*` and
`src/kernel_ir/metal_kernels/logsoftmax_*`, `metal_kernel_registry.cpp`, and
`metal_kernel_artifacts.cpp`. Metal Pool2D must use the descriptor-backed MPS
vendor route; do not reintroduce the removed generic MSL Pool2D fallback.

## OpenCL Work

For OpenCL source-artifact work:

1. Inspect `src/backends/opencl/compiler/opencl_kernel_unit_catalog.*` first
   for the registered OpenCL routes, then inspect the family adapter and the
   family source-artifact builder in `opencl_*_source_artifact.cpp`.
2. Check `src/backends/opencl/compiler/` for route selection, kernel-unit
   registration, and source payload materialization through
   `opencl_kernel_artifacts.*`.
3. Keep `src/backends/opencl/runtime/opencl_source_stage.*` generic.
4. Use `src/backends/opencl/runtime/opencl_runtime_bundle.*` for plugin-local
   OpenCL/CLVK bundle discovery and bundled tool-path setup.
5. Add source id, entry point, role ABI, scalar ABI, source-static u32/f32
   scalars, dynamic-shape metadata, constant materialization, chunk helpers,
   local size, boolean-buffer rules, and cache-payload fields to the
   backend-owned artifact contract.
6. For embedded `.cl` units, add the wrapper under
   `src/kernel_ir/opencl_kernels/`, the `gfx_embed_kernel_source()` entry in
   `src/CMakeLists.txt`, and the source/header entries in
   `cmake/GfxSources.cmake`.
7. Update the focused family test and
   `tests/unit/gfx_backend_architecture_contract_test.cpp` when generated
   kernel-unit registration changes. Use
   `tests/unit/gfx_opencl_source_artifacts_test.cpp` for shared backend-target
   and missing-route contracts, and use the split `gfx_opencl_*_source_artifacts`
   tests for family-specific source-artifact assertions. Add
   `tests/unit/gpu_backend_base_test.cpp` only when compiler executable-bundle
   payload routing changes.
   For reduction source units, update
   `tests/unit/gfx_reduction_kernel_contract_test.cpp` and the shared
   `tests/unit/gfx_opencl_source_artifact_verifier.hpp` helper. For Conv2D,
   Softmax, and Pool2D source units, update
   `tests/unit/gfx_conv_kernel_contract_test.cpp`,
   `tests/unit/gfx_softmax_kernel_contract_test.cpp`,
   `tests/unit/gfx_pool_kernel_contract_test.cpp`, and the shared verifier.
8. Add runtime coverage only when dynamic OpenCL loading, memory, command
   enqueue, or runtime-shape behavior changed.

Current registered generated OpenCL routes include activation, elementwise,
f32 Conv2D/GroupConv2D, f32 MatMul, f32/f16 Softmax,
dynamic-static-rank f32/f16 Softmax, f32/f16 Pool2D, f32/f16/i64 Range,
f32/f16 Interpolate, ShapeOf, Tile, compare/select, logical-bool elementwise,
f32 numeric reduction, and boolean logical reduction. Activation, Eltwise,
Conv2D/GroupConv2D, Interpolate, MatMul, Pool2D, Range, Reduction, ShapeOf,
Softmax, and Tile have family-specific OpenCL kernel-unit adapters under
`src/backends/opencl/compiler/` and are listed in
`opencl_kernel_unit_catalog.*`. The current OpenCL kernel registry has no
active handwritten kernel-unit exception. Transpose, Concat, and Split are
current OpenCL catalog limitations and should continue to report
`missing_opencl_*_kernel_unit` until catalog entries, family adapters, payload
resolvers, and tests are added. OpenCL MatMul has a generated static f32 route;
f16 and unsupported variants should remain rejected by the MatMul family
contract.

Standalone OpenCL Conv2D microbench tools remain experimental probes. Plugin
support must flow through `opencl_conv_kernel_unit.*`, support probing,
generated source artifacts, runtime binding, and contract tests.

## Common Workflows

### New Or Changed Op

1. Update support probing in `src/mlir/`.
2. Update compiler operation support, kernel registry, and artifact routing in
   `src/compiler/` or `src/backends/*/compiler/`.
3. Add lowering in `src/mlir/`, backend source planning under the matching
   `src/backends/*/compiler/` directory, or a transform in `src/transforms/`.
4. Express the shared contract through compiler manifest/executable records,
   backend stage placement, shared stage policy, kernel manifests,
   runtime-value payloads, OpenCL artifacts, or MPSRT/Apple MSL plans.
5. Add backend code only at a real Metal/OpenCL boundary. Executable backend
   stages must consume their `RuntimeStageExecutableDescriptor`; do not
   reconstruct kernel source, artifact payload, or vendor descriptors from the
   OpenVINO node in request-time code.
   For OpenCL, family artifact construction belongs in
   `src/backends/opencl/compiler/opencl_*_source_artifact.cpp` and payload
   resolution/codec logic belongs in
   `src/backends/opencl/compiler/opencl_kernel_artifacts.*`, not in the common
   executable-bundle builder.
6. Add focused unit tests, then backend/functional tests if externally visible.
7. Update docs if supported shapes, route selection, properties, profiling, or
   test workflows changed.

### Runtime Or Scheduling Change

1. Inspect `src/compiler/stage_placement.*`, backend
   `*_stage_placement.*`, `src/compiler/stage_policy.*`,
   `src/runtime/gfx_parallelism.*`, and `src/runtime/gfx_partitioning.*`.
2. Check infer submission, immutable const caches, prepared binding reuse, and
   workspace allocation.
3. Add or update `tests/unit/gfx_stage_policy_test.cpp`,
   `tests/unit/gfx_parallelism_test.cpp`, `tests/unit/infer_submission_test.cpp`,
   or cache/reuse tests as appropriate.
4. For backend-visible behavior, add Metal tests or focused
   `ov_gfx_runtime_micro_tests` scenarios as appropriate.

### Property Or Device Selection

1. Update property parsing and property lists in `src/plugin/`.
2. Check `ov::available_devices`, `ov::device::id`, `ov::device::full_name`,
   `GFX_BACKEND`, and cache-related properties.
3. Add or update `tests/unit/plugin_tests.cpp` and
   `tests/unit/gfx_cache_public_contract_test.cpp` when cache visibility or
   import/export behavior changes.
4. Update `README.md` and `docs/USAGE.md`.

## Validation

Use the smallest credible test set first:

```bash
cmake --build build-gfx-plugin --target ov_gfx_unit_tests
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_unit_tests> --gtest_filter=GfxStagePolicy.*
```

Before publication, run `git diff --check` and the relevant GFX tests for normal
source changes. For documentation/security publication tasks, do not run build
or test targets unless the user explicitly asks; use source inspection,
security grep, stale-reference grep, and staged diff review. Use
`tests/tools/gfx_gtest_matrix.py` for production gtest registration checks when
test target composition changes; do not use source parsing as architecture
readiness evidence.

## Output Expectations

- Keep changes focused by subsystem.
- State which tests were run and which were skipped.
- Update local docs in the same pass when architecture, runtime semantics, or
  public properties changed.
- If the task reaches commit/push stage, apply the same plugin change set to the
  mirrored `ov-ext-labs/gfx-plugin` repository unless the user explicitly says
  otherwise.
