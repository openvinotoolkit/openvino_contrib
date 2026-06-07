# Development Guide

This guide is for contributors working inside `modules/gfx_plugin`.

## Prerequisites

- CMake 3.13 or newer
- Ninja recommended
- OpenVINO Developer Package
- Python 3 when `ENABLE_TESTS=ON`
- Metal SDK/frameworks on macOS for the Metal backend
- OpenCL runtime on non-Apple targets for the OpenCL source-kernel backend
- Optional `third_party/clvk` and `third_party/clspv` submodules when building
  the Raspberry/Linux OpenCL bundle

The module vendors LLVM/MLIR in `third_party/llvm-project` and builds the
required MLIR components as part of the CMake flow. The CLVK and CLSPV
directories are submodules used only by the optional Raspberry/OpenCL bundle.
Do not modify vendored LLVM or third-party submodule contents unless a task
explicitly requires it.

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
- `GFX_ENABLE_RASPBERRY_OPENCL_TOOLCHAIN`
- `GFX_RASPBERRY_CLVK_SOURCE_DIR`
- `GFX_RASPBERRY_CLSPV_SOURCE_DIR`
- `GFX_RASPBERRY_OPENCL_BUILD_DIR`
- `GFX_RASPBERRY_OPENCL_BUNDLE_DIR`
- `GFX_DEFAULT_BACKEND=auto|metal|opencl`
- `ENABLE_TESTS`

On macOS, `cmake/GfxBackendConfig.cmake` disables OpenCL and resolves the Apple
route to Metal/MPS/MPSRT/MSL. On non-Apple builds, the OpenCL source backend is
the runtime route when it is enabled in the build.

Build-system notes:

- `cmake/GfxSources.cmake` is the source list used by both the module root and
  `src/CMakeLists.txt`.
- `src/CMakeLists.txt` embeds selected `.cl` and `.metal` helper sources through
  `cmake/KernelSource.hpp.in`; generated headers stay in the build tree and
  must not be committed.
- `gfx_plugin_core`, `gfx_runtime_common`, and `gfx_mlir_stage_support` contain
  shared plugin, runtime, and MLIR stage support code.
- `src/common/` owns shared backend id, backend availability, artifact payload,
  and device-family/profile types used by compiler, runtime, and backend
  layers.
- `gfx_opencl_kernel_artifacts` contains OpenCL source-artifact payload
  materialization and embedded OpenCL helper wrappers.
- `src/compiler/stage_placement.*` defines the shared stage-placement contract;
  backend decisions live in
  `src/backends/metal/compiler/metal_stage_placement.*` and
  `src/backends/opencl/compiler/opencl_stage_placement.*`.
- `src/compiler/stage_compiler_policy.*` adapts backend capabilities into the
  explicit policy passed into shared stage-policy code and backend
  source-planning hooks.
- `src/compiler/backend_module_provider.*` builds the configured default
  compiler backend modules from `src/backends/*/compiler/*_backend_module.*`.
- `src/runtime/backend_runtime_provider.*` owns runtime provider registration
  for backend-state creation, backend infer execution, and profiling trace-sink
  registration.
- `src/compiler/pipeline_stage_builder.*` owns stage descriptor construction,
  node-to-stage mapping, and backend-policy handoff from `CompiledModel` while
  consuming compiler-owned plan/fusion contracts.
- `src/compiler/pipeline_stage_fusion.*` owns compiler-side fusion selection,
  fusion contracts, residual-add detection, and vendor attention planning.
- `src/compiler/pipeline_stage_plan.*` owns compiler-side compiled-pipeline
  input links, model-output flags, output-source metadata, and fused output
  aliases.
- `src/compiler/runtime_executable_descriptor_builder.*` owns conversion and
  verification from compiler executable bundles into runtime executable
  descriptors. Runtime-facing stage plans are separate compiler outputs and must
  not be embedded into cacheable descriptors.
- `src/compiler/memory_plan.*` owns compiler memory regions, lifetimes, alias
  groups, and transient arenas; request code must consume the runtime descriptor
  instead of reconstructing this information.
- `src/compiler/cache_envelope.*` builds cache metadata and fingerprints,
  serializes/deserializes the envelope wire format, and stores/loads envelopes
  by stable key. It is not a public OpenVINO compiled-model cache or a persisted
  native backend executable cache in the current code.
- `src/runtime/runtime_session.*` owns request-local descriptor binding tables
  and prepared executable objects.
- `src/runtime/backend_stage_factory.hpp` is the backend-facing runtime stage
  creation interface implemented by backend state.
- `src/runtime/pipeline_stage_desc.hpp` owns the pipeline descriptor record
  shared by compiled model, infer planning, and stateful helpers.
- `src/runtime/pipeline_stage_materializer.*` owns runtime descriptor lookup,
  descriptor-backed backend stage creation, vendor primitive artifact
  materialization, and fused sequence materialization.
- `src/runtime/stage_materialization_context.hpp` is the runtime handoff object
  for compiler-owned `RuntimeStageExecutableDescriptor` records. It is
  descriptor-only and does not expose OpenVINO source graph identity to backend
  runtime code.
- `src/runtime/tensor_binding_contract.*` owns descriptor element-type/static
  shape parsing and the generated source `RuntimeParams` ownership rules shared
  by Metal and OpenCL runtime code.
- `src/runtime/descriptor_const_tensor_materializer.*` owns shared
  materialization of descriptor-owned `ConstTensor` payloads. Do not rebuild
  constant inputs from request-time OpenVINO source nodes in backend code.
- `src/runtime/output_lifetime.hpp` and
  `src/runtime/fused_output_lifetime_plan.*` own fused-stage output lifetime and
  alias-storage planning from runtime memory descriptors.
- `gfx_metal_mpsrt_contract` contains shared Metal MPSRT model/ABI contracts
  used by Metal runtime and focused contract tests.
- `gfx_plugin_metal` / `gfx_runtime_metal` and
  `gfx_plugin_opencl` / `gfx_runtime_opencl` contain backend-specific code.
- `src/common/backend_config.hpp.in` is configured into the build tree with
  backend availability booleans and the resolved default backend;
  `src/compiler/backend_config.hpp.in` includes that common header for existing
  compiler-path includes. Explicit `GFX_DEFAULT_BACKEND=metal|opencl` requests
  fail CMake if the requested backend is unavailable.
- The OpenCL backend dynamically loads the target OpenCL runtime; it does not
  require a compile-time OpenCL SDK link.
- `src/backends/opencl/runtime/opencl_runtime_bundle.*` owns plugin-local
  OpenCL/CLVK bundle candidate ordering and CLVK tool environment setup.
- `cmake/GfxRaspberryOpenCLToolchain.cmake` can build and stage a plugin-local
  CLVK/CLSPV OpenCL bundle on Linux ARM targets when OpenCL support is
  available. It expects initialized `third_party/clvk`, `third_party/clspv`,
  and CLVK dependency submodules, plus host LLVM tools from the target
  toolchain layout.
- `cmake/InstallRaspberryOpenCLBundle.cmake` stages `libOpenCL.so.0.1`,
  `clspv`, optional `llvm-spirv`, local `libOpenCL.so*` symlinks, and optional
  TBB runtime library families found through runtime output directories,
  `TBBROOT`, or `TBB_DIR` into `GFX_RASPBERRY_OPENCL_BUNDLE_DIR`.
- `cmake/WriteGfxTestPluginsXml.cmake` writes the controlled test
  `plugins.xml` used by GFX test binaries.
- `tests/tools/gfx_gtest_matrix.py` can capture `--gtest_list_tests` from the
  production test binaries and reject duplicate or disabled registrations.
  Architecture readiness must not be proven through source/file/string-presence
  checks; use executed contract coverage, route coverage, backend conformance
  tests, and profiling evidence.
- Android and generic cross builds forward toolchain settings into the vendored
  LLVM/MLIR configure step.
- The build treats warnings as errors through `-Werror` on Clang/GCC and `/WX`
  on MSVC.

## Where To Start Reading

Read in this order:

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `src/plugin/plugin.cpp`
4. `src/compiler/gfx_compiler_service.*`
5. `src/compiler/backend_registry.*`
6. `src/compiler/backend_module_provider.*`
7. `src/compiler/manifest.*` and `src/compiler/executable_bundle.*`
8. `src/compiler/pipeline_stage_builder.*`
9. `src/compiler/pipeline_stage_fusion.*`
10. `src/compiler/pipeline_stage_plan.*`
11. `src/compiler/runtime_executable_descriptor_builder.*`
12. `src/compiler/memory_plan.*` and `src/compiler/cache_envelope.*`
13. `src/compiler/tensor_layout.*`
14. `src/compiler/stage_placement.*` and
   `src/compiler/stage_compiler_policy.*`
15. `src/compiler/stage_policy.*`
16. `src/runtime/backend_runtime.*` and
   `src/runtime/backend_runtime_provider.*`
17. `src/runtime/backend_stage_factory.hpp`
18. `src/runtime/pipeline_stage_desc.hpp`
19. `src/runtime/pipeline_stage_plan.hpp`
20. `src/runtime/pipeline_stage_materializer.*`
21. `src/runtime/infer_pipeline.*`, `src/runtime/infer_executor.*`, and
   `src/runtime/infer_submission.*`
22. `src/plugin/compiled_model.cpp`
23. the backend directory you are changing

For runtime planning, also inspect:

- `src/compiler/lowering_planner.*`
- `src/compiler/operation_support.*`
- `src/compiler/kernel_registry.*`
- `src/compiler/tensor_layout.*`
- `src/compiler/stage_placement.*`
- `src/compiler/stage_compiler_policy.*`
- `src/compiler/stage_policy.*`
- `src/compiler/pipeline_stage_builder.*`
- `src/compiler/pipeline_stage_fusion.*`
- `src/compiler/pipeline_stage_plan.*`
- `src/compiler/runtime_executable_descriptor_builder.*`
- `src/compiler/memory_plan.*`
- `src/compiler/cache_envelope.*`
- `src/runtime/backend_runtime.*`
- `src/runtime/backend_runtime_provider.*`
- `src/runtime/backend_request_state.hpp`
- `src/runtime/backend_stage_factory.hpp`
- `src/runtime/pipeline_stage_desc.hpp`
- `src/runtime/pipeline_stage_plan.hpp`
- `src/runtime/pipeline_stage_materializer.*`
- `src/runtime/runtime_session.*`
- `src/runtime/infer_pipeline.*`
- `src/runtime/infer_executor.*`
- `src/runtime/infer_submission.*`
- `src/runtime/fused_output_lifetime_plan.*`
- `src/runtime/output_lifetime.hpp`
- `src/runtime/gfx_parallelism.*`
- `src/runtime/gfx_partitioning.*`
- `src/runtime/executable_descriptor.*`
- `src/runtime/view_only_stage.*`
- `src/runtime/gfx_target_profile.*`
- `src/runtime/gfx_profiling_trace_sink.*`
- `src/kernel_ir/gfx_kernel_manifest.hpp`
- `src/kernel_ir/gfx_custom_kernel_families.*`
- `src/kernel_ir/gfx_kernel_source.*`
- `src/kernel_ir/gfx_opencl_source_artifacts.*`
- `src/runtime/gfx_stage_runtime_values.*`
- `src/mlir/gfx_backend_custom_kernel_adapter.*`

For Metal placement, MPSRT, or MSL source planning, also inspect:

- `src/backends/metal/compiler/`
- `src/backends/metal/compiler/metal_stage_placement.*`
- `src/backends/metal/compiler/apple_mlir_stage_hooks.*`
- `src/backends/metal/compiler/apple_stage_pipeline.*`
- `src/backends/metal/compiler/apple_vendor_descriptors.*`
- `src/backends/metal/compiler/apple_mpsrt_source_plan.hpp`
- `src/backends/metal/compiler/msl_codegen_apple_msl*.{cpp,hpp}`
- `src/backends/metal/compiler/msl_codegen_apple_mps.*`
- `src/backends/metal/compiler/msl_codegen_matmul_*`
- `src/backends/metal/compiler/msl_codegen_attention.*`
- `src/backends/metal/compiler/msl_codegen_compressed_matmul.*`
- `src/backends/metal/common/mpsrt/gfx_mpsrt_abi.hpp`
- `src/backends/metal/common/mpsrt/gfx_mpsrt_plan.hpp`
- `src/backends/metal/common/mpsrt/gfx_mpsrt_program.hpp`
- `src/backends/metal/common/mpsrt/gfx_mpsrt_kernel_manifest_adapter.hpp`
- `src/backends/metal/common/mpsrt/gfx_mpsrt_vendor_contract.hpp`
- `src/backends/metal/common/mpsrt/gfx_mpsrt_vendor_artifact_payload.hpp`
- `src/backends/metal/runtime/metal_runtime_kernel_loader.*`
- `src/backends/metal/runtime/mpsrt_vendor_primitive_stage.*`
- `src/backends/metal/runtime/mpsrt/gfx_mpsrt_model.*`
- `src/mlir/gfx_mpsrt_dialect.*`
- `src/mlir/gfx_mpsrt_ops.*`
- `src/backends/metal/runtime/mpsrt/`

For OpenCL source execution, start with:

- `src/backends/opencl/compiler/`
- `src/backends/opencl/compiler/opencl_stage_placement.*`
- `src/backends/opencl/compiler/opencl_kernel_artifacts.*`
- `src/backends/opencl/compiler/opencl_range_kernel_unit.*`
- `src/backends/opencl/compiler/opencl_softmax_kernel_unit.*`
- `src/backends/opencl/compiler/opencl_tile_kernel_unit.*`
- `src/backends/opencl/plugin/`
- `src/backends/opencl/runtime/opencl_api.*`
- `src/backends/opencl/runtime/opencl_runtime_bundle.*`
- `src/backends/opencl/runtime/opencl_buffer_manager.*`
- `src/backends/opencl/runtime/opencl_program_cache.*`
- `src/backends/opencl/runtime/opencl_runtime_kernel_loader.*`
- `src/backends/opencl/runtime/opencl_source_stage.*`
- `src/backends/opencl/runtime/stage_factory.*`
- `tests/unit/gfx_opencl_source_artifacts_test.cpp`
- `tests/unit/gfx_opencl_source_artifact_verifier.hpp`
- `tests/unit/gfx_activation_contract_cases.hpp`
- `tests/unit/gfx_activation_opencl_contract_cases.cpp`
- `tests/unit/gfx_eltwise_contract_cases.hpp`
- `tests/unit/gfx_eltwise_opencl_contract_cases.cpp`
- `tests/unit/gfx_reduction_kernel_contract_test.cpp`

## Adding Or Changing An Operation

1. Update support probing in `src/mlir/`.
2. Update compiler route selection when support changes:
   - common compiler contracts in `src/compiler/`
   - Metal policy, kernel registry, or artifact resolver in
     `src/backends/metal/compiler/`
   - OpenCL policy or kernel registry in `src/backends/opencl/compiler/`
3. Add or adjust lowering in the relevant `mlir_builder_*.cpp`, backend
   compiler source-plan helper, or transform.
4. Decide the shared runtime contract before adding backend code:
   - compiler manifest/executable bundle for stage ABI and artifact payloads
   - compiler-owned runtime executable descriptor builder for runtime ABI
     validation and stage-plan attachment
   - compiler pipeline-stage builder for stage descriptor construction and
     node-to-stage mapping
   - compiler pipeline-stage fusion plan for post-op, residual-add, attention,
     and vendor attention fusion selection
   - compiler pipeline-stage I/O plan for input links, model-output flags, and
     output aliases
   - runtime pipeline-stage materializer for descriptor-backed stage creation
     and vendor attention artifact materialization
   - compiler memory plan for region ids, alias groups, lifetimes, and arenas
   - compiler tensor-layout plan for view-only/materialized layout contracts
   - runtime tensor-binding contract helpers for descriptor-owned element types,
     static shapes, and generated source `RuntimeParams` payload ownership
   - fused-output lifetime plan for aliasing outputs inside a fused stage
   - backend stage-placement policy for domain/storage selection
   - selected backend compiler policy passed through `GpuStageRuntimeOptions`
     and `compiler::StageCompilerPolicy`
   - runtime session binding tables for request-local resource preparation
   - runtime pipeline-stage plan for materialization, separate from the cacheable
     runtime executable descriptor
   - shared stage policy for fusion, precision, and submission
   - kernel manifest for custom-kernel ABI
   - runtime-value payloads for dynamic metadata
   - OpenCL source artifact for source-kernel execution
   - MPSRT/Apple MSL source plan for Metal execution
5. Add backend code only where the shared contract crosses into Metal or OpenCL.
   Backend stage creation now requires the matching runtime executable
   descriptor for executable routes; do not reconstruct kernel or vendor
   payloads from the OpenVINO node in request-time code.
   If a route still needs `ov::Node` metadata, it is not descriptor-owned yet:
   freeze the required metadata into the compiler artifact/descriptor or make
   descriptor verification fail closed. Do not add request-time source-node
   bridges.
6. Add focused unit tests first.
7. Add backend or functional tests when behavior is externally visible.
8. Update docs when public properties, supported shapes, route selection,
   profiling, or test workflow changes.

Common operation families that need extra care:

- dynamic-shape data movement: `ShapeOf`, `Concat`, `Broadcast`, `Select`,
  `Slice`, `Range`, and `Tile`
- compiler-owned tensor-layout classification for view-only versus materialized
  `Reshape`, `Squeeze`, `Unsqueeze`, `ReadValue`, and `Transpose`
- backend-owned stage-placement policies in
  `src/backends/*/compiler/*_stage_placement.*`, where Metal may choose
  Apple MPS/MSL domains and OpenCL currently uses buffer-backed source kernels
- stateful `ReadValue` / `Assign`
- view-style `Split` / `VariadicSplit` aliases
- runtime-shape argument requirements carried by `KernelUnit`,
  `StageRecord`, artifact descriptors, and runtime descriptors
- descriptor-owned generated source `RuntimeParams` payloads for Broadcast,
  binary elementwise broadcast, `Select`, `Tile`, Softmax/LogSoftmax,
  Transpose, and Reduce families; keep the ownership rules in
  `src/runtime/tensor_binding_contract.*`
- Metal MPS/MPSGraph vendor descriptors, vendor attention artifacts, and MPSRT
  storage bridges
- Metal custom MSL source plans with explicit kernel-buffer order
- OpenCL source artifacts with scalar ABI, static u32/f32 scalars, constants,
  chunking, and boolean output padding
- backend-owned OpenCL source payload materialization in
  `src/backends/opencl/compiler/opencl_kernel_artifacts.*`
- OpenCL generated kernel units such as activation, elementwise, f32 MatMul,
  f32 Conv2D/GroupConv2D, bounded f32/f16 Interpolate, f32 reduction, boolean
  reduction, f32/f16 Softmax, dynamic-static-rank f32/f16 Softmax,
  f32/f16 Pool2D, f32/f16/i64 Range, ShapeOf, Tile, Transpose, compare/select,
  logical-bool elementwise, and generated Concat/Split
- OpenCL reduction routes, where numeric f32 reductions and logical boolean
  reductions use separate generated source ids but the same static axis
  metadata contract
- generated activation `Swish` routes, where default/static beta and runtime
  scalar beta must keep the MLIR, Metal MSL, and OpenCL artifact contracts
  aligned
- generated Softmax routes, where Metal `Softmax`/`LogSoftmax` and OpenCL
  `Softmax` must keep axis normalization, scalar metadata, and kernel-unit ids
  aligned with their backend-specific contracts
- Convolution routes, where OpenCL generated Conv2D/GroupConv2D covers f32
  static 4D NCHW data/output with constant weights and Metal Conv2D/GroupConv2D
  must use valid descriptor-backed MPS vendor routes
- Pooling routes, where OpenCL generated Pool2D covers static f32/f16 NCHW
  window contracts and Metal Pool2D must use a valid MPS-family vendor route
- LLM-oriented fusions such as `RoPE`, compressed `MatMul`, and SDPA variants

## Shared Versus Backend-Specific Code

Prefer these shared locations:

- backend target, operation legality, route selection, manifest, executable
  bundles, pipeline-stage builders/fusion planners, tensor-layout plans,
  stage-placement value objects, and artifact descriptors: `src/compiler/`
- graph rewrites: `src/transforms/`
- support probing and lowering: `src/mlir/`
- stage fusion/precision/submission policy and parallelism:
  `src/compiler/stage_policy.*`, `src/runtime/gfx_parallelism.*`, and
  `src/runtime/gfx_partitioning.*`
- descriptor-backed stage creation and pipeline descriptors:
  `src/runtime/backend_stage_factory.hpp`,
  `src/runtime/pipeline_stage_desc.hpp`, and
  `src/runtime/pipeline_stage_materializer.*`
- binding and manifest contracts: `src/kernel_ir/` and
  `src/mlir/gfx_backend_custom_kernel_adapter.*`
- infer planning and submission: `src/runtime/infer_pipeline.*`,
  `src/runtime/infer_executor.*`, and `src/runtime/infer_submission.*`

Use backend directories only for real backend boundaries:

- Metal operation support policy, Metal stage placement, Metal kernel registry,
  and generated MSL payload materialization belong under
  `src/backends/metal/compiler/`.
- Metal Objective-C++ APIs, MTL resources, MPS/MPSGraph setup, MSL compilation,
  and command encoding belong under `src/backends/metal/`.
- OpenCL operation support policy, OpenCL stage placement, and OpenCL kernel
  registry belong under `src/backends/opencl/compiler/`.
- OpenCL source-artifact payload resolution belongs in
  `src/backends/opencl/compiler/opencl_kernel_artifacts.*`.
- OpenCL platform/device discovery, plugin-local runtime-bundle discovery,
  program compilation, buffer management, and kernel enqueue code belong under
  `src/backends/opencl/`.

Do not duplicate shared ABI, route, or shape rules in backend request code.

## OpenCL Source Artifacts

`src/kernel_ir/gfx_opencl_source_artifacts.*` is the source of truth for:

- source id and entry point
- tensor role order
- scalar ABI
- source-static u32/f32 scalar values
- local size
- element-count source
- dynamic-shape scalar metadata
- direct constant materialization
- boolean buffer padding
- generated Concat/Split chunk artifacts

`src/backends/opencl/runtime/opencl_source_stage.*` should stay a generic
artifact executor. Add metadata to artifacts rather than adding op-specific
branches to the executor.

`opencl/generated/transpose_f32` is the current Transpose route. The current
OpenCL kernel registry has no active handwritten kernel-unit exception. Do not
introduce a baseline exception unless the generated-source contract cannot
express the route and the exception is documented, tested, and reviewed.

Generated or embedded source payloads should flow through compiler artifact
descriptors and runtime kernel loaders. Do not pass ad-hoc source strings
through plugin or infer-request properties.
`ExecutableBundleBuilder` does not synthesize OpenCL source payloads by itself;
the selected OpenCL backend module must provide them through its artifact
payload resolver.

When adding an embedded OpenCL source unit:

- place the `.cl`, `.cpp`, and `.hpp` wrapper under
  `src/kernel_ir/opencl_kernels/`
- add the source to `src/CMakeLists.txt` through `gfx_embed_kernel_source()`
- add the wrapper source/header to `cmake/GfxSources.cmake`
- route it from `gfx_opencl_source_artifacts.*` with explicit source id,
  entry point, route kind, scalar ABI, and shape/type limitations
- cover source identity, scalar metadata, support probing, and payload routing
  in `tests/unit/gfx_opencl_source_artifacts_test.cpp`,
  family-specific contract case files, and
  `tests/unit/gpu_backend_base_test.cpp` when the compiler bundle is affected
- update `tests/unit/gfx_backend_architecture_contract_test.cpp` when kernel
  registry, backend-target identity, or manifest-routing contracts change

For generated activation changes, update
`tests/unit/gfx_activation_contract_cases.hpp`,
`tests/unit/gfx_activation_opencl_contract_cases.cpp`, and
`tests/unit/gfx_activation_msl_contract_cases.cpp` together. For generated
elementwise OpenCL changes, update `tests/unit/gfx_eltwise_contract_cases.hpp`,
`tests/unit/gfx_eltwise_opencl_contract_cases.cpp`, and
`tests/unit/gfx_eltwise_opencl_source_artifacts_test.cpp`.

For reduction source-unit changes, update
`tests/unit/gfx_reduction_kernel_contract_test.cpp` and the shared
`tests/unit/gfx_opencl_source_artifact_verifier.hpp` helper. Keep numeric f32
and logical boolean generated reduction source ids, entry points, static u32
metadata, kernel registry entries, and Metal/OpenCL artifact payloads aligned.

For Softmax source-unit changes, update
`tests/unit/gfx_softmax_kernel_contract_test.cpp` and the shared
`tests/unit/gfx_opencl_source_artifact_verifier.hpp` helper. Keep Metal
f32/f16 Softmax and LogSoftmax source ids, OpenCL static and dynamic-static-rank
Softmax source ids, runtime-parameter roles, scalar metadata, and kernel
registry entries aligned.

For Conv2D source-unit changes, update
`tests/unit/gfx_conv_kernel_contract_test.cpp`,
`tests/unit/gfx_backend_architecture_contract_test.cpp`, and the shared
`tests/unit/gfx_opencl_source_artifact_verifier.hpp` helper. Keep OpenCL
Conv2D/GroupConv2D f32 generated source ids, constant-weight tensor bindings,
scalar metadata, descriptor finalization, and Metal MPS vendor routes aligned.

For Pool2D source-unit changes, update
`tests/unit/gfx_pool_kernel_contract_test.cpp`,
`tests/integration/gfx_pooling_func_test.cpp` when externally visible behavior
changes, and `tests/unit/gfx_backend_architecture_contract_test.cpp` when
kernel-unit registration changes. Do not add a Metal MSL Pool2D fallback unless
there is explicit MPS-family rejection evidence and an op-owned narrow artifact
contract.

## Metal MPSRT And MSL

Metal placement must stay coordinated across:

- `src/compiler/stage_policy.*`
- `GfxKernelStageManifest`
- custom-kernel family metadata
- Apple source-plan helpers under `src/backends/metal/compiler/`
- typed `GfxMpsrtProgram` / generated `gfx_mpsrt_ops`
- `GfxMpsrtBuilderPlan`
- `src/backends/metal/common/mpsrt/`
- `src/backends/metal/runtime/mpsrt/gfx_mpsrt_model.*`
- `src/backends/metal/runtime/mpsrt/`

When a manifest supplies explicit external-buffer roles, those roles define the
runtime ABI. MSL buffer scans and signature hints are diagnostics only; they
must not widen, shrink, or replace a typed MPSRT binding contract.
Generated or prebuilt MSL kernels must carry compiler-owned manifest ABI counts;
source-signature scanning is not a current ABI fallback.

Embedded MPSRT helper kernels live under `src/kernel_ir/metal_kernels/` and are
exposed through runtime loaders. Keep helper source ownership there instead of
reintroducing large inline MSL strings in request encoders.

Compiler-owned Metal payloads now include both generated MSL sources and
MPS/MPSGraph `VendorDescriptor` payloads. When adding a vendor primitive route,
update the Metal operation policy, `metal_kernel_artifacts.*`, the typed vendor
descriptor helpers in `src/backends/metal/compiler/apple_vendor_descriptors.*`,
the shared contract/payload headers in `src/backends/metal/common/mpsrt/`, and
`mpsrt_vendor_primitive_stage.*` only if the existing runtime contract cannot
express the new primitive. Do not rebuild vendor descriptors from request-time
node checks.

Fused vendor attention routes are compiler/runtime descriptor routes. Add or
change the fusion contract in `src/compiler/pipeline_stage_fusion.*`, the
backend artifact resolver in `src/backends/metal/compiler/metal_kernel_artifacts.*`,
and the materialization path in `src/runtime/pipeline_stage_materializer.*`.
Do not reintroduce the deleted standalone
`src/backends/metal/runtime/mps_graph_attention_stage.*`.

Generated Metal activation, elementwise, reduction, Softmax, and Transpose paths
are planned through
`src/backends/metal/compiler/msl_codegen_apple_msl_activation.*`,
`src/backends/metal/compiler/msl_codegen_apple_msl_eltwise.*`,
`src/backends/metal/compiler/msl_codegen_apple_msl_reduction.*`, and
`src/backends/metal/compiler/msl_codegen_apple_msl_softmax.*`, plus layout
planning in `src/backends/metal/compiler/msl_codegen_apple_msl_layout.cpp`.
Keep those source plans aligned with
`src/backends/metal/compiler/metal_operation_support.cpp`,
`metal_kernel_registry.cpp`, `metal_kernel_artifacts.cpp`, and embedded helper
source wrappers under `src/kernel_ir/metal_kernels/`. For `Swish`, keep
static-beta and runtime-beta binding roles aligned with
`src/mlir/mlir_builder_unary.cpp` and the OpenCL source artifact ABI. For
Softmax, keep generated `Softmax` and `LogSoftmax` runtime-parameter roles
aligned with the registered f32/f16 kernel units.

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

Before ordinary source commits, run at least:

```bash
git diff --check
ctest --test-dir build-gfx-plugin --output-on-failure -L GFX
```

For documentation/security publication tasks, do not run build or test targets
unless explicitly requested. Use source inspection, targeted security/stale
reference grep, `git diff --check`, and staged diff review for that gate.
For compiler, manifest, or executable-descriptor changes, include
`GpuBackendBaseTest.*` or a narrower relevant filter from
`tests/unit/gpu_backend_base_test.cpp`.

For backend registry or generated source-unit changes, also include the focused
contract suites in `tests/unit/gfx_backend_architecture_contract_test.cpp`,
`tests/unit/gfx_activation_kernel_contract_test.cpp`,
`tests/unit/gfx_eltwise_kernel_contract_test.cpp`, and
`tests/unit/gfx_matmul_kernel_contract_test.cpp` as applicable.
For OpenCL runtime loader or plugin-local bundle changes, include
`tests/unit/gfx_opencl_runtime_bundle_contract_test.cpp` when OpenCL is enabled
or the matching `*_unavailable_test.cpp` adapter when it is not.

Use `tests/tools/gfx_gtest_matrix.py` to capture or compare
`--gtest_list_tests` output across production test targets. It detects
duplicates, forbidden `DISABLED_` registrations, and matrix drift; it does not
skip or filter tests. `gfx_gtest_matrix_compare` requires explicit
`GFX_GTEST_MATRIX_REFERENCE_ROOTS`, and cross-build host capture fails unless
`CMAKE_CROSSCOMPILING_EMULATOR` is configured. Native backend and
unavailable-adapter coverage must be kept aligned through executable test
registration, contract coverage, and route coverage rather than source parsing.

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
- `third_party/clvk/` and `third_party/clspv/` contents; stage them as
  submodule gitlinks only when the submodule pointers intentionally change
- local ignored artifacts
- root-level technical notes outside this module
- backend source files unrelated to the requested behavior
- generated build-tree files
