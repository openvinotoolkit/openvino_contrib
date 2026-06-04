# OpenVINO GFX Plugin

`GFX` is an experimental OpenVINO device plugin that compiles an `ov::Model`
into a stage-based GPU execution pipeline. The module lives as a standalone
OpenVINO contrib module and builds against an OpenVINO Developer Package.

The current implementation has two backend families:

- `metal`: macOS execution through Metal, Apple MPS/MPSGraph vendor
  primitives, MPSRT runtime-model records, and generated MSL custom kernels.
- `opencl`: non-Apple execution through a dynamically loaded OpenCL runtime,
  optional bundled CLVK/CLSPV deployment support, and manifest-backed
  source-kernel artifacts.

Unsupported models fail during `compile_model()` or support probing. The plugin
does not silently fall back to CPU for intermediate execution.

## Start Here

Read these files first:

- `docs/ARCHITECTURE.md`: current pipeline, runtime, MLIR, and backend split
- `docs/DEVELOPMENT.md`: build flow and development rules
- `docs/TESTING.md`: test targets, compare tooling, and validation strategy
- `docs/USAGE.md`: public properties, backend selection, and profiling usage

## Current Status

- Device name: `GFX`
- Runtime backends: `metal`, `opencl`
- Build-time backend options: `GFX_ENABLE_METAL`, `GFX_ENABLE_OPENCL`
- Runtime backend property: `GFX_BACKEND=metal|opencl`
- Default backend: `GFX_DEFAULT_BACKEND=auto|metal|opencl`
- Raspberry/OpenCL bundle option: `GFX_ENABLE_RASPBERRY_OPENCL_TOOLCHAIN`
- OpenCL is disabled by CMake on macOS; the Apple route is Metal/MPS/MPSRT/MSL.
- On non-Apple builds, `auto` resolves to OpenCL when the source backend is
  enabled in the build.
- Explicit default backend requests are strict: CMake fails when the requested
  backend is unavailable instead of selecting another backend.
- `query_model()` uses the same backend-aware support path as compilation.
- `export_model()` serializes the OpenVINO model, not a backend binary cache.
- `GfxRemoteContext` and `GfxRemoteTensor` exist; practical capabilities depend
  on the selected backend.

## Source Layout

- `include/openvino/gfx_plugin/`: public plugin headers and property names
- `src/common/`: backend ids, configured-backend availability values,
  artifact-payload interfaces, activation/bias metadata, dispatch and
  parallelism value types, submit-policy records, constant-evaluation helpers,
  and device-family/profile value types shared by compiler, runtime, and
  backend code
- `src/compiler/`: backend target registry, backend capability records,
  operation support policies, backend stage-placement contracts, tensor-layout
  classification, stage compiler policy, lowering-plan construction,
  pipeline-stage descriptor building, fusion selection, compiler-side
  pipeline-stage I/O planning, memory planning, manifest/cache-envelope
  building, executable-bundle assembly, runtime executable descriptor
  build/verification, backend-module construction, and artifact payload routing
  used by query and compilation
- `src/plugin/`: OpenVINO-facing `Plugin`, `CompiledModel`, infer-request
  API glue, properties, model serialization, backend selection, backend-access
  helpers, and variable-state OpenVINO API objects
- `src/runtime/`: backend-neutral stage interfaces, submission planning,
  profiling report assembly, runtime executable descriptor records, runtime
  pipeline-stage plans, runtime sessions, backend runtime/provider interfaces,
  backend request state, backend stage factory interfaces, infer
  pipeline/executor/submission helpers, stateful runtime helpers,
  pipeline-stage materialization, fused-output lifetime planning,
  liveness-aware output workspaces, remote context/tensor helpers,
  descriptor-backed view-only stages, runtime-value and kernel-launch planning,
  and target profiles
- `src/kernel_ir/`: backend-neutral kernel manifests, custom-kernel family
  metadata, cache keys, dispatch descriptions, embedded Metal/OpenCL helper
  sources, and OpenCL source artifacts
- `src/mlir/`: support probing, OpenVINO-node lowering, pass helpers, backend
  stage hooks, MPSRT dialect/metadata helpers, and generic MLIR kernel
  construction used by backend compiler routes
- `src/backends/metal/`: Metal plugin glue, Objective-C++ runtime, memory
  management, profiling, backend compiler module, Apple MSL/MPS/MPSRT source
  planning, shared MPSRT ABI records, MSL compilation, runtime kernel loading,
  MPSRT preparation, descriptor-backed vendor primitive stages, and
  MPS/MPSGraph vendor primitive execution
- `src/backends/opencl/`: OpenCL plugin glue, dynamic API loader, buffer
  manager, backend compiler policy, backend-owned source-artifact payload
  materialization, program cache, memory ops, runtime kernel loading, and
  generic source-artifact execution
- `src/transforms/`: OpenVINO graph rewrites, fusion passes, and layout cleanup
- `tests/`: unit, functional, backend, integration, compare, source-contract,
  gtest-registration, and microbench coverage
- `bench/`: optional local/remote evaluation helpers
- `tools/`: profiling and microbench post-processing helpers
- `third_party/llvm-project/`: vendored LLVM/MLIR used by the build
- `third_party/clvk/`, `third_party/clspv/`: optional submodules used by the
  Raspberry/OpenCL bundle flow when `GFX_ENABLE_RASPBERRY_OPENCL_TOOLCHAIN` is
  enabled

## Compilation And Execution Pipeline

The high-level path is:

1. `Plugin::compile_model()` parses properties and resolves `GFX_BACKEND`.
2. `src/compiler/BackendRegistry` resolves one of the backend compiler modules
   available in the current configured build. The module owns the immutable
   `BackendTarget`, backend-owned transform `PipelineOptions`, fusion
   capabilities, post-op fusion capabilities, stage-placement policy, source
   kernel dispatch policy, and artifact payload resolver. Default backend
   modules are built by `src/compiler/backend_module_provider.*` and
   backend-specific `*_backend_module.*` files.
3. `GfxCompilerService` runs backend-aware transforms, operation support
   checks, lowering-plan creation, memory-plan creation, manifest building,
   executable-bundle assembly, cache-envelope construction, and artifact payload
   materialization.
4. `CompiledModel` asks
   `src/compiler/runtime_executable_descriptor_builder.*` to build and verify a
   `RuntimeExecutableDescriptor` from the compiler executable bundle. The
   descriptor also carries the runtime `PipelineStageRuntimePlan` used for
   stage materialization.
5. `CompiledModel::build_op_pipeline()` consumes the descriptor-owned runtime
   plan and delegates concrete stage materialization to
   `src/runtime/pipeline_stage_materializer.*`. The compiler stage builder uses
   `src/compiler/pipeline_stage_plan.*` for model-output flags, input links,
   and output aliases, `src/compiler/pipeline_stage_fusion.*` for fusion
   selection, and emits runtime-facing plans in
   `src/runtime/pipeline_stage_plan.hpp`.
6. Pipeline records are `PipelineStageDesc` values from
   `src/runtime/pipeline_stage_desc.hpp`. Fused output lifetimes are derived
   from runtime memory contracts in
   `src/runtime/fused_output_lifetime_plan.*`.
7. `src/compiler/stage_policy.*` selects fusion, precision, submit policy,
   and other shared stage traits. The selected backend `StagePlacementPolicy`,
   `PostOpFusionCapabilities`, and source-kernel dispatch policy are passed
   through `GpuStageRuntimeOptions` / `compiler::StageCompilerPolicy`; stages
   must not resolve the global backend registry at request time.
8. `src/mlir/` lowers supported nodes and exposes backend hooks; Metal and
   OpenCL source/vendor payload planning lives in the matching
   `src/backends/*/compiler/` directory.
9. Runtime backend providers in `src/runtime/backend_runtime_provider.*` create
   the selected `BackendState` and dispatch infer execution to the active
   backend. The active backend creates descriptor-backed view-only stages or
   concrete backend `GpuStage` objects through `BackendStageFactory`.
10. Infer requests create a `RuntimeSession`, bind host or remote tensors against
   descriptor memory-region contracts, prepare kernel executables per request,
   allocate or reuse backend buffers, execute submission windows, and collect
   profiling data.

The pipeline is intentionally stage-based. Fused stages, descriptor-backed
view-only stages, view-style split outputs, stateful variable buffers, reusable
host outputs, prepared binding caches, immutable const caches, and
workspace-managed intermediate outputs all live in the shared runtime/plugin
layer instead of being duplicated per backend.
Runtime stages do not own tensor-view or output-lifetime classification; that
metadata comes from compiler/runtime descriptors.

The compiler layer is the public architecture boundary between OpenVINO graph
semantics and runtime execution. Do not add new support checks directly in
`Plugin` or backend request code when the same decision belongs in
`src/compiler/`.

Backend stage creation is descriptor-owned. Stages that need a kernel, source
artifact, vendor descriptor, or ABI metadata must consume the matching
`RuntimeStageExecutableDescriptor`; request-time code must not reconstruct
payloads from the OpenVINO node.

There is no generic backend-lowering escape route. Metal and OpenCL operation
support must resolve to common metadata, an explicit generated kernel unit, a
vendor primitive, or a consciously documented backend route.

## Backend Split

### Metal

The Metal backend is the Apple production path. It combines:

- Apple MPS/MPSGraph vendor primitives for selected Conv2D, GroupConv, Pool,
  Resize, MatMul/GEMM, Softmax, TopK, and attention-style stages
- Apple MSL custom kernels for general elementwise, layout, reduction, shape,
  slice, scatter/gather, RoPE, RMS, compressed MatMul, and SDPA helper paths
- backend compiler policy, generated MSL artifact payloads, and MPS/MPSGraph
  vendor descriptor payloads under
  `src/backends/metal/compiler/`
- shared MPSRT ABI and builder records under
  `src/backends/metal/common/mpsrt/`
- MPSRT runtime-model construction and request encoding under
  `src/backends/metal/runtime/mpsrt/`
- embedded MPSRT helper kernels under `src/kernel_ir/metal_kernels/`
- Objective-C++ request-time execution under `src/backends/metal/runtime/`

Metal placement is selected by the Metal compiler stage-placement policy and
consumed by shared stage policy. Do not bypass those contracts with ad-hoc
backend switches when adding new Metal routes.

Compiler-owned Metal payloads currently cover generated MSL units such as
`ShapeOf`, `Range`, `Tile`, `Concat`, `Split`, `Slice`, activation, elementwise,
`Transpose`, numeric and logical reduction, Softmax/LogSoftmax, and causal SDPA
helper forms. MPS/MPSGraph vendor descriptor payloads are consumed by the
`MpsrtVendorPrimitive` runtime stage for supported MatMul/GEMM, last-axis
`Softmax`, Pool2D, Resize2D, and SDPA forms. Vendor selection remains
contract-limited: if the descriptor, storage, or external-buffer ABI cannot be
built, the route must be rejected or use another supported current route.
Pooling on Metal requires a valid MPS-family vendor route; the removed generic
MSL Pool2D fallback must not be reintroduced as an unvalidated path.

### OpenCL

The OpenCL backend is the current non-Apple route. It loads OpenCL dynamically
at runtime and executes source artifacts described by
`src/kernel_ir/gfx_opencl_source_artifacts.*`.

OpenCL source payload materialization is owned by
`src/backends/opencl/compiler/opencl_kernel_artifacts.*` and registered through
the backend module. The common executable-bundle builder records artifact
descriptors and requires the selected backend to provide payloads for executable
OpenCL source routes.

OpenCL artifacts are role-based. Source id, entry point, local size, tensor
roles, scalar ABI, runtime-shape scalars, constant materialization, boolean
buffer padding, static u32/f32 scalar payloads, and generated Concat/Split
chunk helpers should stay in the artifact manifest rather than being
reimplemented in infer-request code.
Routes that need runtime shape arguments must set
`KernelUnit::requires_runtime_shape_args`; infer requests consume the resulting
manifest/runtime-descriptor flag instead of special-casing the OpenCL backend.
Backend operation support and kernel-unit registration live under
`src/backends/opencl/compiler/`; runtime loading and enqueue stay under
`src/backends/opencl/runtime/`.

Embedded OpenCL source units live under `src/kernel_ir/opencl_kernels/`.
Current generated units include activation, elementwise, f32 MatMul, f32/f16
Interpolate, f32 reduction, boolean reduction, f32/f16 Softmax,
dynamic-static-rank f32/f16 Softmax, f32/f16 Pool2D, ShapeOf, Tile, Transpose,
logical-bool elementwise, compare/select, and generated Concat/Split helpers.
There is no active handwritten OpenCL kernel-unit exception in the current
registry.
The OpenCL compiler registry requires an explicit kernel unit for generated
routes; there is no generic MLIR fallback for OpenCL operation support.
Unsupported modes, axes, padding, shapes, or element types fail during support
probing instead of falling through to a hidden runtime path.

Generated activation artifacts cover the shared unary activation family and
carry op-specific scalar payloads in the manifest. `Swish` supports the default
beta, a scalar constant beta, and a runtime scalar beta tensor through the
`opencl/generated/activation_runtime_beta_*` units when shape and type contracts
match.

On Linux/Raspberry-style deployments, the build can stage a plugin-local OpenCL
bundle from `third_party/clvk` and `third_party/clspv`. The runtime loader checks
plugin-local `opencl/`, `clvk/`, and plugin directories before system/vendor
OpenCL libraries and configures CLVK tool paths when bundled tools are present.

## MLIR Role

MLIR is shared infrastructure, not a separate backend object. It is used for:

- support probing through `mlir_supports_node()`
- node lowering through `mlir_builder_*.cpp`
- pass and cleanup utilities
- backend hooks consumed by Metal and OpenCL compiler modules
- typed MPSRT dialect/metadata helpers shared with the Metal compiler path

Backend-specific source planning is no longer described as generic MLIR
ownership. Apple MSL/MPS source planning and typed MPSRT builder-plan assembly
live under `src/backends/metal/compiler/`; OpenCL source-artifact payload
materialization lives under `src/backends/opencl/compiler/`. Runtime-value
payload planning for dynamic shapes and source artifacts is shared runtime code
under `src/runtime/gfx_stage_runtime_values.*`.

Activation lowering keeps OpenCL and Metal source plans on the same operation
contract. `Swish` beta is represented either as a static scalar payload or as a
second scalar tensor input when the runtime-beta path is supported.

Reduction lowering uses the same source-plan boundary. Numeric reductions cover
the current f32 generated-kernel contract; logical reductions cover the current
boolean contract. Both require static input/output shapes and constant axes.

Softmax lowering is family-owned for `Softmax` and `LogSoftmax` on Metal. The
generated Metal units cover f32/f16 static-shape Softmax and LogSoftmax with
runtime-parameter binding. OpenCL currently covers `Softmax` f32/f16 static
shapes and dynamic-output shapes with static rank; OpenCL `LogSoftmax` is not a
current source-artifact route.

Pooling lowering is family-owned for `MaxPool` and `AvgPool`. OpenCL generated
Pool2D units cover f32/f16 static 4D NCHW input/output shapes with 2D kernel,
stride, dilation, and padding metadata. Metal Pool2D uses the MPS vendor route
only when the descriptor and external-buffer ABI are valid.

When adding or changing an op, keep support probing, lowering, backend source
planning, runtime binding, and tests on the same contract.

The current compiler service does not serialize a native backend executable
cache. It constructs an in-memory manifest/executable bundle for the selected
backend and the runtime consumes that descriptor during stage creation.
Compiler-owned tensor layout classification lives in
`src/compiler/tensor_layout.*`; shared stage policy consumes the resulting
layout contract instead of reclassifying view-only or materialized layout ops.

## Supported Operation Families

Coverage is shape- and backend-dependent. The active code contains support for:

- Conv2D, Conv3D, GroupConv, MatMul
- Add, Sub, Mul, Div, Pow, Mod, FloorMod
- compare, logical, Select, unary activation, and elementwise transforms
- RMS, RoPE, Softmax, reductions, BatchNormInference
- MaxPool, AvgPool, Interpolate
- Concat, Split, VariadicSplit, Slice, StridedSlice-style paths, Transpose,
  Reshape, Convert, ShapeOf, Range, Tile
- Gather, GatherND, GatherElements, ScatterUpdate, ScatterElementsUpdate,
  ScatterNDUpdate, TopK, SpaceToDepth, DepthToSpace
- stateful graph behavior through `ReadValue` / `Assign`
- selected Metal attention paths, including compiler-owned MPSGraph-backed
  vendor SDPA artifacts and fused causal-mask attention forms

Important constraints:

- Many paths require static rank or static shape.
- Many ops require constant attributes.
- Backend parity is not automatic between Metal and OpenCL.
- Unsupported shapes or types should fail clearly instead of falling back.

## Build

Configure against an OpenVINO Developer Package:

```bash
cmake -S . -B build-gfx-plugin -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINODeveloperPackage_DIR=/path/to/openvino/install/cmake \
  -DENABLE_TESTS=ON \
  -DGFX_DEFAULT_BACKEND=auto
```

Build the plugin and common test tools:

```bash
cmake --build build-gfx-plugin --target openvino_gfx_plugin
cmake --build build-gfx-plugin --target ov_gfx_func_tests ov_gfx_unit_tests ov_gfx_runtime_micro_tests
cmake --build build-gfx-plugin --target ov_gfx_compare_runner ov_gfx_microbench
```

Useful options:

- `-DGFX_ENABLE_METAL=ON|OFF`
- `-DGFX_ENABLE_OPENCL=ON|OFF`
- `-DGFX_ENABLE_RASPBERRY_OPENCL_TOOLCHAIN=ON|OFF`
- `-DGFX_RASPBERRY_OPENCL_BUNDLE_DIR=<path>`
- `-DGFX_DEFAULT_BACKEND=auto|metal|opencl`
- `-DENABLE_TESTS=ON|OFF`

`GFX_DEFAULT_BACKEND=auto` must resolve to an available backend. Explicit
`metal` or `opencl` requests are configure-time requirements and fail when that
backend is unavailable in the current build.

## Test

Run the module test label:

```bash
ctest --test-dir build-gfx-plugin --output-on-failure -L GFX
```

Run focused tests directly:

```bash
find build-gfx-plugin -name ov_gfx_unit_tests -type f
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_unit_tests> --gtest_filter=GfxStagePolicy.*
```

Use `ov_gfx_compare_runner` for accuracy triage and `ov_gfx_microbench` for
profiling/microbench triage. Do not use microbench output as correctness
evidence, and do not use compare-runner timing as performance evidence.
The test CMake flow also uses `tests/tools/gfx_gtest_source_contract.py` and
`tests/tools/gfx_gtest_matrix.py` to guard native/unavailable-adapter source
parity, duplicate registrations, and forbidden `DISABLED_` registrations.

## Adding A New Operation

1. Add or update support probing in `src/mlir/`.
2. Update the backend operation support policy and route selection under
   `src/compiler/` or `src/backends/*/compiler/` when the supported route
   changes.
3. Add MLIR lowering or a transform in the appropriate `src/mlir/` or
   `src/transforms/` file.
4. Choose the shared runtime contract first: stage policy, kernel manifest,
   compiler tensor-layout plan, compiler pipeline-stage builder/fusion plan,
   compiler pipeline-stage I/O plan, runtime stage materializer, compiler
   memory plan, fused-output lifetime plan, compiler-owned runtime executable
   descriptor builder, runtime stage plan, runtime-value payloads, OpenCL
   artifact metadata, or Metal MPSRT/Apple MSL source planning under the
   backend compiler directory.
5. Add backend-specific code only under `src/backends/metal/` or
   `src/backends/opencl/` when the shared path cannot express the behavior.
6. Add focused unit tests first, then backend or functional coverage when the
   behavior is externally visible.
7. Update this README and the relevant file under `docs/` when supported
   shapes, public properties, runtime routing, profiling, or test workflows
   change.

## Development Rules

- Keep shared behavior in `src/plugin/`, `src/runtime/`, `src/kernel_ir/`, or
  `src/mlir/`; keep compiler contracts in `src/compiler/`; keep backend-only
  code under the matching backend directory.
- Do not add CPU fallback for unsupported GPU stages.
- Do not add runtime switches to preserve obsolete execution routes.
- Do not duplicate OpenCL artifact ABI rules in request code.
- Do not bypass backend stage-placement policies through one-off MPS/MSL/OpenCL
  selection logic.
- Do not move pipeline descriptor construction or fusion selection back into
  `CompiledModel` or backend request code; keep descriptor construction in
  `src/compiler/pipeline_stage_builder.*` and fusion selection in
  `src/compiler/pipeline_stage_fusion.*`.
- Do not reintroduce the deleted standalone
  `src/backends/metal/runtime/mps_graph_attention_stage.*`; MPSGraph attention
  is represented by compiler-owned vendor artifacts and MPSRT vendor primitive
  execution.
- Do not bypass `GfxCompilerService`, `ManifestBundle`, or
  `ExecutableBundle` with ad-hoc backend support checks.
- Do not reintroduce `BackendLowering`, `metal_lowering`, or source-signature
  scanning fallbacks. Generated/prebuilt kernels must carry compiler-owned
  manifest ABI metadata.
- Do not modify `third_party/llvm-project/` unless the task explicitly requires
  vendored LLVM/MLIR changes. Keep `third_party/clvk` and `third_party/clspv`
  as reviewed submodule gitlinks, not copied source trees.
- Keep local artifacts, build trees, sensitive access material, machine names, and agent notes
  out of commits.

## Profiling

Enable profiling through standard OpenVINO profiling and GFX-specific report
properties:

- `ov::enable_profiling`
- `GFX_PROFILING_LEVEL`
- `GFX_PROFILING_REPORT`
- `GFX_MEM_STATS`

The profiling JSON includes compile and infer sections, target-profile counters
such as `target_backend_metal` and `target_backend_opencl`, stage estimates such
as bytes moved and MAC/FLOP counts, and backend-specific Metal/OpenCL segments
when available. Backend trace exporters are registered through the shared trace
sink registry; the Metal backend currently registers `signpost` and
`os_signpost`. See `docs/PROFILING_RUNBOOK.md` and
`docs/MICROBENCH_SCHEMA.md`.
