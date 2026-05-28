# OpenVINO GFX Plugin

`GFX` is an experimental OpenVINO device plugin that compiles an `ov::Model`
into a stage-based GPU execution pipeline. The module lives as a standalone
OpenVINO contrib module and builds against an OpenVINO Developer Package.

The current implementation has two backend families:

- `metal`: macOS execution through Metal, Apple MPS/MPSGraph vendor stages,
  MPSRT runtime-model records, and generated MSL custom kernels.
- `opencl`: non-Apple execution through a dynamically loaded OpenCL runtime and
  manifest-backed source-kernel artifacts.

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
- OpenCL is disabled by CMake on macOS; the Apple route is Metal/MPS/MPSRT/MSL.
- On non-Apple builds, `auto` resolves to OpenCL when the source backend is
  enabled in the build.
- `query_model()` uses the same backend-aware support path as compilation.
- `export_model()` serializes the OpenVINO model, not a backend binary cache.
- `GfxRemoteContext` and `GfxRemoteTensor` exist; practical capabilities depend
  on the selected backend.

## Source Layout

- `include/openvino/gfx_plugin/`: public plugin headers and property names
- `src/compiler/`: backend target registry, operation support policies,
  lowering-plan construction, manifest building, executable-bundle assembly, and
  artifact payload routing used by query and compilation
- `src/plugin/`: OpenVINO-facing `Plugin`, `CompiledModel`, infer-request
  plumbing, properties, model serialization, backend selection, and stateful
  `ReadValue` / `Assign` handling
- `src/runtime/`: backend-neutral stage interfaces, submission planning,
  profiling report assembly, liveness-aware output workspaces, remote
  context/tensor helpers, descriptor-backed view-only stages, target profiles,
  and shared MPSRT runtime-model types
- `src/kernel_ir/`: backend-neutral kernel manifests, custom-kernel family
  metadata, cache keys, dispatch descriptions, embedded Metal/OpenCL helper
  sources, and OpenCL source artifacts
- `src/mlir/`: support probing, OpenVINO-node lowering, pass helpers, Apple
  stage-pipeline materialization, MPSRT dialect/program helpers, Apple MSL/MPS
  source-plan helpers, and shared runtime-value planning
- `src/backends/metal/`: Metal plugin glue, Objective-C++ runtime, memory
  management, profiling, backend compiler policy, MSL compilation, runtime
  kernel loading, MPSRT preparation, descriptor-backed vendor primitive stages,
  and MPSGraph stages
- `src/backends/opencl/`: OpenCL plugin glue, dynamic API loader, buffer
  manager, backend compiler policy, program cache, memory ops, runtime kernel
  loading, and generic source-artifact execution
- `src/transforms/`: OpenVINO graph rewrites, fusion passes, and layout cleanup
- `tests/`: unit, functional, backend, integration, compare, and microbench
  coverage
- `bench/`: optional local/remote evaluation helpers
- `tools/`: profiling and microbench post-processing helpers
- `third_party/llvm-project/`: vendored LLVM/MLIR used by the build

## Compilation And Execution Pipeline

The high-level path is:

1. `Plugin::compile_model()` parses properties and resolves `GFX_BACKEND`.
2. `src/compiler/BackendRegistry` resolves the compiled backend module and
   immutable `BackendTarget`.
3. `GfxCompilerService` runs backend-aware transforms, operation support
   checks, lowering-plan creation, manifest building, and executable-bundle
   assembly.
4. `CompiledModel` validates the executable bundle and builds a
   `RuntimeExecutableDescriptor` that is passed into backend stage creation.
5. `CompiledModel::build_op_pipeline()` creates a sequence of stage descriptors.
6. `src/runtime/gfx_stage_policy.*` selects placement, storage, fusion, and
   submit policy from node type, shape, element type, backend, and device caps.
7. `src/mlir/` lowers supported nodes and materializes backend source plans.
8. The active backend creates descriptor-backed view-only stages or concrete
   backend `GpuStage` objects through `ExecutionDispatcher`.
9. Infer requests bind host or remote tensors, allocate or reuse backend
   buffers, execute submission windows, and collect profiling data.

The pipeline is intentionally stage-based. Fused stages, descriptor-backed
view-only stages, view-style split outputs, stateful variable buffers, reusable
host outputs, prepared binding caches, immutable const caches, and
workspace-managed intermediate outputs all live in the shared runtime/plugin
layer instead of being duplicated per backend.

The compiler layer is the public architecture boundary between OpenVINO graph
semantics and runtime execution. Do not add new support checks directly in
`Plugin` or backend request code when the same decision belongs in
`src/compiler/`.

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
- MPSRT runtime-model records under `src/runtime/gfx_mpsrt_*`
- embedded MPSRT helper kernels under `src/kernel_ir/metal_kernels/`
- Objective-C++ request-time execution under `src/backends/metal/runtime/`

Metal placement is selected by shared stage policy. Do not bypass it with
ad-hoc backend switches when adding new Metal routes.

Compiler-owned Metal payloads currently cover generated MSL units such as
`ShapeOf`, `Range`, `Tile`, `Concat`, `Split`, `Slice`, activation, elementwise,
numeric and logical reduction, and causal SDPA helper forms. MPS/MPSGraph vendor
descriptor payloads are consumed by the `MpsrtVendorPrimitive` runtime stage for
supported MatMul/GEMM, `Softmax`, Pool2D, Resize2D, and SDPA forms. Vendor
selection remains contract-limited: if the descriptor, storage, or
external-buffer ABI cannot be built, the route must be rejected or use another
supported current route.

### OpenCL

The OpenCL backend is the current non-Apple route. It loads OpenCL dynamically
at runtime and executes source artifacts described by
`src/kernel_ir/gfx_opencl_source_artifacts.*`.

OpenCL artifacts are role-based. Source id, entry point, local size, tensor
roles, scalar ABI, runtime-shape scalars, constant materialization, boolean
buffer padding, static u32/f32 scalar payloads, and generated Concat/Split
chunk helpers should stay in the artifact manifest rather than being
reimplemented in infer-request code.
Backend operation support and kernel-unit registration live under
`src/backends/opencl/compiler/`; runtime loading and enqueue stay under
`src/backends/opencl/runtime/`.

Embedded OpenCL source units live under `src/kernel_ir/opencl_kernels/`.
Current named units include generated activation, elementwise, f32 MatMul, and
f32/f16 Interpolate helpers, generated f32 reduction helpers, plus f32/f16
Softmax and logical-bool reduction baseline helpers. The OpenCL compiler
registry requires an explicit kernel unit for generated routes; there is no
generic MLIR fallback for OpenCL operation support. Unsupported modes, axes,
padding, shapes, or element types fail during support probing instead of
falling through to a hidden runtime path.

Generated activation artifacts cover the shared unary activation family and
carry op-specific scalar payloads in the manifest. `Swish` supports the default
beta, a scalar constant beta, and a runtime scalar beta tensor through the
`opencl/generated/activation_runtime_beta_*` units when shape and type contracts
match.

## MLIR Role

MLIR is shared infrastructure, not a separate backend object. It is used for:

- support probing through `mlir_supports_node()`
- node lowering through `mlir_builder_*.cpp`
- pass and cleanup utilities
- Apple MSL/MPS source planning
- typed MPSRT program materialization
- runtime-value payload planning for dynamic shapes and source artifacts

Activation lowering keeps OpenCL and Metal source plans on the same operation
contract. `Swish` beta is represented either as a static scalar payload or as a
second scalar tensor input when the runtime-beta path is supported.

Reduction lowering uses the same source-plan boundary. Numeric reductions cover
the current f32 generated-kernel contract; logical reductions cover the current
boolean contract. Both require static input/output shapes and constant axes.

When adding or changing an op, keep support probing, lowering, backend source
planning, runtime binding, and tests on the same contract.

The current compiler service does not serialize a native backend executable
cache. It constructs an in-memory manifest/executable bundle for the selected
backend and the runtime consumes that descriptor during stage creation.

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
- selected Metal attention paths, including MPSGraph-backed SDPA and fused
  causal-mask attention forms

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
- `-DGFX_DEFAULT_BACKEND=auto|metal|opencl`
- `-DENABLE_TESTS=ON|OFF`

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

## Adding A New Operation

1. Add or update support probing in `src/mlir/`.
2. Update the backend operation support policy and route selection under
   `src/compiler/` or `src/backends/*/compiler/` when the supported route
   changes.
3. Add MLIR lowering or a transform in the appropriate `src/mlir/` or
   `src/transforms/` file.
4. Choose the shared runtime contract first: stage policy, kernel manifest,
   runtime-value payloads, OpenCL artifact metadata, or Metal MPSRT/Apple MSL
   source planning.
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
- Do not bypass Metal placement through one-off MPS/MSL selection logic.
- Do not bypass `GfxCompilerService`, `ManifestBundle`, or
  `ExecutableBundle` with ad-hoc backend support checks.
- Do not modify `third_party/llvm-project/` unless the task explicitly requires
  vendored LLVM/MLIR changes.
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
when available. See `docs/PROFILING_RUNBOOK.md` and
`docs/MICROBENCH_SCHEMA.md`.
