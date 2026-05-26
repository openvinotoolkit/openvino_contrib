# GFX Plugin Architecture

This document describes the architecture implemented in `modules/gfx_plugin`.
The live source tree is the authority. Do not treat older removed-backend design
notes or deleted backend code as current behavior.

## Goals

- Expose a single OpenVINO device named `GFX`.
- Share graph, MLIR, scheduling, memory, profiling, and binding logic across
  backends where possible.
- Keep backend-specific implementation under `src/backends/metal/` and
  `src/backends/opencl/`.
- Fail unsupported models during query or compilation instead of hiding gaps
  behind CPU fallback.

## Top-Level Structure

- `src/plugin/`: OpenVINO runtime integration, properties, backend resolution,
  compiled-model state, infer-request helpers, output resolution, model
  serialization, and stateful graph handling
- `src/runtime/`: backend-neutral stage interfaces, execution dispatcher,
  submission windows, target profiles, profiling reports, remote tensor/context
  helpers, common buffer abstractions, MPSRT model records, and reusable caches
- `src/kernel_ir/`: kernel manifests, custom-kernel family registry, dispatch
  metadata, argument helpers, cache keys, and OpenCL source artifacts
- `src/mlir/`: support probing, MLIR builders, pass helpers, runtime-value
  planning, Apple MSL/MPS source planning, and typed MPSRT materialization
- `src/backends/metal/`: Metal plugin glue, Objective-C++ runtime, MSL compiler,
  memory allocator, MPSRT request encoding, MPS/MPSGraph stages, and profiling
- `src/backends/opencl/`: OpenCL plugin glue, dynamic API loader, buffer manager,
  program cache, memory ops, and generic source-stage executor
- `src/transforms/`: graph rewrites, fusions, and layout cleanup
- `tests/`: unit, functional, integration, backend, compare, and microbench
  coverage

## Main Runtime Objects

### `Plugin`

`src/plugin/plugin.cpp` owns the OpenVINO-facing entry points:

- property parsing and validation
- `GFX_BACKEND` resolution
- backend-aware transform pipeline selection
- `query_model()`
- `compile_model()`
- model import/export plumbing
- remote-context creation

The backend parser accepts only `metal` and `opencl`. `auto` is resolved at
configure time by `cmake/GfxBackendConfig.cmake`.

### `CompiledModel`

`src/plugin/compiled_model.cpp` owns the compiled graph:

- transformed OpenVINO model
- selected backend state from `create_backend_state()`
- `build_op_pipeline()` stage construction
- compiled-model properties
- profiling configuration and report ownership
- output-source and output-alias metadata for fused or view-style stages

The compiled pipeline is a sequence of `PipelineStageDesc` records wrapping
backend-specific `GpuStage` instances. One stage may represent one OpenVINO node,
a fused sequence, a stateful helper, a view-style alias, or a materialized
constant output.

### `InferRequest`

Infer-request logic is split between shared helpers and backend files:

- shared state and binding helpers in `src/plugin/infer_request_state.hpp`,
  `src/plugin/infer_pipeline.*`, `src/plugin/infer_submission.*`,
  `src/plugin/infer_io_utils.*`, and `src/plugin/stateful_execution.*`
- Metal request implementation under `src/backends/metal/plugin/`
- OpenCL request implementation under `src/backends/opencl/plugin/`

The shared infer path prepares input/output binding plans, manages reusable host
outputs, allocates or reuses backend buffers, handles `ReadValue` / `Assign`
state, groups stages into submission windows, and records profiling segments.

## Backend Selection

Build-time availability is configured by:

- `GFX_ENABLE_METAL`
- `GFX_ENABLE_OPENCL`
- `GFX_DEFAULT_BACKEND=auto|metal|opencl`

Runtime selection is requested with:

- `GFX_BACKEND=metal`
- `GFX_BACKEND=opencl`

On macOS, CMake disables OpenCL and uses Metal. On non-Apple builds, OpenCL is
the source-kernel backend when enabled and available. Backend support checks are
compiled into `plugin/gfx_backend_config.hpp` from
`src/plugin/gfx_backend_config.hpp.in`.

## Stage Pipeline

`src/runtime/gpu_stage.hpp` defines the backend-neutral `GpuStage` interface.
Backends register factories through `GpuStageFactory`, and
`src/runtime/execution_dispatcher.*` creates concrete stages for the selected
backend.

Important stage hooks:

- `set_input_transform()`: lets a stage consume absorbed-transpose metadata
  instead of requiring a separate materialized stage
- `submit_policy()`: communicates scheduling weight and isolation hints to the
  submission planner
- `describe_output_lifetimes()`: lets complex stages describe internal output
  lifetimes for workspace reuse

`src/runtime/fused_sequence_stage.*` combines compatible stages when fusion
rules allow one runtime path to cover several OpenVINO nodes.

## Planning And Scheduling

`src/runtime/gfx_stage_policy.*` selects:

- backend domain, such as `apple_mps`, `apple_msl`, or `opencl`
- storage kind, such as `buffer`, `image`, `matrix`, `ndarray`, or `alias`
- fusion policy for bias, activation, and batchnorm
- precision handling
- submit policy
- convolution family and route metadata

`src/runtime/gfx_parallelism.*` and `src/runtime/gfx_partitioning.*` derive
workgroup and partitioning decisions from `GpuExecutionDeviceInfo` reported by
the active buffer manager. Those helpers are shared; add device-family or
capability logic there instead of adding per-backend shortcuts.

`src/plugin/infer_submission.*` records stages into submission windows. The
planner considers stage count, recorded output bytes, MAC estimates,
producer-consumer dependencies, and boundary stages such as layout, split,
transpose, softmax, and attention.

## MLIR Pipeline

MLIR lives under `src/mlir/` and is shared infrastructure:

- `mlir_supports_node()` provides support probing for query and compilation.
- `mlir_builder_*.cpp` files lower OpenVINO ops to MLIR forms.
- `gfx_mlir_transform_pipeline.*`, `mlir_passes.*`, and lowering helpers apply
  cleanup and backend-oriented conversion steps.
- `gfx_stage_runtime_values.*` owns runtime-value payload planning for shape,
  slice, split, range, tile, gather/scatter, and source-artifact metadata.
- `gfx_backend_custom_kernel_adapter.*` converts kernel manifests into backend
  source-binding plans.

Apple source planning is split by responsibility:

- `msl_codegen_apple_msl_*`: Apple MSL custom-kernel source plans
- `msl_codegen_apple_mps.*`: Apple MPS/MPSGraph vendor source plans
- `msl_codegen_matmul_*`: direct Metal and MPSRT MatMul planning
- `msl_codegen_attention.*`: attention and SDPA source planning
- `msl_codegen_compressed_matmul.*`: compressed MatMul helpers
- `gfx_apple_stage_pipeline.*` and `gfx_apple_vendor_descriptors.*`: typed
  Apple vendor descriptor extraction and materialization

## Kernel Manifests

`src/kernel_ir/gfx_kernel_manifest.hpp` and
`src/kernel_ir/gfx_custom_kernel_families.*` describe stage contracts that need
to survive from MLIR/source planning into runtime binding:

- execution kind: vendor primitive or custom kernel
- backend domain and storage family
- semantic input/output roles
- external-buffer ABI roles
- dispatch policy
- family id and required entry point

New custom-kernel paths should express their buffer order and scalar payloads
through these shared structures first. Backend runtime code should consume the
manifested contract instead of inferring argument layout from local conventions.

## Metal Architecture

The Metal backend is split into:

- plugin glue in `src/backends/metal/plugin/`
- runtime/memory/profiling in `src/backends/metal/runtime/`
- MSL compilation in `src/backends/metal/codegen/`
- MPSRT preparation and request encoding in
  `src/backends/metal/runtime/mpsrt/`

Metal stages can use Apple MPS/MPSGraph vendor primitives or Apple MSL custom
kernels. The shared stage policy selects placement and storage. The compile path
can materialize a typed MPSRT program and backend-neutral runtime model through
`src/runtime/gfx_mpsrt_model.*`, `gfx_mpsrt_plan.hpp`,
`gfx_mpsrt_program.hpp`, and `gfx_mpsrt_kernel_manifest_adapter.hpp`.

Request-time Metal execution validates external-buffer bindings, model-owned
resources, transient resources, storage bridges, prepared heaps, and
kernel-buffer order before encoding commands. MSL dispatch stages must not
depend on implicit positional conventions when a manifest supplies explicit
roles.

## OpenCL Architecture

The OpenCL backend is split into:

- plugin glue in `src/backends/opencl/plugin/`
- dynamic loader and device selection in `src/backends/opencl/runtime/opencl_api.*`
- buffer manager and memory ops in `src/backends/opencl/runtime/`
- program cache in `src/backends/opencl/runtime/opencl_program_cache.*`
- source-stage execution in `src/backends/opencl/runtime/opencl_source_stage.*`

Source kernels are described by `src/kernel_ir/gfx_opencl_source_artifacts.*`.
The source-stage executor is intentionally generic. Op-specific behavior should
reach it through artifact metadata, generated chunk artifacts, shared
runtime-value planners, constant materialization, and boolean-buffer contracts.

Current OpenCL source artifacts cover data movement, selected converts, MatMul,
Softmax, Range, Tile, gather/scatter families, ShapeOf, Concat/Split, unary and
binary elementwise families, compare/select, and boolean logical/reduction
families when shapes and element types match their contracts.

## Stateful And Reusable Inference

Stateful behavior is not treated as ordinary stateless data movement:

- `ReadValue` can read from infer-request variable state.
- `Assign` updates the request-owned variable buffer.
- Source-node-aware output links keep stateful consumers connected to the
  original graph output when stages are fused or rewritten.

Reusable inference layers include:

- prepared input plans
- prepared output plans
- reusable host outputs for static output signatures
- immutable device-buffer caches for constants
- prepared backend binding caches
- liveness-managed stage-output workspaces
- view-style aliases for compatible Split/VariadicSplit outputs

## Profiling Architecture

Profiling is assembled in shared code and enriched by backend runtimes:

- compile-time report sections
- infer-time node, segment, transfer, allocation, and counter records
- stage estimates such as `bytes_in`, `bytes_out`, `macs_est`, and `flops_est`
- target profile under `extended.target_profile`
- backend counters such as `target_backend_metal` and `target_backend_opencl`

Use the target profile to confirm which route actually ran before comparing
Metal and OpenCL measurements.

## Extension Rules

- Prefer shared contracts in `src/runtime/`, `src/kernel_ir/`, and `src/mlir/`.
- Add backend-specific code only when the shared contract reaches a real backend
  boundary.
- Keep `query_model()`, compilation, runtime binding, and tests aligned.
- Keep OpenCL source ids, scalar ABI, constant materialization, chunk helpers,
  and boolean padding in `gfx_opencl_source_artifacts.*`.
- Keep Metal placement, MPSRT records, MSL source plans, and runtime binding on
  the manifest/MPSRT contract.
- Do not reintroduce removed backend routes, CPU fallback, runtime defines, or
  stale architecture notes.
