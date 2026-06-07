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
  compiled-model state, infer-request OpenVINO API glue, output resolution,
  disabled compiled-model cache/import/export entry points, backend-access
  helpers, and variable-state API objects
- `src/common/`: backend ids, configured-backend availability values,
  artifact-payload interfaces, activation/bias metadata, dispatch and
  parallelism value types, submit-policy records, constant-evaluation helpers,
  and GPU device-family/profile value types shared across compiler, runtime,
  and backend code
- `src/compiler/`: backend target description, backend module registry,
  backend capability records, operation support policies, backend
  stage-placement contracts, operation legalization, tensor-layout
  classification, stage compiler policy, lowering plans, stage policy,
  pipeline-stage descriptor building, fusion selection, compiler-side
  pipeline-stage I/O plans, memory plans, manifest bundles, cache envelopes,
  executable bundles, runtime executable descriptor build/verification,
  backend-module construction, and artifact payload routing
- `src/runtime/`: backend-neutral stage interfaces, execution dispatcher,
  backend runtime/provider interfaces, backend request state, backend stage
  factory interfaces, infer pipeline/executor/submission helpers,
  pipeline-stage descriptors and materializers, descriptor-backed view-only
  stages, stateful runtime helpers, submission windows, target profiles,
  profiling reports, remote tensor/context helpers, common buffer abstractions,
  runtime executable descriptor records, runtime pipeline-stage plans, runtime
  sessions, fused-output lifetime planning, descriptor-owned const-tensor
  materialization, runtime-value and kernel-launch planning, and reusable caches
- `src/kernel_ir/`: kernel manifests, custom-kernel family registry, dispatch
  metadata, argument helpers, cache keys, embedded helper kernel sources, and
  OpenCL source artifacts
- `src/mlir/`: support probing, MLIR builders, pass helpers, backend hooks,
  MPSRT dialect/metadata helpers, and generic MLIR kernel construction
- `src/backends/metal/`: Metal plugin glue, backend compiler module, Apple
  MSL/MPS/MPSRT source planning, shared MPSRT ABI records and vendor primitive
  contracts, Objective-C++ runtime, MSL compiler, memory allocator, MPSRT
  request encoding, MPS/MPSGraph vendor primitive execution, and profiling
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
- import/export API entry points that currently reject round-trip cache use
- remote-context creation

The backend parser accepts only `metal` and `opencl`. `auto` is resolved at
configure time by `cmake/GfxBackendConfig.cmake`.

`compile_model()` delegates support probing, transformation, lowering,
manifest construction, and executable-bundle construction to
`compiler::GfxCompilerService`. `query_model()` uses the same compiler service
and keeps support probing aligned with compilation. Plugin code should not grow
parallel support tables.

### `CompiledModel`

`src/plugin/compiled_model.cpp` owns the compiled graph:

- transformed OpenVINO model
- selected backend state from `create_backend_state()`
- runtime executable descriptor produced by
  `src/compiler/runtime_executable_descriptor_builder.*` from the compiler
  executable bundle
- runtime descriptor ownership exposed to infer-request construction
- `build_op_pipeline()` delegation to the compiler stage builder
- compiled-model properties
- profiling configuration and report ownership

The compiled pipeline is a sequence of `PipelineStageDesc` records wrapping
backend-specific `GpuStage` instances. One stage may represent one OpenVINO node,
a fused sequence, a stateful helper, a view-style alias, or a materialized
constant output.

`build_op_pipeline()` consumes the compiler-owned `PipelineStageRuntimePlan`
alongside the runtime executable descriptor and delegates concrete stage
materialization to `src/runtime/pipeline_stage_materializer.*`. The runtime
plan is emitted by `src/compiler/pipeline_stage_builder.*` as a separate
compiler output, not as part of the cacheable runtime executable descriptor.
The builder owns stage ordering, node-to-stage mapping, parameter index map,
absorbed input transforms, model-output alias handling, and the handoff from
compiler-owned planning contracts in
`src/compiler/pipeline_stage_plan.*` and
`src/compiler/pipeline_stage_fusion.*`. `CompiledModel` should not accumulate
backend-specific stage construction logic.

`src/runtime/pipeline_stage_materializer.*` passes the matching
`RuntimeStageExecutableDescriptor` to backend stage creation through
`BackendStageFactory`. Backends may consume that descriptor for payload kind,
entry point, ABI fingerprint, explicit roles, and artifact payload. If a
descriptor is missing or invalid, the backend must fail or fall back only to a
supported current path, never to a removed route.
`src/runtime/tensor_binding_contract.*` is the shared parser for descriptor
element-type and static-shape contracts and the shared classifier for generated
source `RuntimeParams` payloads that can be materialized without an OpenVINO
source node. The current descriptor-owned families are Broadcast, binary
elementwise broadcast, Select, Tile, Softmax/LogSoftmax, Transpose, and Reduce
when the descriptor carries the required static tensor contracts and runtime
metadata.

Kernel preparation is request-local and is mediated by
`src/runtime/runtime_session.*` so descriptor-owned resource bindings and
memory-region ids are checked against the current tensor bindings.

### `InferRequest`

Infer-request logic is split between shared helpers and backend files:

- OpenVINO-facing request state and IO helpers in
  `src/plugin/infer_request_state.hpp`, `src/plugin/infer_io_utils.*`, and
  `src/plugin/infer_request_variable_state.*`; host input binding refreshes
  from the current OpenVINO request tensor on every infer, so public
  port/node-based `set_tensor()` entry points cannot leave stale request-local
  cached views behind. Public output tensors obtained before `infer()` are
  pinned as request-owned host outputs when their type and shape match the
  model output contract; the shared output binder then writes into that same
  tensor instead of replacing it with a backend-specific view.
- shared runtime execution helpers in `src/runtime/infer_pipeline.*`,
  `src/runtime/infer_submission.*`, `src/runtime/infer_executor.*`, and
  `src/runtime/stateful_execution.*`
- request-local stateful tensor slots in
  `src/runtime/stateful_variable_state.hpp`; plugin request state owns the map,
  but runtime helpers consume only that runtime-level state, not
  `InferRequestState`. Initial OpenVINO variable state is materialized as a
  zero-filled host tensor and uploaded on the first `ReadValue` through the same
  runtime state map used by `set_state()` / `reset()`. `ReadValue` materializes
  a compiler-planned output snapshot from the persistent GPU state buffer, so
  downstream consumers observe the pre-`Assign` value even when the state is
  updated later in the same graph. `Assign` updates the persistent GPU state
  buffer and marks the request-local host mirror stale;
  `query_state()` readback is then handled by the shared runtime helper through
  the same backend memory-copy API used for normal output readback. Direct
  producer-to-state aliasing for `Assign` is compiler-planned only when that
  producer output has no non-`Assign` consumers, so state update storage cannot
  overwrite a value still needed by downstream stages.
- Metal request implementation under `src/backends/metal/plugin/`
- OpenCL request implementation under `src/backends/opencl/plugin/`

The shared infer path prepares input/output binding plans, manages reusable host
outputs, allocates or reuses backend buffers, handles `ReadValue` / `Assign`
state, creates `RuntimeSession` binding tables for descriptor-backed stages,
prepares kernel executables, groups stages into submission windows, and records
profiling segments.

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
compiled into `common/backend_config.hpp` from
`src/common/backend_config.hpp.in`; `compiler/backend_config.hpp` is a wrapper
include for existing compiler-path users. Explicit `GFX_DEFAULT_BACKEND=metal`
or `GFX_DEFAULT_BACKEND=opencl` is a hard configure-time request: CMake fails if
the requested backend is unavailable instead of silently falling back to a
different backend.

The compiler registry and runtime availability share the generated availability
contract. The default `BackendRegistry` contains only the production backend
modules available in the current build. `src/compiler/backend_module_provider.*`
assembles the configured backend modules, including
`src/backends/metal/compiler/metal_backend_module.*` and
`src/backends/opencl/compiler/opencl_backend_module.*`. Runtime availability is
provided separately through `src/runtime/backend_runtime_provider.*`, where each
backend registers callbacks for backend-state creation, infer execution, and
profiling trace-sink registration. Query, lowering, manifest, runtime-provider,
and architecture contract tests reason about that configured backend set.

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

`src/runtime/fused_sequence_stage.*` combines compatible stages when fusion
rules allow one runtime path to cover several OpenVINO nodes. Internal output
lifetimes for such stages are carried by `PipelineStageDesc` records in
`src/runtime/pipeline_stage_desc.hpp` and consumed by the shared inference
pipeline; runtime stages must not reclassify tensor views or lifetimes from
operation type names.
`src/runtime/fused_output_lifetime_plan.*` computes that metadata from
`RuntimeStageExecutableDescriptor` and `RuntimeMemoryPlanDescriptor` contracts.

`src/runtime/backend_stage_factory.hpp` is the runtime-facing backend boundary.
`BackendState` implements that interface, while
`src/runtime/pipeline_stage_materializer.*` owns descriptor lookup,
stage-index validation, descriptor-backed backend stage creation, vendor
primitive artifact materialization, and fused sequence materialization. Do not
recreate these decisions inside backend request code.

`src/runtime/view_only_stage.*` handles compiler-owned metadata descriptors for
ops whose outputs can alias an input buffer without launching a backend kernel.
The current use is limited to view-compatible stage-policy contracts such as
compatible Split/VariadicSplit aliases. It must not become a hidden fallback for
ops that need real computation.

Concrete stages receive `GpuStageRuntimeOptions` from `CompiledModel`. These
options carry the selected backend `StagePlacementPolicy` and
`PostOpFusionCapabilities` as `compiler::StageCompilerPolicy` inputs to shared
stage policy. The policy also carries source-kernel dispatch fallback
parallelism for OpenCL-style source routes. Runtime stages and MLIR stages
should consume that explicit policy instead of resolving
`BackendRegistry::default_registry()` themselves.

## Planning And Scheduling

`src/compiler/stage_policy.*` selects:

- fusion policy for bias, activation, and batchnorm
- precision handling
- submit policy
- convolution family and route metadata

Backend domain and storage placement are selected through
`compiler::StagePlacementPolicy`. The shared value types live in
`src/compiler/stage_placement.*`; backend-specific placement decisions live in
`src/backends/metal/compiler/metal_stage_placement.*` and
`src/backends/opencl/compiler/opencl_stage_placement.*`. `CompiledModel` builds
`compiler::StageCompilerPolicy` from the selected backend capabilities and
passes it to stage creation, MPS/MSL planning helpers, and shared stage-policy
calls. Shared stage policy must use that explicit policy instead of carrying all
backend placement rules inline or resolving a registry on its own.

`src/runtime/gfx_parallelism.*` and `src/runtime/gfx_partitioning.*` derive
workgroup and partitioning decisions from `GpuExecutionDeviceInfo` reported by
the active buffer manager. Those helpers are shared; add device-family or
capability logic there instead of adding per-backend shortcuts.

`src/runtime/infer_submission.*` records stages into submission windows. The
planner considers stage count, recorded output bytes, MAC estimates,
producer-consumer dependencies, and boundary stages such as layout, split,
transpose, softmax, and attention.

## Compiler Service

`src/compiler/` is the current boundary between OpenVINO graph semantics and
runtime execution. The main objects are:

- `BackendTarget`: immutable backend id, runtime API, feature bits, and cache
  compatibility fingerprint for one selected backend.
- `BackendRegistry`: compiled-backend registry. Its default instance contains
  only the production backend modules available in the configured build, as
  reported by `compiler/backend_config.hpp`. Each module owns backend transform
  `PipelineOptions`, fusion capabilities, post-op fusion capabilities,
  stage-placement policy, artifact payload resolver, and backend-owned vendor
  attention artifact resolver when the backend supports that route.
- `OperationSupportPolicy` and `OperationLegalizer`: backend-aware operation
  legality and route selection.
- `LoweringPlanner`: converts a transformed model into ordered
  `PlannedOperation` records with a `KernelUnit` route and compiler-owned
  tensor-layout plan.
- `MemoryPlanBuilder`: assigns logical tensor memory regions, alias groups,
  lifetimes, and transient arenas. Hidden host copies are rejected by contract.
- `ManifestBuilder`: emits a `ManifestBundle` with stage records, tensor
  contracts, memory-region ids, runtime parameter contracts, dispatch contract,
  and memory contract.
- `ExecutableBundleBuilder`: converts the manifest and lowering plan into
  executable stage records, artifact descriptors, ABI fingerprints, and optional
  artifact payloads.
- `CacheEnvelopeBuilder`: builds a cache envelope with model, manifest,
  backend-capability, compile-option, kernel-unit, artifact-descriptor, and
  backend-payload identity fingerprints. The envelope has a deterministic
  wire/store/load contract used for validation, not for public model import.
- `PipelineStageBuilder`: converts the transformed model plus runtime
  executable descriptor into `PipelineStageDesc` records using the selected
  backend policy, fusion planner, stage materializer, and pipeline-stage I/O
  planner.
- `PipelineStageFusion`: selects post-op, residual-add, attention, and vendor
  attention fusion groups from compiler-owned fusion contracts instead of
  runtime stage type probes.

The compiler service currently builds an executable description and a serialized
cache-envelope record. It does not emit or load a native backend binary cache,
and the serialized envelope is not a public OpenVINO cache format.
`export_model()`, `import_model()`, and `ov::cache_dir` remain disabled through
`src/plugin/compiled_model_cache_contract.hpp` until a complete
executable-bundle import/export path and backend-payload materialization exist.

Backend kernel registries are explicit. Generated and vendor routes must name a
registered `KernelUnit`; generic catch-all ids are rejected by contract tests.
Backend stage creation is also descriptor-owned: every runtime stage that needs
a kernel, source artifact, vendor primitive, or ABI fingerprint consumes the
matching `RuntimeStageExecutableDescriptor`. The old pattern where a backend
stage inferred its executable payload from the OpenVINO node is not a current
route.
Source-node bridge routes are removed from the runtime descriptor contract.
Vendor primitive metadata, runtime-shape arguments, `ConstTensor` ABI, and
source `RuntimeParams` ABI must be descriptor-owned. If the compiler cannot
freeze the required data into the executable descriptor/artifact payload,
runtime descriptor verification must fail instead of allowing backend runtime
code to inspect the OpenVINO node.

`BackendTarget`, lowering plans, backend capability objects, buffers, and
runtime capability records default to `GpuBackend::Unknown` until a backend is
explicitly selected. Unknown backends cannot materialize a target, stage
factory, or memory ops. This prevents accidental Metal-default behavior in
shared code. `GpuBackend` and `GpuDeviceFamily` live in `src/common/` so shared
compiler and runtime code do not depend on plugin or backend headers for these
identity types.

`BackendLowering` and `metal_lowering` are removed legacy routes. A supported
operation must lower through common metadata, an explicit generated kernel unit,
a vendor primitive, or a consciously documented backend route. Metal codegen also
requires compiler-owned manifest ABI counts for generated/prebuilt MSL kernels;
source-signature scanning is not a supported ABI fallback.

## MLIR Pipeline

MLIR lives under `src/mlir/` and is shared infrastructure:

- `mlir_supports_node()` provides support probing for query and compilation.
- `mlir_builder_*.cpp` files lower OpenVINO ops to MLIR forms.
- `gfx_mlir_transform_pipeline.*`, `mlir_passes.*`, and lowering helpers apply
  cleanup and backend-oriented conversion steps.
- `src/runtime/gfx_stage_runtime_values.*` owns runtime-value payload planning
  for shape, slice, split, range, tile, gather/scatter, and source-artifact
  metadata.
- `gfx_backend_custom_kernel_adapter.*` converts kernel manifests into backend
  source-binding plans.

Backend custom-kernel adapters now require manifest-derived ABI metadata when a
generated/prebuilt backend kernel is used. Missing custom-kernel manifest ABI is
a compile-time error, not a request-time fallback to source scanning.

Unary activation lowering is shared across backend source plans. The current
contract supports static activation scalars such as `Elu` alpha and `Clamp`
range, and treats `Swish` beta as either a static scalar payload or a second
scalar tensor input when the generated runtime-beta path is selected.

Reduction lowering is also shared at the source-plan boundary. Numeric
reductions currently use f32 generated-kernel contracts; logical reductions use
boolean contracts. The supported source paths require static input/output shape
metadata and constant axes so the compiler can materialize scalar and
runtime-parameter bindings.

Softmax lowering is owned by the Softmax family. Metal has generated f32/f16
Softmax and LogSoftmax MSL units with runtime-parameter binding; MPS vendor
Softmax is only a descriptor-backed last-axis Softmax route. OpenCL has
generated f32/f16 Softmax units for static shapes and dynamic-output
static-rank shapes. OpenCL LogSoftmax is not currently implemented.

Pooling lowering is owned by the Pooling family. OpenCL generated Pool2D units
cover f32/f16 static 4D NCHW MaxPool/AvgPool contracts with 2D window metadata.
Metal Pool2D is intentionally restricted to descriptor-backed MPS-family vendor
routes; the old generic MSL Pool2D fallback is removed and must not be treated
as current behavior.

Apple source planning lives under `src/backends/metal/compiler/` and is split
by responsibility:

- `msl_codegen_apple_msl_*`: Apple MSL custom-kernel source plans
- `msl_codegen_apple_mps.*`: Apple MPS/MPSGraph vendor source plans
- `msl_codegen_matmul_*`: direct Metal and MPSRT MatMul planning
- `msl_codegen_attention.*`: attention and SDPA source planning
- `msl_codegen_compressed_matmul.*`: compressed MatMul helpers
- `apple_stage_pipeline.*` and `apple_vendor_descriptors.*`: typed Apple
  vendor descriptor extraction and materialization

## Kernel Manifests

Kernel contracts are split across the current compiler and kernel IR layers:

- `src/compiler/manifest.*`: normalized stage records, tensor contracts,
  memory-region ids, runtime-parameter contracts, dispatch contracts,
  runtime-shape-argument requirements, and memory contracts
- `src/compiler/pipeline_stage_plan.*`: compiler-side model-output flags,
  input links, and output-alias metadata used while planning compiled pipeline
  descriptors
- `src/compiler/pipeline_stage_builder.*`: stage descriptor construction and
  node-to-stage mapping from compiler-owned planning contracts
- `src/compiler/pipeline_stage_fusion.*`: compiler-owned fusion selection and
  vendor attention planning
- `src/compiler/memory_plan.*`: compiler-owned memory regions, lifetimes, alias
  groups, transient arenas, and memory-plan fingerprints
- `src/compiler/cache_envelope.*`: cache keys, wire-format serialization,
  store/load helpers, payload identity records, and stable-key validation;
  currently not exposed through `export_model()` or `import_model()`
- `src/compiler/executable_bundle.*`: artifact descriptors, ABI fingerprints,
  payload kind, manifest references, memory plan, and runtime-facing stage
  records
- `src/compiler/runtime_executable_descriptor_builder.*`: conversion and
  verification from compiler executable bundles into runtime executable
  descriptors; graph-owned runtime stage plans are emitted separately and are not
  embedded into cacheable descriptors
- `src/runtime/executable_descriptor.*`: backend-neutral runtime descriptor
  records consumed by stage factories and `RuntimeSession`
- `src/runtime/pipeline_stage_plan.hpp`: runtime-facing stage materialization
  plan emitted by the compiler builder and consumed by the materializer
- `src/runtime/backend_stage_factory.hpp`: backend-facing stage creation
  interface implemented by backend state
- `src/runtime/pipeline_stage_desc.hpp`: runtime pipeline descriptor record
  shared by compiled model, infer planning, and stateful helpers
- `src/runtime/pipeline_stage_materializer.*`: descriptor lookup, backend stage
  creation, vendor primitive artifact materialization, and fused sequence
  materialization
- `src/runtime/descriptor_const_tensor_materializer.*`: shared materialization
  of descriptor-owned `ConstTensor` payloads for Metal, OpenCL, and vendor
  primitive stages
- `src/runtime/stage_materialization_context.hpp`: the context object that hands
  compiler-owned `RuntimeStageExecutableDescriptor` records to backend stage
  factories without exposing OpenVINO source graph identity to runtime code
- `src/runtime/tensor_binding_contract.*`: descriptor-owned tensor
  element-type/static-shape parsing and generated `RuntimeParams` ownership
  classification shared by Metal and OpenCL runtime code
- `src/runtime/runtime_session.*`: request-local resource binding tables and
  prepared executable objects that validate tensor bindings against descriptor
  memory-region ids
- `src/runtime/fused_output_lifetime_plan.*`: fused-stage output lifetime and
  alias-storage planning based on runtime memory contracts
- `src/kernel_ir/gfx_kernel_manifest.hpp` and
  `src/kernel_ir/gfx_custom_kernel_families.*`: custom-kernel family metadata
  and source-plan contracts
- `src/kernel_ir/gfx_kernel_source.*`: embedded source payload wrapper for
  generated MSL/OpenCL helper sources

New custom-kernel paths should express their buffer order and scalar payloads
through these shared structures first. Backend runtime code should consume the
manifested contract instead of inferring argument layout from local conventions.
When an operation needs a source payload, the backend compiler policy should
materialize it through an artifact resolver; request-time code should load it
from the runtime descriptor.

## Metal Architecture

The Metal backend is split into:

- compiler policy and artifact materialization in `src/backends/metal/compiler/`
- plugin glue in `src/backends/metal/plugin/`
- runtime/memory/profiling in `src/backends/metal/runtime/`
- descriptor-backed vendor primitive execution in
  `src/backends/metal/runtime/mpsrt_vendor_primitive_stage.*`
- MSL compilation in `src/backends/metal/codegen/`
- shared MPSRT ABI, builder-plan, and vendor primitive contract records in
  `src/backends/metal/common/mpsrt/`
- MPSRT runtime-model construction, preparation, and request encoding in
  `src/backends/metal/runtime/mpsrt/`
- embedded helper kernels in `src/kernel_ir/metal_kernels/`

Metal stages can use Apple MPS/MPSGraph vendor primitives or Apple MSL custom
kernels. The shared stage policy selects placement and storage. The compile path
can materialize typed MPSRT builder plans and Apple source/vendor descriptors
through `src/backends/metal/compiler/` and the shared MPSRT contract headers in
`src/backends/metal/common/mpsrt/`. Runtime model construction and request
encoding then use `src/backends/metal/runtime/mpsrt/gfx_mpsrt_model.*` and the
rest of the runtime MPSRT implementation.

Request-time Metal execution validates external-buffer bindings, model-owned
resources, transient resources, storage bridges, prepared heaps, and
kernel-buffer order before encoding commands. MSL dispatch stages must not
depend on implicit positional conventions when a manifest supplies explicit
roles.

Generated MSL and vendor descriptor payloads are loaded through runtime
descriptors. Current descriptor-backed Metal payload coverage includes generated
MSL for `ShapeOf`, `Range`, `Tile`, `Concat`, `Split`, `Slice`, `Transpose`,
activation, elementwise, numeric/logical reduction, Softmax/LogSoftmax, and
causal SDPA helper forms, plus embedded MPSRT helper kernels for image bridges
and TopK post-processing.

Generated Metal activation payloads are produced by
`src/backends/metal/compiler/msl_codegen_apple_msl_activation.*`. They use
compiler-owned binding plans; `Swish` with a runtime scalar beta uses an
explicit second input role instead of request-time argument inference.

Generated Metal reduction payloads are produced by
`src/backends/metal/compiler/msl_codegen_apple_msl_reduction.*` and loaded from
embedded helper sources in `src/kernel_ir/metal_kernels/reduction_*`. Numeric
f32 reductions and logical boolean reductions have separate source ids and
entry points, but share the same explicit role-based binding shape.

Generated Metal Softmax payloads are produced by
`src/backends/metal/compiler/msl_codegen_apple_msl_softmax.*` and loaded from
embedded helper sources in `src/kernel_ir/metal_kernels/softmax_*` and
`src/kernel_ir/metal_kernels/logsoftmax_*`. `Softmax` and `LogSoftmax` have
separate f32/f16 source ids and entry points. The generated route uses explicit
runtime-parameter roles instead of request-time buffer-order inference.

MPS/MPSGraph vendor routes are compiler-owned `VendorDescriptor` payloads. The
Metal compiler policy can select descriptor-backed payloads for supported
MatMul/GEMM, `Softmax`, Pool2D, Resize2D, and SDPA forms. The shared payload and
contract types live in
`src/backends/metal/common/mpsrt/gfx_mpsrt_vendor_artifact_payload.hpp` and
`src/backends/metal/common/mpsrt/gfx_mpsrt_vendor_contract.hpp`. At runtime,
`MpsrtVendorPrimitiveStage` validates the descriptor payload, builds a typed
single-stage MPSRT model from the vendor contract, adapts the external-buffer
ABI, and encodes the prepared MPSRT model with explicit input/output roles.
Request code should not reconstruct vendor primitive descriptors from local
node checks.

Fused vendor attention is no longer represented by a standalone
`mps_graph_attention_stage.*` runtime stage. The compiler fusion planner builds
a `PipelineVendorAttentionPlan`, the selected Metal backend module
materializes a `VendorDescriptor` artifact, and the pipeline materializer
creates a descriptor-backed `MpsrtVendorPrimitiveStage`.

## OpenCL Architecture

The OpenCL backend is split into:

- compiler policy and kernel-unit registration in `src/backends/opencl/compiler/`
- plugin glue in `src/backends/opencl/plugin/`
- dynamic loader and device selection in `src/backends/opencl/runtime/opencl_api.*`
- plugin-local CLVK/CLSPV bundle discovery in
  `src/backends/opencl/runtime/opencl_runtime_bundle.*`
- buffer manager and memory ops in `src/backends/opencl/runtime/`
- program cache in `src/backends/opencl/runtime/opencl_program_cache.*`
- source-stage execution in `src/backends/opencl/runtime/opencl_source_stage.*`
- runtime source loading in `src/backends/opencl/runtime/opencl_runtime_kernel_loader.*`
- embedded OpenCL helper sources in `src/kernel_ir/opencl_kernels/`

Source kernels are described by `src/kernel_ir/gfx_opencl_source_artifacts.*`.
Payload materialization is owned by
`src/backends/opencl/compiler/opencl_kernel_artifacts.*` and registered through
the OpenCL backend module. The source-stage executor is intentionally generic.
Op-specific behavior should reach it through artifact metadata, generated chunk
artifacts, shared runtime-value planners, static u32/f32 scalar payloads,
constant materialization, and boolean-buffer contracts.
Family-specific backend compiler adapters, currently including
`opencl_conv_kernel_unit.*`, `opencl_pool_kernel_unit.*`,
`opencl_range_kernel_unit.*`, `opencl_softmax_kernel_unit.*`, and
`opencl_tile_kernel_unit.*`, resolve generated `KernelUnit` records and
materialize operation payloads before runtime descriptor construction.

OpenCL support is reported through explicit generated-kernel routes in the
current registry. Source execution still requires a matching artifact payload,
registered kernel unit, and runtime validation; a route in the compiler plan
alone is not enough to guarantee backend parity. The OpenCL operation support
policy intentionally does not fall back to generic MLIR support when no source
artifact exists.

Current OpenCL source artifacts cover data movement, selected converts, MatMul,
Conv2D/GroupConv2D, Softmax, Pool2D, bounded static NCHW spatial Interpolate,
Range, Tile, gather/scatter families, ShapeOf, Concat/Split, unary and binary
elementwise families, compare/select, and boolean logical/reduction families
when shapes and element types match their contracts.

OpenCL source coverage is mostly generated-kernel based. Conv2D and
GroupConv2D use generated f32 units for static 4D NCHW data/output with
constant weights and explicit 2D stride/dilation/padding metadata. Softmax uses
generated f32/f16 units, including dynamic-output static-rank variants whose
scalar ABI carries runtime shape metadata. Range uses generated f32/f16/i64
units, including a specialized i64 unit-step source. Pool2D uses generated
f32/f16 units for static 4D NCHW MaxPool and AvgPool. Interpolate uses embedded
f32/f16 generated kernel units with explicit scalar metadata for resize mode,
coordinate transform, nearest rounding, and NCHW spatial dimensions. ShapeOf,
Tile, logical-bool elementwise, compare/select, boolean reduction, generated
Concat/Split helpers, and Transpose also use explicit `opencl/generated/*`
source ids. The current OpenCL kernel registry has no active handwritten
kernel-unit exception.
Keep those distinctions in the artifact contract rather than duplicating them
in the OpenCL stage executor.
Routes that require runtime shape arguments must mark the `KernelUnit`; the
manifest, executable bundle, runtime descriptor, and infer pipeline carry that
flag instead of deriving it from backend identity at request time.

Reduction OpenCL paths are split by contract. Numeric f32 `ReduceSum`,
`ReduceMean`, `ReduceMax`, `ReduceMin`, `ReduceProd`, `ReduceL1`, and
`ReduceL2` use `opencl/generated/reduction_f32`. Boolean `ReduceLogicalAnd` and
`ReduceLogicalOr` use `opencl/generated/reduction_bool`. Both forms carry
static axis/shape metadata as source-static u32 scalars.

Generated activation artifacts use manifest metadata for opcode, static f32
scalars, direct tensor inputs, and scalar-parameter order. `Swish` supports the
default beta, scalar constant beta, and runtime scalar beta tensor through
dedicated `activation_runtime_beta_*` source ids when the beta input is a static
scalar tensor with the same element type as the data input.

The OpenCL runtime loader checks plugin-local bundle directories before system
or vendor OpenCL libraries. A Raspberry/Linux bundle can be staged from the
`third_party/clvk` and `third_party/clspv` submodules; when bundled CLVK tools
are present, the loader sets the CLVK tool environment only if the caller has
not already provided it. The bundle staging script also copies optional TBB
runtime library families from known runtime output directories, `TBBROOT`, or
`TBB_DIR` when those libraries are available.

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
- fused-output lifetime plans derived from runtime memory descriptors

## Profiling Architecture

Profiling is assembled in shared code and enriched by backend runtimes:

- compile-time report sections
- infer-time node, segment, transfer, allocation, and counter records
- stage estimates such as `bytes_in`, `bytes_out`, `macs_est`, and `flops_est`
- target profile under `extended.target_profile`
- backend counters such as `target_backend_metal` and `target_backend_opencl`
- optional trace sinks registered through `src/runtime/gfx_profiling_trace_sink.*`
  by backend code; the Metal backend currently registers `signpost` and
  `os_signpost`

Use the target profile to confirm which route actually ran before comparing
Metal and OpenCL measurements.

## Build-Time Component Split

The CMake layout separates shared contracts from backend payload ownership:

- `gfx_plugin_core`: plugin-facing compiler and OpenVINO integration
- `gfx_runtime_common`: backend-neutral runtime and scheduling helpers
- `src/compiler/pipeline_stage_builder.*`: stage descriptor construction from
  compiler-owned stage plans
- `src/compiler/pipeline_stage_fusion.*`: compiler-owned fusion selection
- `src/runtime/backend_stage_factory.hpp`,
  `src/runtime/pipeline_stage_desc.hpp`, and
  `src/runtime/pipeline_stage_materializer.*`: runtime-facing stage creation
  and materialization contracts
- `gfx_mlir_stage_support`: MLIR lowering, pass, dialect, and backend-hook
  support
- `gfx_opencl_kernel_artifacts`: OpenCL source artifacts and embedded OpenCL
  helper source wrappers used by the OpenCL backend resolver
- `gfx_runtime_opencl`: OpenCL runtime loader, plugin-local runtime-bundle
  discovery, buffers, program cache, and source-stage execution
- `gfx_metal_mpsrt_contract`: shared Metal MPSRT model/ABI contracts used by
  Metal runtime and tests

Do not move OpenCL source payload materialization back into
`ExecutableBundleBuilder`; backend modules own executable payload resolution.

## Extension Rules

- Prefer shared contracts in `src/compiler/`, `src/runtime/`, `src/kernel_ir/`,
  and `src/mlir/`.
- Add backend-specific code only when the shared contract reaches a real backend
  boundary.
- Keep `query_model()`, compilation, runtime binding, and tests aligned.
- Add operation support and route selection through compiler policies, not
  separate plugin-side tables.
- Keep OpenCL source ids, scalar ABI, constant materialization, chunk helpers,
  and boolean padding in `gfx_opencl_source_artifacts.*`.
- Keep OpenCL payload materialization in
  `src/backends/opencl/compiler/opencl_kernel_artifacts.*`.
- Keep backend stage placement in `src/compiler/stage_placement.*` and
  `src/backends/*/compiler/*_stage_placement.*`; do not move backend-specific
  placement rules back into request code.
- Keep selected backend compiler policy explicit through `CompiledModel`,
  `GpuStageRuntimeOptions`, and `compiler::StageCompilerPolicy`; do not make
  runtime stages rediscover compiler modules through the default registry.
- Keep stage descriptor construction in
  `src/compiler/pipeline_stage_builder.*` and fusion selection in
  `src/compiler/pipeline_stage_fusion.*`; do not rebuild this logic inside
  `CompiledModel` or backend-specific request code.
- Keep descriptor-backed stage creation in
  `src/runtime/pipeline_stage_materializer.*` and backend implementations of
  `BackendStageFactory`.
- Keep memory-region ids, alias groups, transient arenas, and descriptor
  verification in `src/compiler/memory_plan.*`,
  `src/compiler/runtime_executable_descriptor_builder.*`, and
  `src/runtime/executable_descriptor.*`; do not infer resource identity from
  tensor position in request code.
- Keep Metal MPSRT records, MSL source plans, and runtime binding on the
  compiler manifest/MPSRT contract.
- Do not reintroduce the deleted standalone
  `src/backends/metal/runtime/mps_graph_attention_stage.*`; MPSGraph attention
  routes must flow through compiler-owned vendor artifacts and MPSRT vendor
  primitive execution.
- Do not reintroduce removed backend routes, CPU fallback, runtime defines, or
  stale architecture notes.
