# GFX Plugin Architecture

This document describes the current architecture implemented in `modules/gfx_plugin`.

## Design Goals
- expose a single OpenVINO device named `"GFX"`
- share as much logic as possible between Metal and Vulkan
- keep backend-specific code isolated under `src/backends/`
- reject unsupported models early instead of silently falling back to CPU

## Top-Level Structure
- `src/plugin/`: OpenVINO-facing integration
- `src/runtime/`: backend-neutral runtime interfaces and helpers
- `src/runtime/gfx_mpsrt_*`: Apple MPS/MSL ABI, stage-plan, builder-plan, runtime-model, program, storage-bridge, and kernel-manifest helpers
- `src/kernel_ir/gfx_kernel_manifest.hpp`: backend-neutral manifest for stage family, execution kind, storage, and custom-kernel ABI
- `src/mlir/`: MLIR support probing, Apple stage-pipeline lowering, typed MPSRT dialect/materialization, split Apple MSL/MPS source-plan helpers, attention and MatMul Metal/MPSRT helpers, SPIR-V binding adapters, and shared codegen helpers
- `src/backends/metal/`: Metal-specific plugin, runtime, memory, profiling, MPSGraph-backed vendor stages, and codegen pieces
- `src/backends/vulkan/`: Vulkan-specific plugin, runtime, profiling, and codegen pieces
- `src/transforms/`: graph rewrites and fusion logic
- `src/kernel_ir/`: shared kernel metadata and planning structures

## Main Objects
### `ov::gfx_plugin::Plugin`
Defined through `src/plugin/plugin.cpp` and the public header in `include/openvino/gfx_plugin/plugin.hpp`.

Responsibilities:
- register the `GFX` device
- validate and normalize properties
- resolve the backend (`metal` or `vulkan`)
- run backend-aware `transforms::run_pipeline()`
- answer `query_model()`
- create remote contexts

### `ov::gfx_plugin::CompiledModel`
Implemented in `src/plugin/compiled_model.cpp`.

Responsibilities:
- own the transformed runtime model and original model
- create a backend state via `create_backend_state()`
- build the execution pipeline in `build_op_pipeline()`
- expose compiled-model properties and profiling state
- thread `ov::cache_dir` into Vulkan pipeline-cache persistence when the active backend is Vulkan

The compiled pipeline is represented as `PipelineStageDesc` entries that wrap backend-specific `GpuStage` objects. A stage descriptor can also carry output aliases when several OpenVINO node outputs are served by the same runtime stage output buffer.

### `ov::gfx_plugin::InferRequest`
Declared in `include/openvino/gfx_plugin/infer_request.hpp` and implemented by backend-specific infer paths.

Responsibilities:
- bind host tensors or remote tensors
- create per-request backend state
- wire the compiled stage pipeline to actual buffers
- execute the pipeline and collect profiling data

The infer path now has an explicit submission layer. `src/plugin/infer_submission.*` abstracts command-buffer recording and submission windows, while `src/plugin/infer_pipeline.*` can pre-resolve reusable stage-input bindings, reusable output bindings, and reusable output handles for repeated requests.
`src/plugin/stateful_execution.*` adds a second interception layer for stateful graphs: `ReadValue` can source a persisted variable tensor from infer-request state, while `Assign` copies the latest value into a persistent backend buffer owned by that request.
Compiled-model output descriptors now also remember the original source node and source port. That source mapping is used when one pipeline slot covers several fused OpenVINO nodes or when stateful assign prebinding needs to reason about the original graph consumer rather than only the final pipeline stage wrapper.

## Pipeline Model
The runtime is stage-based.

`src/runtime/gpu_stage.hpp` defines the backend-neutral execution interface. Each backend registers a stage factory through `GpuStageFactory` in `src/runtime/execution_dispatcher.*`. During compilation, the plugin translates OpenVINO nodes into stage descriptors. During inference, these descriptors are cloned, connected to inputs and outputs, and executed in order.

The pipeline may fuse multiple nodes into a `FusedSequenceStage` when fusion logic decides that a combined execution path is valid.

Two interface points are important in the current code:
- `set_input_transform()`: a stage can consume metadata about an absorbed upstream transpose instead of requiring a separate materialized transpose stage
- `submit_policy()`: a stage reports scheduling weight and isolation hints used by backend infer paths during command-buffer submission

`src/runtime/gfx_stage_policy.*` selects runtime policy from node shape, element type, backend, and backend capabilities. The policy layer currently decides:
- whether a stage stays on shared SPIR-V dispatch, Apple MSL dispatch, or Apple MPS/MPSGraph vendor primitives
- which storage kind that route expects, such as `buffer`, `image`, or `matrix`
- whether a stage should use direct or chunked execution
- whether bias, activation, or batchnorm can be fused
- how expensive a stage is for submission ordering
- which convolution family or algorithm should be preferred

`src/runtime/gfx_parallelism.*` and `src/runtime/gfx_partitioning.*` now sit next to the policy layer. They use backend-reported execution-device limits to derive:
- backend-neutral parallelism caps
- preferred 1D thread counts
- ranked 2D workgroup shapes for partitioned execution
- capability-gated Conv2D output-channel blocking and spatial micro-tile hints
- stable device keys used by runtime planning caches

Those caps are no longer purely backend-wide defaults. The current runtime tags devices by family through `GpuDeviceFamily` and feeds that into planning and cache keys. Current families include:
- `apple`
- `adreno`
- `broadcom_v3d`
- `generic`

During inference, `execute_pipeline_with_submission()` groups recorded stages into submission windows. The current grouping rules use:
- `GpuStageSubmitPolicy`
- maximum stages per submit
- maximum recorded output bytes per submit
- maximum recorded MAC estimates per submit
- direct producer-consumer dependency tracking for soft-budget extension
- backend support for incremental submit versus single-flight submit

This lets Metal and Vulkan keep different submission mechanics while sharing the same stage-level batching logic.

Submission tuning is no longer a fixed constant table. The current infer path derives per-backend submit-window sizing from backend capability snapshots and records the selected tuning into the profiling path when profiling is enabled.
When a stage directly consumes an output already recorded in the current window, the submit layer may extend past the soft stage/output/MAC budget up to a configured dependency-extension cap. That extension is deliberately blocked at layout or fan-out boundaries such as `Concat`, `Split`, `VariadicSplit`, `Transpose`, `Reshape`, `Softmax`, `LogSoftmax`, and fused attention so dependency tracking does not hide required synchronization points.
Vulkan infer paths can also batch immutable constant uploads through the shared infer command buffer path instead of forcing one upload submit per constant buffer materialization.
Metal infer paths now also keep one compute encoder alive across consecutive dispatches on the same command buffer and skip redundant pipeline-state or buffer rebinds when the cached encoder state already matches the next kernel launch.

The reusable infer pipeline layer now has two precomputed plans:
- `PreparedInferExecutionPlan` for stage inputs that resolve to parameters or previous stage outputs
- `PreparedInferOutputPlan` for public outputs that resolve to stage outputs, passthrough parameters, or synthetic pipeline entries created for constant outputs

When output shape and element type are statically known, infer requests may also keep reusable host output tensors in backend infer state to avoid recreating host tensors on each execution.
If a reusable host output no longer matches the expected static type or shape, the infer path now recreates it instead of treating that mismatch as a hard internal error.

Stage-output allocation also has a liveness-aware workspace layer:
- `StageOutputBufferWorkspace` can recycle intermediate output buffers across stages when the output lifetime does not escape the current live range
- `GpuStage::describe_output_lifetimes()` lets complex stages such as `FusedSequenceStage` report finer-grained internal lifetimes
- infer requests report workspace allocation counters such as slots used and peak live slots through the normal profiling path

Some view-style data movement now skips copies entirely. `Split` and `VariadicSplit` can alias contiguous byte ranges of the input buffer when the inferred split plan is view-compatible, so downstream stages can consume output tensors backed by slices of the original allocation.
Fused-stage lifetime reporting also recognizes some guaranteed storage-alias paths such as `Reshape`, `Squeeze`, and `Unsqueeze`, allowing internal outputs to reuse the first input allocation instead of reserving a fresh workspace slot.

## MLIR Role
MLIR lives in `src/mlir/` and is shared infrastructure, not a separate monolithic backend object.

It is used for:
- support probing through `mlir_supports_node()`
- node lowering via `mlir_builder_*.cpp`
- backend code generation helpers such as MSL or SPIR-V preparation

Current lowering also supports absorbed input transforms. `CompiledModel` detects selected transpose patterns and forwards them as `GfxInputTransform` metadata to builders such as Add, Conv2D, GroupConv2D, and Split. This keeps the runtime pipeline smaller while preserving the original layout semantics.

The MLIR pass pipeline also contains parallel-lowering and cleanup steps used by current Vulkan codegen, including Conv2D parallel lowering, Conv2D im2col rewrite/lowering, matmul parallel lowering, and post-fusion cleanup passes.

Recent MLIR-specific changes reflected in the current code:
- OpenVINO `RMSFusion` now runs before plugin-local cleanup so common RMSNorm tails reach a dedicated `RMS` lowering path
- Softmax lowering handles arbitrary normalized axes
- Reduce lowering now resolves axes and `keep_dims` through concrete Reduce op types instead of a generic reduction-base lookup
- dedicated builders now exist for `RMS`, `ScatterUpdate`, and `RoPE` instead of relying only on more generic lowering families
- fusion now also supports selected input-side activations for `Multiply`, not only output activations or bias/post-op forms
- Metal `RMS` stages can fuse one residual `Add` input directly into the RMS kernel when the upstream shape contract is preserved
- the backend-aware transform pipeline can fuse the common LLaMA rotate-half arithmetic pattern into native `RoPE` before stage compilation on Metal
- the backend-aware transform pipeline can also fuse selected LLM `ScaledDotProductAttention` masking graphs into `GfxSDPAWithCausalMask`, including peeling some broadcast-expanded GQA K/V views back to their compact tensors
- the backend-aware transform pipeline can regroup compatible compressed `MatMul` nodes that share one data input into a fused horizontal `MatMul` plus `VariadicSplit`
- Metal MSL source planning now includes dedicated compressed `MatMul`, SDPA, Apple MSL custom-kernel, Apple MPS vendor, and MatMul MPSRT source-plan helpers and uses runtime binding plans so generated modules carry tensor, output, scalar, runtime-parameter, and const-tensor roles explicitly
- `ShapeOf` has a dedicated runtime-materialized dims path, while `TopK` and unary codegen now respect more of the concrete output/index typing rules from the original node
- Slice lowering now prefers `tensor.extract_slice`, while slice metadata extraction still accepts both `tensor.extract_slice` and the older `linalg.generic` form
- shape and data-movement lowering is now more permissive for dynamic shapes in paths such as `ShapeOf`, `Concat`, `Broadcast`, `Select`, `StridedSlice`, and `Range`
- buffer-results-to-out-params promotion now allows public function signatures to be rewritten when required by the lowering pipeline
- shared helpers now prefer the common `gfx_mlir_context()` path instead of ad-hoc local MLIR contexts in selected code paths
- convolution parallel lowering can now consume explicit module-level dispatch attrs such as `gfx.dispatch_threads_*` and `gfx.dispatch_tile_*` instead of relying only on coarse algorithm variants
- convolution parallel lowering also consumes `gfx.dispatch_channel_block` and `gfx.dispatch_channel_block_accumulation` so Vulkan can compute multiple adjacent output channels per work item without introducing a separate Conv executor path
- convolution parallel lowering now has a separate interior-tile fast path that skips lane guards for full tiles and keeps guarded edge handling only where needed
- interior-tile eligibility is now factored through separate height and width window checks before the combined 2D interior fast path is selected
- manual Vulkan MLIR kernels can now emit `gpu.func` entry points while materializing buffer ABI through the common `GfxKernelStageManifest` custom-kernel adapter rather than local `gfx.fixed_arg_count` shortcuts; this covers the executor's unary/binary, concat/split, slice, interpolate, transpose, convert, gather, reduce, RMS, MatMul, broadcast, and select routes; Conv2D and GroupConv no longer have separate manual Vulkan builders or executor-level direct/chunked routes and use the shared canonical MLIR lowering path instead
- SPIR-V compact-memref preservation is manifest-native: `gfx.fixed_arg_count` is no longer a compile-path source for compact ABI reconstruction or entry metadata annotation
- kernel-signature and metadata helpers now resolve `gpu.func` entry points before falling back to plain `func.func`, so Vulkan launch metadata stays aligned with GPU-entry modules
- TopK now uses rank-aware MLIR indexing for the shared custom-kernel path. The Vulkan lowering does not depend on collapse/expand-shape legalization for TopK anymore, and `i64` index outputs are represented at the shader boundary as two `i32` lanes in the public `i64` output buffer to avoid relying on Vulkan shader `Int64` support on Android/RPi-class devices.
- Metal `MatMul` codegen can now derive input element types from the effective runtime tensors, which matters when stage compilation repacks a constant RHS from `f32` to `f16` for dynamic-shape paths
- layout cleanup can now fold the DFL softmax expectation tail into a `Softmax -> MatMul -> Reshape/Transpose` form that preserves output values without the older synthetic convolution path

Lowered kernels also rely on backend-neutral argument and binding helpers:
- `src/kernel_ir/gfx_kernel_args.hpp` materializes runtime kernel arguments and can turn scalar byte payloads into cached immutable device buffers
- `src/runtime/gpu_backend_base.hpp` provides shared prepared-binding caches reused across compatible compiled-kernel wrappers

Prepared-binding caches are no longer hard-capped at one fixed size. They start from an initial bound and can grow when repeated executions observe more distinct compatible binding tables than the initial cache budget allowed.

Outside kernel execution, `src/plugin/infer_pipeline.cpp` can also materialize constant graph outputs into synthetic pipeline tensors so output resolution stays uniform even when the public output does not come from a runtime stage.
On Metal, MLIR and stage-policy metadata can also be serialized into a compact MPSRT runtime-model boundary: stage placement is converted into `GfxMpsrtStageDesc`, tensor/storage metadata becomes `GfxMpsrtTensorDesc` or ABI structs, and MSL-dispatch stages carry kernel-family plus external-buffer-role metadata so request-time binding does not rely on a hard-coded argument convention.
The runtime-model reconstruction and resource-table logic lives in the backend-neutral `src/runtime/gfx_mpsrt_model.*` files. Metal-specific files consume that model to prepare pipelines, Apple MPS kernels, resource heaps, and request-time encodes.
That boundary is now manifest-driven rather than ad-hoc. `GfxKernelStageManifest` describes whether a stage is a vendor primitive or a custom kernel, which backend/storage family it belongs to, and what external-buffer ABI the custom kernel expects. The MPSRT adapter layer converts stage-policy placement into backend-neutral manifest domains, resolves custom-kernel family dispatch metadata, and then converts that manifest into the runtime record and buffer-order form used by Metal execution. MPSRT stage planning must request that adapter instead of directly calling the custom-kernel family factory.
The compile path now emits one generated MLIR helper through `gfx_mpsrt_ops.*`:
- `gfx_mpsrt_ops`, a typed stage/program facade derived from `GfxMpsrtProgram`

Metal codegen consumes this typed program directly and materializes
`GfxMpsrtBuilderPlan` in C++ from the manifest-backed program records. The older
runtime ABI helper layer was removed, so the typed program and
`GfxKernelStageManifest` remain the only semantic source for stage record keys,
external-buffer roles, per-stage descriptors, runtime resource classification,
and explicit storage bridges such as `buffer_to_image` and `image_to_buffer`.
`GfxMpsrtBuilderPlan` carries only the model-level `model_record_key`; per-stage
record keys are derived from `GfxMpsrtStageDesc` while emitting encode records.
When a custom-kernel manifest carries explicit external buffer roles, those
roles are an exact ABI. MPSRT source planning and typed-program materialization
must use the manifest/builder-plan external buffer count ahead of any raw
`KernelSource.signature` hint, and must not trim `RuntimeParams` or `Metadata`
roles to match an older source signature. Signature hints are retained only as a
compatibility fallback for manifest shapes such as leading-IO specs that do not
carry exact roles.
Metal compile-time binding uses the same rule: once the MPSRT runtime model has
an exact external-buffer ABI, `KernelSource.signature` and MSL `[[buffer(N)]]`
scans are diagnostics/fallback inputs only and must not widen the compiled
binding schema or resolved-MSL cache identity.
This includes stale or unused high-index `[[buffer(N)]]` declarations in source:
explicit MPSRT roles win, and direct MSL buffer scans remain a no-MPSRT fallback.
The same rule applies before runtime-model construction: early `source_arg_count`
normalization must use the exact manifest/typed ABI when it exists.
The compiled Metal `KernelBindingPlan` is still a full runtime argument schema:
when the stage manifest has explicit roles, its custom-kernel role list is the
exact source for that count, so `ScalarParam` byte arguments are included there
while the MPSRT external-buffer ABI remains the narrower tensor/buffer-only
boundary.
MPSRT runtime-model adaptation must not repair a too-small legacy arg count by
dropping explicit `RuntimeParams` or `Metadata` roles from that external-buffer
role list. A count/role mismatch is a lowering error and must fail before
request binding; the semantic ABI is never silently narrowed inside the runtime.
After `gfx_mpsrt_ops` has been materialized, external-buffer ABI readback must
prefer the typed `GfxMpsrtProgram` records over any module-level
`gfx.stage_manifest.*` attrs that may remain from the annotation pass. The
module manifest is only a pre-materialization fallback, not a second ABI source.
Metal compile-time schema sizing follows that same priority: when a typed
program has an MSL dispatch stage with a custom-kernel manifest, its stage-local
manifest supplies the exact Metal runtime argument count before any stale
module-level `gfx.stage_manifest.*` attrs are considered. If generated typed ops
are present but cannot be read back, module-level attrs must not mask that
failure; the stale manifest is not a recovery path after materialization.
The Metal compile boundary treats this as a hard typed-program error: a module
with generated but invalid `gfx_mpsrt_ops` must not compile as a raw MSL kernel
or rebuild its runtime schema from legacy source/signature hints.
For direct Apple MSL sources that have not been materialized into an MPSRT
model yet, an exact custom-kernel stage manifest still wins over stale
`KernelSource.signature` counts and source scans when sizing the Metal binding
schema.
If a prebuilt Apple MSL source already carries that exact external-buffer ABI
and its signature agrees with the ABI, `msl_codegen.cpp` treats the source as
source-plan owned and does not rerun node-based family configuration over it.
`MlirStage::compile_prebuilt_kernel_source()` then installs the supplied
runtime binding as source-plan owned, so request-time binding cannot fall back
to a wider generic family ABI.
When MSL source configuration sees a module that already carries an Apple MSL
custom-kernel stage manifest, that manifest wins over the per-family
`stage_type`/`entry_point` factory defaults. The family factory is only a
compatibility fallback for modules that have not been annotated yet; it must not
rewrite an exact role ABI that came from the canonical lowering path.
An exact manifest ABI also must not be widened to the generic family default just
because the fallback factory can describe more operands. Source configuration may
use the fallback only after the manifest-backed binding is absent or invalid.
When that fallback creates a valid Apple MSL binding from `KernelSource` plus the
requested OpenVINO stage type, a wider legacy `KernelSource.signature` must not
discard the binding. The adapter-resolved manifest roles replace the source
signature; the old signature remains only a no-manifest fallback hint.
Apple MSL source-plan helpers delegate manifest/fallback resolution to
`gfx_backend_custom_kernel_adapter.*`. Per-family MSL files can request a
binding plan and apply it to a source/module through selector/source helpers
such as `make_backend_custom_kernel_source_binding_plan()`, but they must not
duplicate local manifest readers, backend-domain selection, specialization
prefixes, or `make_binding_plan_from_*` stacks; Apple MSL and SPIR-V
custom-kernel ABI mapping stays in the shared backend custom-kernel adapter.
Generated MSL source families use the shared
`configure_msl_generated_custom_kernel_source()` path for entry-point
normalization, source-signature materialization, exact role preservation, and
optional module annotation.
Prebuilt/direct MSL source plans, including direct Split, must apply their
resolved custom-kernel binding through
`configure_backend_custom_kernel_source_from_binding_plan()` so annotation,
signature materialization, and exact-role preservation stay in the same shared
adapter boundary.
Direct-IO and role-based source-plan helpers must also keep the manifest
`entry_point` equal to the generated kernel entry point, not the generic family
name, so typed/readback validation and Metal cache keys describe the same
kernel.
Compile-time stage routes must use required binding helpers from
`gfx_backend_custom_kernel_adapter.*` instead of spelling out local
`make_*_binding_plan` plus annotation sequences in `mlir_stage.cpp`.
Scalar-payload routes and ABI-only routes have separate helpers, but both stay
inside the same adapter boundary.
MSL source-family files also use the same adapter boundary through
`require_backend_custom_kernel_source_binding()`; the old Apple-only source
binding adapter layer was removed instead of being kept as a parallel shortcut.

The Apple stage pipeline in `gfx_apple_stage_pipeline.*` sits earlier in this path. It canonicalizes an Apple-targeted module, validates placement/storage/fusion boundaries, lowers the resulting stage plan into the canonical stage manifest, and can then materialize the typed MPSRT program facade from that same stage plan. It does not use `gfx.apple.pipeline.*` module attributes as a second metadata surface: pass boundaries stay available through the C++ pipeline API, semantic state lives in `gfx.stage_manifest.*`, and storage bridges are serialized only in the typed `GfxMpsrtProgram`/`gfx_mpsrt_ops` records. Vendor descriptor extraction for Apple MPS/MPSGraph stages is shared through `gfx_apple_vendor_descriptors.*`, so Conv2D, Pool2D, Resize2D, Softmax, TopK, and SDPA all feed the same descriptor and tensor-desc path.

Generated `gfx.mpsrt.*` stage ops are verified from their op name and canonical `gfx.stage_manifest.*` attributes. They do not emit duplicated `gfx.mpsrt.op.stage.backend`, `gfx.mpsrt.op.stage.stage_kind`, `gfx.mpsrt.op.stage.stage_record_key`, `gfx.mpsrt.op.stage.input_storage`, `gfx.mpsrt.op.stage.output_storage`, or `gfx.mpsrt.op.stage.layout` attrs; those belonged to the older side-channel descriptor surface. Readback reconstructs storage/layout from explicit typed value edges and input/output tensor descriptors, so transient outputs from earlier stages remain part of the same typed program model.
Builder-plan stage specs, module stage plans, and Apple MPS lowering plans follow the same identity rule: they do not carry a stored per-stage record key, and `gfx_mpsrt_make_builder_plan()` derives the key from `GfxMpsrtStageDesc` at the runtime boundary.

## Profiling Stack
Profiling is split into compile-time and infer-time collection.

- `src/runtime/gfx_compile_profiling.hpp` provides thread-local compile scopes used while `CompiledModel` creates backend state, builds stages, and compiles kernels
- `src/runtime/gfx_profiling_report.*` owns the JSON-ready report model for nodes, segments, transfers, allocations, and counters
- backend profilers under `src/backends/*/runtime/profiling/` populate runtime timing and backend-specific counters
- `src/plugin/gfx_profiling_utils.hpp` merges compile and infer reports into the final `GFX_PROFILING_REPORT` JSON and can optionally emit Perfetto-style trace payloads
- `src/plugin/infer_submission.cpp` now also attaches lightweight `bytes_in`, `bytes_out`, `macs_est`, and `flops_est` estimates to `stage_execute` segments so the extended summaries can derive simple roofline-style hints without a separate replay pass

When profiling is disabled, these paths stay out of the fast path.

## Metal Backend
`src/backends/metal/` contains:
- `plugin/`: backend state creation, infer request, remote context, remote tensor
- `runtime/`: request-time execution, memory allocators, MPSRT runtime-model execution, executor, profiler
- `codegen/`: Metal shader compilation helpers

Direct Metal API usage lives in Objective-C++ files (`.mm`).

The current Metal backend also shares immutable constant-buffer state across compatible requests through `MetalConstCache` and the backend-neutral immutable buffer cache helper.
It now also reports execution-device limits through `GpuExecutionDeviceInfo`, so backend-neutral planning code can use real Metal subgroup and threadgroup limits.
Stage policy now splits Metal work into two internal domains:
- Apple MPS image or matrix stages for selected conv/pool/interpolate/matmul/softmax/topk shapes
- Apple MSL buffer dispatch for the remaining custom-kernel cases

The Metal path also now has a local MPSRT execution layer under `src/backends/metal/runtime/mpsrt/`:
- `src/runtime/gfx_mpsrt_model.*` validates builder records, reconstructs compact runtime models, finalizes runtime resources, and derives external/tensor binding plans used by Metal execution
- `mpsrt_context.*` prepares and caches Metal pipeline states or Apple MPS kernels, prepares model resources, and allocates transient buffers/images through a Metal heap
- `mpsrt_request.*` binds resources from the explicit runtime resource table and encodes prepared dispatches into a command buffer
- `gfx_mpsrt_storage_bridge.hpp` describes explicit conversions between external buffer bindings and Apple MPS image storage when a stage crosses that boundary
- `gfx_mpsrt_program.hpp` defines the typed MPSRT program contract validated before builder-plan generation
- `gfx_mpsrt_dialect.*` defines typed `gfx.mpsrt.*` helper ops used by the generated program facade, including explicit storage-conversion ops such as `to_image`, `to_matrix`, `to_ndarray`, `to_buffer`, and `alias`

That split is driven by the backend-neutral manifest layer in `gfx_kernel_manifest.hpp` plus the custom-kernel family registry in `gfx_custom_kernel_families.*`. The registry classifies MSL kernels into stable families such as eltwise, transpose/packing, concat/split, gather/scatter, RMS/RoPE, masked softmax-attention, Conv2D/Conv3D, MatMul, pooling, BatchNorm, and reduction dispatch. It also defines the external-buffer ABI roles and dispatch policy expected by request-time binding, while `runtime/gfx_mpsrt_kernel_manifest_adapter.hpp` is the runtime boundary that turns those common manifests into MPSRT custom-dispatch specs.
The manifest layer now also carries stage-family information for vendor primitives and custom kernels, semantic input/output roles, and dispatch-grid metadata, which lets one MPSRT model mix execution kinds while keeping one stable record key and buffer-order contract.
Recent Metal compile/runtime changes extend the MPSRT path beyond one standalone MSL stage:
- vendor-only plans such as `MPSGemm` and MPSGraph-backed SDPA can now execute through the MPSRT runtime boundary without requiring generated MSL source
- hybrid multi-stage plans such as `MPSGemm + MSL epilogue` can be serialized as one MPSRT model with semantic inputs/outputs, explicit intermediate value edges, storage bridges, and stage-local descriptors
- request-time execution can choose full-context MPSRT execution for those mixed models instead of falling back to one raw compiled-kernel dispatch
- vendor primitive coverage now also includes Apple MPS convolution, group convolution, pooling, spatial bilinear Resize2D, softmax, TopK, and MPSGraph SDPA stages when stage policy selects those routes
- compile-time source planning now uses `gfx_mpsrt_source_plan.hpp` to pick `SingleStage` versus `MultiStage` source contracts directly from the typed program/module metadata, while `gfx_mpsrt_const_tensor_sources.hpp` attaches evaluated constant payloads from typed MPSRT program inputs marked as `ConstTensor`/`GfxMpsrtTensorFlagConst`
- metadata cleanup now removes stale flat `gfx.mpsrt.*` stage attrs after the generated program/ops facade is materialized, so current readers should prefer `read_module_mpsrt_program()`, typed `gfx.mpsrt.*` ops, and runtime-ABI helpers over direct attribute scraping
- runtime-model finalization now builds `MpsrtRuntimeResource` entries and `external_buffer_bindings`, then request binding validates external IO, runtime-parameter resources, model-owned const buffers, and prepared transient resources against that table; Metal vendor-stage preparation also consumes model-owned constants through `MpsrtPreparedResource` instead of scanning the const-pack cache directly
- prepared MPSRT execution can allocate transient buffer and image resources from one Metal heap and call `makeAliasable` after each resource's live window, so non-overlapping intermediate resources may reuse heap storage

MatMul is the clearest current example of that split:
- plain supported GEMM shapes can lower to vendor `MPSGemm`
- bias or supported activation epilogues can extend that into `MPSGemm + MSL epilogue`
- the MSL epilogue custom-kernel manifest is resolved through the shared backend custom-kernel adapter instead of hand-writing Apple MSL family IDs, dispatch policy, or `apple_msl:buffer:*` specialization strings in MatMul metadata
- the final kernel source plan is selected through the `gfx_mpsrt_source_plan` helpers rather than only by a single `entry_point` string

Convolution follows the same direction on the metadata side:
- Apple MPS convolution/group-convolution stages now serialize explicit `GfxMpsrtConv2DAbiDesc` stride, dilation, pad, and grouping metadata
- the custom Metal Conv2D kernel family is also represented through the same manifest path, so legacy MSL `conv2d_kernel` dispatch still shares the common kernel-family and ABI contract
Interpolate follows the vendor-primitive path for the supported static NCHW spatial bilinear cases: the Apple stage pipeline emits a `GfxMpsrtResize2DAbiDesc`, source planning treats it as an IO-only MPS vendor stage, and the Metal runtime prepares/encodes `MPSImageBilinearScale`.
When those Apple MPS image-backed stages connect to public buffer inputs or outputs, the runtime model carries storage-bridge descriptors and the request path materializes the needed bridge resources explicitly rather than assuming one storage class end-to-end.
Metal custom MSL source generation is centralized in MLIR source-plan helpers and `MlirStage` binding-plan annotation. That path covers dynamically shaped or metadata-heavy kernels such as `Softmax`, `Select`, `ScatterUpdate`, `RMS`, `RoPE`, direct-IO `Concat`, rank-4 `ScaledDotProductAttention`, fused causal-mask `ScaledDotProductAttention`, compressed `MatMul`, and slice handling including negative-step `StridedSlice`.
The source-plan files are now split by responsibility: `msl_codegen_apple_msl_*` owns Apple MSL custom-kernel source planning, stage-manifest materialization, source entry-point normalization, and family dispatch, including direct Concat/Split structural generators in `msl_codegen_apple_msl_concat_split.cpp`; `msl_codegen_attention.*` owns SDPA MSL kernels and runtime params; `msl_codegen_compressed_matmul.*` owns compressed MatMul detection, packing, and source planning; `msl_codegen_apple_mps.*` owns vendor source-plan selection; and `msl_codegen_matmul_metal.*` plus `msl_codegen_matmul_mpsrt.*` own the direct Metal and MPSRT MatMul routes. The top-level `msl_codegen.cpp`/`.hpp` should stay a thin cross-route coordinator surface and must not grow new Apple MSL dispatch policy, family-specific source-plan APIs, structural source generators, or specialization-key construction. Concat request execution now uses the same direct-IO runtime binding installed by the source plan; the older per-input runtime-parameter launch path has been removed.
`GfxMslRuntimeBindingPlan` converts each manifest role order into the module/runtime operand metadata used by request-time binding. For direct MSL dispatch inside MPSRT, the runtime model must carry materialized `kernel_buffer_order`; request execution rejects an MSL dispatch stage when that order is absent.
Conv3D follows the same custom-kernel role contract on both Apple MSL and SPIR-V: `TensorInput`, `ConstTensor`, `TensorOutput`, and `RuntimeParams`. `MlirStage` materializes that role order as a const-weight extra buffer followed by the packed `Conv3DParams` runtime buffer, annotates rank-5 convolution modules as the `conv3d` custom-kernel family before `KernelSource` creation, and configures the source signature from that manifest on every backend. Request-time binding is therefore driven by the shared manifest ABI instead of a Conv3D-only argument shortcut or the older Conv2D direct/im2col manifest.
Rank-4 Conv2D and GroupConv custom-kernel paths use the same backend custom-kernel adapter. `MlirStage` now refreshes their const-weight and packed runtime-parameter extra inputs through one helper used by both compile-time source creation and request-time execution. On Vulkan that request-time refresh must preserve the compiled final SPIR-V binding state; it updates payload buffers, but it does not replace final adapter metadata with the older pre-lowering source binding.
SPIR-V compact-ABI serialization is kept separate in `spirv_kernel_binding_adapter.hpp`, but Vulkan manual routes now request explicit buffer roles through the common `GfxKernelStageManifest` custom-kernel adapter. Required backend custom-kernel annotations write the common manifest for SPIR-V and then serialize temporary `gfx.kernel_operand_*` and `gfx.kernel_scalar_values` attrs only as the final SPIR-V adapter surface under a valid SPIR-V custom-kernel manifest. When SPIR-V lowering scalarizes or reorders `gpu.launch_func` operands, the reconciler updates the manifest's external-buffer roles and `custom_kernel.scalar_args` where the roles can represent the final ABI. If final lowering still needs the serialized adapter surface to preserve scalar byte-buffer positions, runtime metadata extraction accepts those attrs only because the module already carries the valid SPIR-V custom-kernel manifest. A SPIR-V module-cache hit restores the lowered launch metadata and immediately reconciles the stage manifest again, so cached binaries and runtime binding always share the same final ABI. `KernelPlan`, arg-count inference, and runtime metadata extraction reject no-manifest legacy operand attrs by default; `gfx.fixed_arg_count` is no longer accepted as a runtime metadata source or arg-count source. The only no-manifest fallback kept in the ordinary path is the explicit signature/count fallback for modules that contain no legacy operand or scalar attrs. The SPIR-V cache/retry path, arg-count/runtime metadata extraction, and `run_mlir_pipeline` compact-ABI preservation all use manifest state first. Vulkan codegen no longer shrinks a manifest-defined binding plan down to the shader-observed descriptor count; narrow kernels must declare their direct buffer ABI in the manifest. Supported Vulkan binary elementwise stages of every size use the same manifest-backed `linear_binary` metadata/broadcast path, not executor-local same-shape or bias-add shortcuts.
For dynamic-shape `MatMul`, the current MLIR stage path may also pack a constant RHS from `f32` to `f16` before wrapping it as an immutable const buffer. Compile profiling counters track the original and packed byte sizes for that optimization.
The backend-aware transform pipeline is also important for Metal now: compressed `MatMul` decompression subgraphs can be marked as decompression and protected from generic folding so later stage compilation can still recognize weight-only compressed patterns.
That same transform pipeline can also collapse supported LLaMA rotate-half arithmetic into native `RoPE`, letting Metal compile one backend kernel instead of preserving the original Multiply/Add subgraph.
For attention-heavy LLM graphs, the same frontend/backend split now also allows Metal-only `GfxSDPAWithCausalMask` lowering while leaving Vulkan on the conservative unsupported path for that exact fused form.
Recent Metal codegen changes also make Conv2D and MaxPool honor dilation metadata directly and allow Conv2D dispatch planning to block output channels and output width per thread for selected float-like kernels.

## Vulkan Backend
`src/backends/vulkan/` mirrors the same broad split:
- `plugin/`: backend state, infer request, remote context, remote tensor
- `runtime/`: executor, buffers, profiling, runtime helpers
- `codegen/`: SPIR-V / Vulkan codegen helpers

This backend is built only when Vulkan support is available and enabled in CMake.

The current Vulkan runtime also:
- records physical-device limits such as subgroup size and compute workgroup limits in `VulkanContext`
- classifies devices into families such as `adreno` and `broadcom_v3d`
- uses those limits when selecting parallelism and stage execution policy
- contains specialized execution routes for chunked unary/binary/softmax/layout ops
- routes supported binary elementwise cases through the single `linear_binary` metadata/broadcast mechanism
- reuses immutable constant buffers and prepared descriptor bindings across compatible submissions
- batches pending constant-buffer uploads before the main infer recording path begins
- increases per-submit batching thresholds in the infer path to reduce Android-oriented driver overhead
- persists Vulkan pipeline-cache data under `ov::cache_dir` when a cache directory is supplied through standard OpenVINO properties
- reports execution-device limits through `GpuExecutionDeviceInfo`, matching the Metal path and removing backend-specific probing from shared planning code
- keeps Conv2D and GroupConv2D dispatch metadata owned by the shared MLIR/SPIR-V custom-kernel path instead of executor-local convolution kernels
- reconciles final SPIR-V launch/binding metadata back into the common stage manifest before runtime metadata extraction; shader-observed binding counts are diagnostic or fallback input only and do not shrink a manifest-defined ABI
- keeps TopK on the shared manifest-backed MLIR/SPIR-V custom-kernel route. TopK may widen only the internal shader view of an `i64` index output to packed `i32` lanes; the OpenVINO-visible tensor type and shape remain the model contract.
- does not keep a second direct Vulkan TopK runtime route. TopK correctness must be fixed in the shared lowering/manifest path, not hidden behind a skip, CPU fallback, or runtime environment switch.
- keeps Gather i64 handling on the GPU by using the same packed-lane principle for Vulkan i64 indices/data and by binding Metal Gather `RuntimeParams` through the manifest-backed extra-buffer ABI instead of treating axis constants as kernel params
- broadens specialized handling for boolean, compare, select, range, broadcast, gather, slice, and reduction-style kernels used in dynamic-shape-heavy graphs
- adds specialized chunked kernels for binary `Concat` and `RMS` when their shape and element-type constraints are satisfied

The current policy layer also deliberately prefers the shared MLIR/SPIR-V convolution path over older dedicated Vulkan 1x1 and 3x3 direct routes on current mobile-class stacks, keeping the backend contract more stable for Android and Raspberry Pi flows.

## Remote Contexts And Tensors
The shared abstractions are:
- `src/runtime/gfx_remote_context.*`
- `src/runtime/gfx_remote_tensor.*`

Backend-specific files provide the actual buffer wrapping and device-handle logic. Remote objects exist today, but the feature surface is still backend-dependent and not yet a fully polished portability layer.

## Import / Export Semantics
- `import_model()` reads a serialized OpenVINO model and recompiles it for the resolved backend.
- `export_model()` writes an OpenVINO model representation.

This is model serialization, not compiled-kernel caching.

## Important Constraints
- no partial CPU fallback
- backend implementation parity is not automatic: Mac uses the Metal/MPS/MSL backend, while Android and Raspberry Pi use the Vulkan/SPIR-V backend. The public OpenVINO tensor contract and compare-runner acceptance criteria are shared, so a Mac-only pass is not enough evidence for Android/RPi correctness.
- targeted per-op skips are represented as explicit compare-runner outcomes (`PER_OP_SKIPPED`) and preserved by `bench/gfx_eval.py`; backend gaps should not be hidden as successful matches or per-device test exclusions.
- once an Apple MPS vendor contract is selected, its typed input/output descriptors own the MPSRT ABI and storage-bridge assignment; the intermediate MLIR helper signature must not remain a second source of descriptor counts.
- Apple MPS Pool2D materialization is single-output only; indexed `MaxPool` variants with a second indices output must remain in the custom-kernel path until the typed vendor contract can represent both outputs.
- `f32` MPSImage placement is quality-guarded: precision-sensitive nodes and any image producer that can influence an order-sensitive `TopK` must stay on the route selected by stage policy, because small score drift can change indices and later detection gathers even when local tensor diffs are below the ordinary numeric threshold. The diagnostic property is for route localization, not for bypassing that policy.
- many ops still require static rank, static shape, or constant attributes
- the plugin remains experimental
- some lowering and runtime optimizations are intentionally backend-specific, so architecture docs should describe the shared model first and backend specialization second
