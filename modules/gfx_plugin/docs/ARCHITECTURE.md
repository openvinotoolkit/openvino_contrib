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
- `src/mlir/`: MLIR support probing, Apple stage-pipeline lowering, typed MPSRT dialect/materialization, split Apple MSL/MPS source-plan helpers, MatMul Metal/MPSRT helpers, SPIR-V binding adapters, and shared codegen helpers
- `src/backends/metal/`: Metal-specific plugin, runtime, memory, profiling, and codegen pieces
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
- whether a stage stays on shared SPIR-V dispatch, Apple MSL dispatch, or Apple MPS vendor primitives
- which storage kind that route expects, such as `buffer`, `image`, or `matrix`
- whether a stage should use direct or chunked execution
- whether bias, activation, or batchnorm can be fused
- how expensive a stage is for submission ordering
- which convolution family or algorithm should be preferred

`src/runtime/gfx_parallelism.*` and `src/runtime/gfx_partitioning.*` now sit next to the policy layer. They use backend-reported execution-device limits to derive:
- backend-neutral parallelism caps
- preferred 1D thread counts
- ranked 2D workgroup shapes for partitioned execution
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
- backend support for incremental submit versus single-flight submit

This lets Metal and Vulkan keep different submission mechanics while sharing the same stage-level batching logic.

Submission tuning is no longer a fixed constant table. The current infer path derives per-backend submit-window sizing from backend capability snapshots and records the selected tuning into the profiling path when profiling is enabled.
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
- convolution parallel lowering now has a separate interior-tile fast path that skips lane guards for full tiles and keeps guarded edge handling only where needed
- interior-tile eligibility is now factored through separate height and width window checks before the combined 2D interior fast path is selected
- manual Vulkan Conv2D MLIR building can now emit `gpu.func` entry points for batch-1 parallel dispatch plans and keep a serial `func.func` entry path for larger batches
- kernel-signature and metadata helpers now resolve `gpu.func` entry points before falling back to plain `func.func`, so Vulkan launch metadata stays aligned with GPU-entry modules
- Metal `MatMul` codegen can now derive input element types from the effective runtime tensors, which matters when stage compilation repacks a constant RHS from `f32` to `f16` for dynamic-shape paths
- layout cleanup can now fold the DFL softmax expectation tail into a `Softmax -> MatMul -> Reshape/Transpose` form that preserves output values without the older synthetic convolution path

Lowered kernels also rely on backend-neutral argument and binding helpers:
- `src/kernel_ir/gfx_kernel_args.hpp` materializes runtime kernel arguments and can turn scalar byte payloads into cached immutable device buffers
- `src/runtime/gpu_backend_base.hpp` provides shared prepared-binding caches reused across compatible compiled-kernel wrappers

Prepared-binding caches are no longer hard-capped at one fixed size. They start from an initial bound and can grow when repeated executions observe more distinct compatible binding tables than the initial cache budget allowed.

Outside kernel execution, `src/plugin/infer_pipeline.cpp` can also materialize constant graph outputs into synthetic pipeline tensors so output resolution stays uniform even when the public output does not come from a runtime stage.
On Metal, MLIR and stage-policy metadata can also be serialized into a compact MPSRT runtime-model boundary: stage placement is converted into `GfxMpsrtStageDesc`, tensor/storage metadata becomes `GfxMpsrtTensorDesc` or ABI structs, and MSL-dispatch stages carry kernel-family plus external-buffer-role metadata so request-time binding does not rely on a hard-coded argument convention.
The runtime-model reconstruction and resource-table logic lives in the backend-neutral `src/runtime/gfx_mpsrt_model.*` files. Metal-specific files consume that model to prepare pipelines, Apple MPS kernels, resource heaps, and request-time encodes.
That boundary is now manifest-driven rather than ad-hoc. `GfxKernelStageManifest` describes whether a stage is a vendor primitive or a custom kernel, which backend/storage family it belongs to, and what external-buffer ABI the custom kernel expects. The MPSRT adapter layer converts that manifest into the runtime record and buffer-order form used by Metal execution.
The compile path now also emits two generated MLIR helper functions through `gfx_mpsrt_ops.*` and `gfx_mpsrt_runtime_abi_pipeline.*`:
- `gfx_mpsrt_ops`, a typed stage/program facade derived from `GfxMpsrtProgram`
- `gfx_mpsrt_runtime_abi_plan`, a lower-level builder-record/call-plan view used by compile/runtime validation

The runtime-ABI call plan serializes:
- model record keys and record counts
- external-buffer roles and counts
- per-stage ABI descriptors for GEMM, Conv2D, Pool2D, Resize2D, Softmax, TopK, or MSL dispatch
- runtime resource tables that classify external, model-owned, and transient resources before request-time binding
- explicit storage bridges such as `buffer_to_image` and `image_to_buffer`

The Apple stage pipeline in `gfx_apple_stage_pipeline.*` sits earlier in this path. It canonicalizes an Apple-targeted module, materializes placement/storage/fusion metadata, lowers that metadata into the canonical stage manifest, and can then materialize the typed MPSRT program facade from the resulting stage plan. Vendor descriptor extraction for Apple MPS stages is shared through `gfx_apple_vendor_descriptors.*`, so Conv2D, Pool2D, Resize2D, Softmax, and TopK all feed the same descriptor and tensor-desc path.

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

That split is driven by the backend-neutral manifest layer in `gfx_kernel_manifest.hpp` plus the custom-kernel family registry in `gfx_custom_kernel_families.*`. The registry classifies MSL kernels into stable families such as eltwise, transpose/packing, concat/split, gather/scatter, RMS/RoPE, masked softmax-attention, Conv2D/Conv3D, MatMul, pooling, BatchNorm, and reduction dispatch. It also defines the external-buffer ABI roles and dispatch policy expected by request-time binding.
The manifest layer now also carries stage-family information for vendor primitives and custom kernels, semantic input/output roles, and dispatch-grid metadata, which lets one MPSRT model mix execution kinds while keeping one stable record key and buffer-order contract.
Recent Metal compile/runtime changes extend the MPSRT path beyond one standalone MSL stage:
- vendor-only plans such as `MPSGemm` can now execute through the MPSRT runtime boundary without requiring generated MSL source
- hybrid multi-stage plans such as `MPSGemm + MSL epilogue` can be serialized as one MPSRT model with semantic inputs/outputs, explicit intermediate value edges, storage bridges, and stage-local descriptors
- request-time execution can choose full-context MPSRT execution for those mixed models instead of falling back to one raw compiled-kernel dispatch
- vendor primitive coverage now also includes Apple MPS convolution, group convolution, pooling, spatial bilinear Resize2D, softmax, and TopK stages when stage policy selects those routes
- compile-time source planning now uses `gfx_mpsrt_source_plan.hpp` to pick `SingleStage` versus `MultiStage` source contracts directly from the typed program/module metadata, while `gfx_mpsrt_const_tensor_sources.hpp` can attach evaluated constant payloads for vendor convolution-family stages
- metadata cleanup now removes stale flat `gfx.mpsrt.*` stage attrs after the generated program/ops facade is materialized, so current readers should prefer `read_module_mpsrt_program()`, typed `gfx.mpsrt.*` ops, and runtime-ABI helpers over direct attribute scraping
- runtime-model finalization now builds `MpsrtRuntimeResource` entries and `external_buffer_bindings`, then request binding validates external IO, runtime-parameter resources, model-owned const buffers, and prepared transient resources against that table
- prepared MPSRT execution can allocate transient buffer and image resources from one Metal heap and call `makeAliasable` after each resource's live window, so non-overlapping intermediate resources may reuse heap storage

MatMul is the clearest current example of that split:
- plain supported GEMM shapes can lower to vendor `MPSGemm`
- bias or supported activation epilogues can extend that into `MPSGemm + MSL epilogue`
- the final kernel source plan is selected through the `gfx_mpsrt_source_plan` helpers rather than only by a single `entry_point` string

Convolution follows the same direction on the metadata side:
- Apple MPS convolution/group-convolution stages now serialize explicit `GfxMpsrtConv2DAbiDesc` stride, dilation, pad, and grouping metadata
- the custom Metal Conv2D kernel family is also represented through the same manifest path, so legacy MSL `conv2d_kernel` dispatch still shares the common kernel-family and ABI contract
Interpolate follows the vendor-primitive path for the supported static NCHW spatial bilinear cases: the Apple stage pipeline emits a `GfxMpsrtResize2DAbiDesc`, source planning treats it as an IO-only MPS vendor stage, and the Metal runtime prepares/encodes `MPSImageBilinearScale`.
When those Apple MPS image-backed stages connect to public buffer inputs or outputs, the runtime model carries storage-bridge descriptors and the request path materializes the needed bridge resources explicitly rather than assuming one storage class end-to-end.
Metal custom MSL source generation is centralized in MLIR source-plan helpers and `MlirStage` binding-plan annotation. That path covers dynamically shaped or metadata-heavy kernels such as `Softmax`, `Select`, `ScatterUpdate`, `RMS`, `RoPE`, binary `Concat`, rank-4 `ScaledDotProductAttention`, fused causal-mask `ScaledDotProductAttention`, compressed `MatMul`, and slice handling including negative-step `StridedSlice`.
The source-plan files are now split by responsibility: `msl_codegen_apple_msl_*` owns Apple MSL adapter/common/compute/data-movement/dispatch/structural routing, `msl_codegen_apple_mps.*` owns vendor source-plan selection, and `msl_codegen_matmul_metal.*` plus `msl_codegen_matmul_mpsrt.*` own the direct Metal and MPSRT MatMul routes.
`GfxMslRuntimeBindingPlan` converts each manifest role order into the module/runtime operand metadata used by request-time binding. For direct MSL dispatch inside MPSRT, the runtime model must carry materialized `kernel_buffer_order`; request execution rejects an MSL dispatch stage when that order is absent.
SPIR-V compact-ABI annotation is kept separate in `spirv_kernel_binding_adapter.hpp`, so Vulkan fixed-argument kernels do not inherit legacy Apple MSL operand or scalar attrs.
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
- contains specialized execution routes for chunked unary/binary/softmax/layout ops and for multiple Conv2D or GroupConv2D cases
- supports direct handling of some common binary patterns such as same-shape and bias-add style cases
- reuses immutable constant buffers and prepared descriptor bindings across compatible submissions
- batches pending constant-buffer uploads before the main infer recording path begins
- increases per-submit batching thresholds in the infer path to reduce Android-oriented driver overhead
- persists Vulkan pipeline-cache data under `ov::cache_dir` when a cache directory is supplied through standard OpenVINO properties
- reports execution-device limits through `GpuExecutionDeviceInfo`, matching the Metal path and removing backend-specific probing from shared planning code
- recompiles selected specialized Conv2D and GroupConv2D kernels when the chosen dispatch workgroup shape changes, so launch shape and kernel metadata stay aligned
- prefers SPIR-V-observed binding counts when reconciling compiled-kernel argument metadata, reducing drift between MLIR-side arg inference and final Vulkan shader bindings
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
- backend parity is not guaranteed
- many ops still require static rank, static shape, or constant attributes
- the plugin remains experimental
- some lowering and runtime optimizations are intentionally backend-specific, so architecture docs should describe the shared model first and backend specialization second
