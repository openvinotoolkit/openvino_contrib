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
- `src/mlir/`: MLIR support probing and lowering helpers
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

The compiled pipeline is represented as `PipelineStageDesc` entries that wrap backend-specific `GpuStage` objects.

### `ov::gfx_plugin::InferRequest`
Declared in `include/openvino/gfx_plugin/infer_request.hpp` and implemented by backend-specific infer paths.

Responsibilities:
- bind host tensors or remote tensors
- create per-request backend state
- wire the compiled stage pipeline to actual buffers
- execute the pipeline and collect profiling data

The infer path now has an explicit submission layer. `src/plugin/infer_submission.*` abstracts command-buffer recording and submission windows, while `src/plugin/infer_pipeline.*` can pre-resolve reusable stage-input bindings, reusable output bindings, and reusable output handles for repeated requests.
`src/plugin/stateful_execution.*` adds a second interception layer for stateful graphs: `ReadValue` can source a persisted variable tensor from infer-request state, while `Assign` copies the latest value into a persistent backend buffer owned by that request.

## Pipeline Model
The runtime is stage-based.

`src/runtime/gpu_stage.hpp` defines the backend-neutral execution interface. Each backend registers a stage factory through `GpuStageFactory` in `src/runtime/execution_dispatcher.*`. During compilation, the plugin translates OpenVINO nodes into stage descriptors. During inference, these descriptors are cloned, connected to inputs and outputs, and executed in order.

The pipeline may fuse multiple nodes into a `FusedSequenceStage` when fusion logic decides that a combined execution path is valid.

Two interface points are important in the current code:
- `set_input_transform()`: a stage can consume metadata about an absorbed upstream transpose instead of requiring a separate materialized transpose stage
- `submit_policy()`: a stage reports scheduling weight and isolation hints used by backend infer paths during command-buffer submission

`src/runtime/gfx_stage_policy.*` selects runtime policy from node shape, element type, backend, and backend capabilities. The policy layer currently decides:
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

The reusable infer pipeline layer now has two precomputed plans:
- `PreparedInferExecutionPlan` for stage inputs that resolve to parameters or previous stage outputs
- `PreparedInferOutputPlan` for public outputs that resolve to stage outputs, passthrough parameters, or synthetic pipeline entries created for constant outputs

When output shape and element type are statically known, infer requests may also keep reusable host output tensors in backend infer state to avoid recreating host tensors on each execution.
If a reusable host output no longer matches the expected static type or shape, the infer path now recreates it instead of treating that mismatch as a hard internal error.

Stage-output allocation also has a liveness-aware workspace layer:
- `StageOutputBufferWorkspace` can recycle intermediate output buffers across stages when the output lifetime does not escape the current live range
- `GpuStage::describe_output_lifetimes()` lets complex stages such as `FusedSequenceStage` report finer-grained internal lifetimes
- infer requests report workspace allocation counters such as slots used and peak live slots through the normal profiling path

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
- dedicated builders now exist for `RMS` and `ScatterUpdate` instead of relying only on more generic lowering families
- fusion now also supports selected input-side activations for `Multiply`, not only output activations or bias/post-op forms
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

Outside kernel execution, `src/plugin/infer_pipeline.cpp` can also materialize constant graph outputs into synthetic pipeline tensors so output resolution stays uniform even when the public output does not come from a runtime stage.

## Profiling Stack
Profiling is split into compile-time and infer-time collection.

- `src/runtime/gfx_compile_profiling.hpp` provides thread-local compile scopes used while `CompiledModel` creates backend state, builds stages, and compiles kernels
- `src/runtime/gfx_profiling_report.*` owns the JSON-ready report model for nodes, segments, transfers, allocations, and counters
- backend profilers under `src/backends/*/runtime/profiling/` populate runtime timing and backend-specific counters
- `src/plugin/gfx_profiling_utils.hpp` merges compile and infer reports into the final `GFX_PROFILING_REPORT` JSON and can optionally emit Perfetto-style trace payloads

When profiling is disabled, these paths stay out of the fast path.

## Metal Backend
`src/backends/metal/` contains:
- `plugin/`: backend state creation, infer request, remote context, remote tensor
- `runtime/`: stage implementations, memory allocators, executor, profiler
- `codegen/`: Metal shader compilation helpers

Direct Metal API usage lives in Objective-C++ files (`.mm`).

The current Metal backend also shares immutable constant-buffer state across compatible requests through `MetalConstCache` and the backend-neutral immutable buffer cache helper.
It now also reports execution-device limits through `GpuExecutionDeviceInfo`, so backend-neutral planning code can use real Metal subgroup and threadgroup limits.
The current Metal executor also contains runtime-specialized codegen paths for dynamically shaped `Softmax`, `Select`, `ScatterUpdate`, `RMS`, binary `Concat`, rank-4 `ScaledDotProductAttention`, and more permissive slice handling, including negative-step `StridedSlice`.
For dynamic-shape `MatMul`, the current MLIR stage path may also pack a constant RHS from `f32` to `f16` before wrapping it as an immutable const buffer. Compile profiling counters track the original and packed byte sizes for that optimization.
The backend-aware transform pipeline is also important for Metal now: compressed `MatMul` decompression subgraphs can be marked as decompression and protected from generic folding so later stage compilation can still recognize weight-only compressed patterns.

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
