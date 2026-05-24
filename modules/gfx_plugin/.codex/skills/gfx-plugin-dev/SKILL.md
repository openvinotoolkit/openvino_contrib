---
name: gfx-plugin-dev
description: Use when working on OpenVINO GFX plugin code in modules/gfx_plugin, especially MLIR lowering, runtime stages, backend routing, plugin properties, or architecture-sensitive refactors across Metal, OpenCL, and Vulkan backends.
---

# GFX Plugin Development

This skill is for implementing or refactoring the `GFX` OpenVINO plugin in `modules/gfx_plugin/`.

## Use This Skill When

- The task touches `src/`, `include/`, `tests/`, `CMakeLists.txt`, or backend-specific code.
- The user asks for architecture changes, new ops, runtime fixes, property behavior, backend routing, or MLIR/codegen work.
- The task mentions Metal, OpenCL, Vulkan, `GFX`, `CompiledModel`, `InferRequest`, remote tensors, or MLIR lowering.

## Primary References

Read these first:

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/DEVELOPMENT.md`
4. `docs/TESTING.md`

Then read the relevant code path:

- Plugin contract: `src/plugin/`
- Backend-neutral runtime: `src/runtime/`
- MLIR builders and stage planning: `src/mlir/`
- Parallel and graph rewrites: `src/transforms/`
- Metal backend: `src/backends/metal/`
- OpenCL backend: `src/backends/opencl/`
- Vulkan backend: `src/backends/vulkan/`
- Public API surface: `include/openvino/gfx_plugin/`

## Development Rules

- Treat local module docs as the published source of truth.
- Reuse existing OpenVINO plugin patterns; do not invent unrelated abstractions.
- Keep shared logic backend-neutral unless the behavior is truly backend-specific.
- Prefer extending `gfx_parallelism.*`, `gfx_partitioning.*`, `gfx_stage_policy.*`, or shared caches before adding new special-purpose plumbing.
- For MLIR changes, keep compile-time support probing, lowering, and runtime behavior aligned.
- When changing plugin-visible behavior, also check properties, `query_model()`, and compiled-model/runtime property exposure.
- When changing stateful graph behavior, treat `ReadValue` / `Assign` as a dedicated infer-request-state path, not just generic stateless runtime stages.
- For Metal placement work, keep `gfx_stage_policy.*`, `gfx_mpsrt_*`, `src/runtime/gfx_mpsrt_model.*`, `gfx_kernel_manifest.hpp`, `gfx_custom_kernel_families.*`, MLIR attrs, and `src/backends/metal/runtime/mpsrt/*` aligned as one contract.
- For hybrid Metal paths, also keep `gfx_kernel_manifest.hpp`, `gfx_custom_kernel_families.*`, `gfx_mpsrt_program.hpp`, `gfx_mpsrt_dialect.*`, `gfx_mpsrt_ops.*`, `gfx_apple_stage_pipeline.*`, `gfx_mpsrt_kernel_manifest_adapter.hpp`, `gfx_mpsrt_storage_bridge.hpp`, `gfx_mpsrt_source_plan.hpp`, `gfx_backend_custom_kernel_adapter.*`, `gfx_stage_kernel_binding.hpp`, and `gfx_stage_runtime_values.*` aligned with that contract.
- For Apple MPS/MPSGraph vendor primitive changes, prefer `src/mlir/gfx_apple_vendor_descriptors.*`, `GfxAppleMpsVendorPrimitiveContract`, and `materialize_apple_mps_vendor_contract_program()` so Conv2D, Pool2D, Resize2D, Softmax, TopK, GEMM, and SDPA descriptor extraction stays shared.
- For MPSRT request-binding changes, keep `MpsrtRuntimeResource`, `external_buffer_bindings`, prepared model resources, storage bridges, and request-time validation aligned instead of reintroducing ad-hoc transient allocation.
- For Metal custom MSL source changes, prefer `src/mlir/msl_codegen_apple_msl*`, `src/mlir/msl_codegen_apple_mps.*`, `src/mlir/msl_codegen_matmul_*`, and `GfxMslRuntimeBindingPlan`; keep module operand annotations, manifest external-buffer roles, inferred `[[buffer(N)]]` counts, and MPSRT `kernel_buffer_order` aligned.
- For Vulkan compact-ABI changes, prefer `src/mlir/spirv_kernel_binding_adapter.hpp` so SPIR-V fixed-argument metadata stays separate from Apple MSL binding attrs.
- For Vulkan Conv2D dispatch tuning, keep output-channel blocking and spatial micro-tiling in `gfx_parallelism.*`, `gfx.dispatch_channel_block` metadata, and shared convolution lowering; do not add executor-local Conv variants or device-name tables.
- For OpenCL source-kernel work, keep source artifacts in `src/kernel_ir/gfx_opencl_source_artifacts.*` and runtime execution in `src/backends/opencl/runtime/opencl_source_stage.*`; do not scatter source ids, scalar ABI, local sizes, element-count rules, runtime-shape scalars, constant materialization, or boolean-buffer packing through infer-request code.
- For target-device profiling changes, keep `GpuExecutionDeviceInfo`, `gfx_target_profile.*`, backend buffer managers, and `GFX_PROFILING_REPORT` JSON aligned.

## Common Workflows

### New or changed op path

1. Update support probing in `src/mlir/`.
2. Update lowering or specialized path in `src/mlir/` or `src/transforms/`.
3. Update backend runtime handling if needed.
4. Add or update focused unit tests.
5. Update docs if supported shapes, route selection, or backend behavior changed.

Check whether the change belongs to one of the current special families:

- dynamic-shape data movement and shape ops such as `ShapeOf`, `Concat`, `Broadcast`, `Select`, `StridedSlice`, and `Range`
- dedicated lowered ops such as `RMS`, `ScatterUpdate`, and `RoPE`
- stateful `ReadValue` / `Assign` handling through infer-request variable storage
- source-node-aware output routing for fused stages or direct stateful-assign prebinding
- output aliases or storage-source reuse inside `FusedSequenceStage` and infer output planning
- backend-specialized launch paths that now depend on final runtime shape or final shader binding counts
- backend-only fused LLM ops such as `GfxSDPAWithCausalMask` and vendor attention groups routed to MPSGraph-backed `MPSSdpa`
- Metal-native op contracts that now carry more ABI metadata, such as dilated MaxPool, generalized TopK, ShapeOf, or blocked Conv2D dispatch
- Metal placement-domain and storage selection, such as Apple MPS image or matrix stages versus Apple MSL buffer dispatch
- MPSRT runtime-model boundaries, including tensor descriptors, runtime resources, stage record keys, external-buffer roles, and prepared MSL-dispatch pipeline caching
- shared MPSRT runtime-model reconstruction in `src/runtime/gfx_mpsrt_model.*`, including resource finalization, external tensor binding plans, and external-buffer ABI adaptation
- typed MPSRT builder-plan/runtime-model records, storage bridges, resource tables, prepared Metal heaps, and const-tensor-source attachment for Apple MPS models
- typed `GfxMpsrtProgram` validation and generated `gfx_mpsrt_ops` materialization, including cleanup of stale legacy attrs
- Apple stage-pipeline passes, shared vendor descriptors, and typed storage-conversion ops for image, matrix, ndarray, or alias boundaries
- manifest-backed execution-kind routing, including vendor-only stages such as MPS Resize2D, MPSGraph SDPA, and mixed vendor-plus-custom multi-stage plans
- custom-kernel family classification, external-buffer ABI roles, semantic input/output roles, and dispatch-grid policy in `src/kernel_ir/gfx_custom_kernel_families.*`
- Metal MSL runtime binding plans for tensor inputs, tensor outputs, const tensors, scalar params, and runtime params
- MLIR-owned Metal MSL source plans such as compressed `MatMul`, SDPA, causal SDPA, Apple MSL binding/dispatch/op-family kernels, Apple MPS/MPSGraph vendor plans, and direct/MPSRT MatMul helpers
- SPIR-V compact-ABI adapter metadata for fixed-argument Vulkan kernels
- compile-time data repacking paths, such as Metal dynamic-shape `MatMul` packing a constant RHS from `f32` to `f16` and recompiling against the effective runtime tensor types
- backend-aware transform preservation, such as keeping compressed `MatMul` decompression subgraphs intact for Metal-only downstream routes
- backend-aware transform fusion, such as LLaMA rotate-half rewriting into native `RoPE` on Metal or compatible compressed `MatMul` nodes regrouping into a fused horizontal path
- backend-side fused epilogues, such as Metal `RMS` absorbing a residual `Add`
- input-side fusion paths, such as `Multiply` absorbing an activation on one selected input instead of only post-op activation on the output
- public `ov::hint::inference_precision` handling and precision-aware accuracy expectations
- Vulkan Conv2D output-channel blocking, including `GpuExecutionDeviceInfo` capability flags, dispatch metadata, SPIR-V cache metadata, and shared lowering support
- OpenCL source-artifact manifests for baseline f32/f16 data movement, typed casts, MatMul/Softmax, Range/Tile, gather/scatter, elementwise, logical, and logical-reduction kernels
- dynamic runtime-shape planning for OpenCL `Concat`, `Broadcast`, `Select`, `ShapeOf`, `Slice` / `StridedSlice`, and `Range`
- target profile recording through `extended.target_profile` and `target_backend_*` counters

### Runtime or backend scheduling change

1. Inspect `src/runtime/gfx_stage_policy.*`, `gfx_parallelism.*`, and `gfx_partitioning.*`.
2. Check whether the change is backend-neutral or family-specific.
3. Verify interaction with infer submission, immutable const caches, and prepared binding reuse when applicable.
4. Add tests in `tests/unit/` and backend tests when behavior is externally visible.
5. On Metal, inspect `src/backends/metal/runtime/metal_command_encoder.*` before adding new encoder/pipeline/buffer binding logic.
6. If the change affects Apple placement or request-time Metal binding, also inspect `src/runtime/gfx_mpsrt_model.*`, `src/backends/metal/runtime/mpsrt/*`, `src/kernel_ir/gfx_kernel_manifest.hpp`, `src/kernel_ir/gfx_custom_kernel_families.*`, and the split `src/mlir/msl_codegen_apple_*` / `src/mlir/msl_codegen_matmul_*` files.
7. For infer submission changes, keep direct producer-consumer dependency extension bounded by common stage/output/MAC budgets and keep layout, split, transpose, softmax, and attention stages as hard extension boundaries.

### OpenCL source-artifact change

1. Inspect `src/kernel_ir/gfx_opencl_source_artifacts.*` first; the artifact manifest is the source of truth for source id, entry point, role ABI, scalar ABI, element-count source, local size, and dynamic shape scalar metadata.
2. Keep `src/backends/opencl/runtime/opencl_source_stage.*` as a generic artifact executor. New op-specific behavior should reach it through artifact metadata, constant materialization, or shared runtime-value planners, not local runtime branches.
3. Check `src/plugin/infer_pipeline.*` when dynamic output shapes or runtime input metadata affect OpenCL stage allocation before execution.
4. Check `src/backends/opencl/runtime/opencl_api.*` and `opencl_buffer_manager.*` only when device selection, dynamic loading, memory ops, aligned allocation, or target-profile reporting changed.
5. Update `tests/unit/gfx_opencl_source_artifacts_test.cpp` and docs when the supported OpenCL subset changes.
6. Treat standalone OpenCL Conv2D microbench tools as experiments until the route is promoted into the plugin manifest/runtime/test contract.

### Stateful or reusable infer-path change

1. Inspect `src/plugin/infer_request_state.hpp`, `src/plugin/infer_pipeline.*`, `src/plugin/infer_io_utils.*`, and `src/plugin/stateful_execution.*`.
2. Keep variable-buffer lifetime, reusable host-output lifetime, and stage-output shape/type recovery aligned.
3. Treat `ReadValue` as a view-style stage and `Assign` as a persisted copy/update path unless the code explicitly changes that contract.
4. When output allocation changed, also inspect `StageOutputBufferWorkspace` and `GpuStage::describe_output_lifetimes()` so liveness-based reuse and profiling counters stay coherent.
5. For split/view-style output changes, also inspect contiguous `Split` / `VariadicSplit` aliasing in `src/mlir/mlir_stage.cpp` so byte-range views stay valid.
6. When one pipeline stage serves multiple graph outputs, keep `output_sources` and `output_aliases` aligned so public outputs, compare tooling, and stateful consumers still resolve the right node/port.

### Property or device-selection change

1. Update property parsing and lists in `src/plugin/`.
2. Check `ov::available_devices`, `ov::device::id`, `ov::device::full_name`, and cache-related properties.
3. Add or update `tests/unit/plugin_tests.cpp`.
4. Update `README.md` and `docs/USAGE.md` if the public contract changed.

## What To Avoid

- Do not document removed architectures as current behavior.
- Do not leave backend-specific fast-path changes undocumented when they affect supported shapes, dispatch rules, or profiling semantics.
- Do not add public-facing docs outside `modules/gfx_plugin/`.
- Do not modify `third_party/llvm-project/` unless the task explicitly requires vendored LLVM changes.

## Output Expectations

- Keep changes focused by subsystem.
- If the task changes architecture, runtime semantics, or public properties, update local docs in the same pass.
- If the task reaches commit/push stage, also apply the same plugin change set to the mirrored `ov-ext-labs/gfx-plugin` repository unless the user explicitly says otherwise.
