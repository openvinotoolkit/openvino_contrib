# Development Guide

This document is for contributors working inside `modules/gfx_plugin`.

## Prerequisites
- CMake 3.13 or newer
- an OpenVINO Developer Package build
- Ninja recommended
- Metal toolchain on macOS for the Metal backend
- Vulkan SDK or system Vulkan development files for the Vulkan backend

The module vendors LLVM/MLIR under `third_party/llvm-project` and can build the required MLIR pieces as part of the CMake flow.
`third_party/Vulkan-Headers` is tracked separately as a git submodule for the Raspberry Pi Vulkan flow.

## Configure And Build
Example configuration:

```bash
cd /path/to/gfx_plugin
cmake -S . -B build-gfx-plugin -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINODeveloperPackage_DIR=/path/to/openvino/install/cmake \
  -DENABLE_TESTS=ON \
  -DGFX_DEFAULT_BACKEND=auto
```

Build:

```bash
cmake --build build-gfx-plugin --target openvino_gfx_plugin ov_gfx_func_tests
cmake --build build-gfx-plugin --target ov_gfx_unit_tests ov_gfx_runtime_micro_tests ov_gfx_microbench
```

Useful CMake options:
- `GFX_ENABLE_METAL`
- `GFX_ENABLE_VULKAN`
- `GFX_DEFAULT_BACKEND`
- `ENABLE_TESTS`

On macOS, Vulkan is disabled by `cmake/GfxBackendConfig.cmake`.

Current build-system notes:
- vendored LLVM/MLIR is built as a static external toolchain under `third_party/llvm-project`
- the external LLVM bootstrap injects a tiny local dummy fuzzing-engine archive so the bundled llvmorg-22.1.2 MLIR configure path does not fail in `mlir-parser-fuzzer`
- the module now reuses installed package exports when present and otherwise falls back to build-tree exports during bootstrap
- Android and generic cross-compiling flows forward toolchain settings into the vendored LLVM/MLIR configure step
- that bootstrap path also forwards `CMAKE_C_FLAGS`, `CMAKE_CXX_FLAGS`, and the executable/shared/module linker flag families into the nested LLVM configure step
- Apple MSL source planning now builds with `gfx_runtime_mlir`; keep target wiring in `cmake/GfxSources.cmake` and `src/CMakeLists.txt` aligned when adding `msl_codegen_apple_*` or `msl_codegen_matmul_*` files
- the module build treats warnings as errors by default through `-Werror` on Clang/GCC and `/WX` on MSVC
- `cmake/GfxAndroidRuntimeBundle.cmake.in` is used to copy Android runtime-side dependencies next to deployed plugin artifacts

## Hermetic RPi Vulkan Toolchain
To build a Raspberry Pi Vulkan cross-toolchain bundle for this module, use:

```bash
python3 modules/gfx_plugin/tools/gfx_rpi_vulkan_toolchain_builder.py \
  --output-dir /path/to/build-gfx-plugin-rpi-toolchain
```

The script builds host LLVM tools from the vendored
`modules/gfx_plugin/third_party/llvm-project`, assembles a generic
Raspberry Pi 4/5 Bookworm arm64 sysroot, normalizes absolute sysroot symlinks,
copies both `vulkan/` and `vk_video/` headers from the vendored
`modules/gfx_plugin/third_party/Vulkan-Headers` tree, and generates:

```text
/path/to/build-gfx-plugin-rpi-toolchain/cmake/gfx-rpi-vulkan-aarch64.toolchain.cmake
```

Use that generated file directly with a normal CMake configure command:

```bash
cmake -S /path/to/openvino -B /path/to/build-gfx-plugin-rpi \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/build-gfx-plugin-rpi-toolchain/cmake/gfx-rpi-vulkan-aarch64.toolchain.cmake \
  -DOPENVINO_EXTRA_MODULES=/path/to/openvino_contrib/modules/gfx_plugin \
  -DGFX_ENABLE_METAL=OFF \
  -DGFX_ENABLE_VULKAN=ON \
  -DGFX_DEFAULT_BACKEND=vulkan
```

The builder is host-generic and is intended to work on Linux, macOS, and
Windows. If needed, you can override the generic sysroot source with:
- `--sysroot-dir /path/to/extracted-rootfs`
- `--sysroot-tarball /path/to/rootfs.tar.xz`

The generated compiler wrappers and toolchain file now use portable
`-march=armv8-a` flags instead of a Raspberry Pi 5-only `-mcpu=` default, so
the same bundle can be reused more safely across Raspberry Pi 4/5 class Vulkan
targets.

Before running the builder, prepare the Vulkan-Headers submodule dependency in:

```text
modules/gfx_plugin/third_party/Vulkan-Headers
```

Initialize that dependency with git submodules, for example:

```bash
git submodule update --init modules/gfx_plugin/third_party/Vulkan-Headers
```

The toolchain builder expects that submodule to be checked out at the upstream
`Vulkan-Headers` release currently pinned by the module for Raspberry Pi 4/5
Bookworm compatibility: `v1.3.239`.

## Where To Start Reading
Read in this order:
1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `src/plugin/plugin.cpp`
4. `src/plugin/compiled_model.cpp`
5. `include/openvino/gfx_plugin/infer_request.hpp`
6. the backend directory you are changing

For runtime execution, follow:
- `Plugin::compile_model()`
- `CompiledModel::build_op_pipeline()`
- backend-specific `infer_*_impl()`
- backend stage factory and executor code

If the behavior depends on route or scheduling selection, also read:
- `src/runtime/gfx_stage_policy.*`
- `src/runtime/gfx_parallelism.*`
- `src/runtime/gfx_mpsrt_abi.hpp`
- `src/runtime/gfx_mpsrt_plan.hpp`
- `src/runtime/gfx_mpsrt_builder_plan.hpp`
- `src/runtime/gfx_mpsrt_model.*`
- `src/runtime/gfx_mpsrt_program.hpp`
- `src/runtime/gfx_mpsrt_storage_bridge.hpp`
- `src/kernel_ir/gfx_kernel_manifest.hpp`
- `src/kernel_ir/gfx_custom_kernel_families.*`
- `src/runtime/gfx_mpsrt_kernel_manifest_adapter.hpp`
- `src/mlir/gfx_apple_vendor_descriptors.*`
- `src/mlir/gfx_apple_stage_pipeline.*`
- `src/mlir/gfx_mpsrt_dialect.*`
- `src/mlir/gfx_mpsrt_ops.*`
- `src/mlir/gfx_mpsrt_source_plan.hpp`
- `src/mlir/gfx_mpsrt_const_tensor_sources.hpp`
- `src/mlir/gfx_kernel_runtime_params.hpp`
- `src/mlir/msl_codegen.hpp`
- `src/mlir/msl_codegen_apple_msl*.{cpp,hpp}`
- `src/mlir/msl_codegen_apple_mps.*`
- `src/mlir/msl_codegen_matmul_metal.*`
- `src/mlir/msl_codegen_matmul_mpsrt.*`
- `src/mlir/msl_codegen_attention.*`
- `src/mlir/msl_codegen_compressed_matmul.*`
- `src/mlir/spirv_kernel_binding_adapter.hpp`

`tests/unit/gfx_parallelism_test.cpp` now covers Broadcom-oriented matmul
selection plus dense stride-1, huge-spatial, and ultra-dense convolution
threadgroup decisions.
- `src/runtime/gfx_partitioning.*`
- `src/runtime/gpu_buffer_manager.hpp` for `GpuDeviceFamily` and backend-reported execution-device info
- the active backend executor, especially under `src/backends/vulkan/runtime/`
- `src/runtime/gfx_profiling_report.*` when the change affects counters, trace sinks, or JSON report shape

The current planning path is no longer just backend-wide. It includes family-specific tuning hooks, especially for:
- Broadcom V3D Vulkan devices
- Qualcomm Adreno Vulkan devices
- explicit convolution dispatch attrs forwarded into MLIR lowering
- Metal placement decisions between Apple MPS image or matrix primitives and Apple MSL buffer dispatch
- manifest-backed execution-kind selection between vendor primitives and custom kernels
- custom-kernel family classification, external-buffer ABI roles, semantic input/output roles, and dispatch policies from `src/kernel_ir/gfx_custom_kernel_families.*`
- typed MPSRT builder-plan records, resource tables, and storage bridges that must stay aligned with request-time binding and stage reconstruction
- backend-neutral MPSRT runtime-model reconstruction in `src/runtime/gfx_mpsrt_model.*`, which now owns resource finalization, tensor binding plans, and external-buffer ABI adaptation before Metal consumes the model
- the typed `GfxMpsrtProgram` facade and generated `gfx_mpsrt_ops` materialization, which now sit between legacy module attrs and final builder-plan/runtime-model materialization
- the Apple stage pipeline, shared vendor descriptor helpers, and typed `gfx.mpsrt` helper ops, which now mediate descriptor extraction, storage assignment, and program materialization for Apple MPS routes
- MPSRT const payload materialization from typed program input descriptors: vendor stages and typed model-owned resources should attach const data for inputs marked as `ConstTensor`/`GfxMpsrtTensorFlagConst`, not by checking individual OpenVINO op types such as Conv2D
- Metal runtime preparation must consume model-owned const tensors through the prepared-resource table (`MpsrtPreparedResource`), leaving the const-pack cache as the upload/backing layer rather than a stage-local lookup API
- Single-stage Apple MSL dispatch keeps `ConstTensor` and runtime-parameter buffers in the external `KernelBindingPlan` ABI unless a typed multi-stage/vendor plan explicitly owns the payload; for that path `ConstTensor` means immutable kernel argument semantics, not model-owned lifetime

For current convolution work, there are now two important lowering details to keep in mind:
- full interior tiles in conv parallel lowering can skip lane-level bounds guards on the fast path
- the interior-window decision is now split into reusable height and width checks before the combined 2D helper decides whether the tile is fully interior
- Vulkan specialized kernel compilation may re-resolve effective argument count from final SPIR-V bindings instead of trusting only pre-SPIR-V metadata
- plain Vulkan Conv2D and GroupConv now stay on the shared canonical MLIR builders plus the common convolution lowering pipeline; the older manual `gpu.func` convolution builders and executor-level direct/chunked convolution routes were deleted so correctness, binding ABI, and dispatch metadata flow through one MLIR route. Keep rank-4 Conv2D/GroupConv const weights and packed runtime payloads on the shared `MlirStage` extra-input helper for both compile-time and request-time paths; Vulkan request-time refresh may rebuild the buffers, but must not overwrite the final SPIR-V binding metadata with the pre-lowering source binding.
- Broadcom V3D pointwise/light-reduction Conv2D tuning belongs in `gfx_parallelism.*` and must flow through the same MLIR dispatch attrs as other convolution plans. For huge-spatial 1x1-style workloads, the V3D policy keeps occupancy headroom instead of choosing the full hardware-cap `16x16`/256-thread tile: pointwise/light-reduction Conv uses a 64-thread target, and ultra-dense Conv is capped at 128 threads per group. Do not implement this as a Vulkan executor shortcut or a separate Conv2D route.
- Decomposed Conv2D algorithms such as `im2col + matmul + restore` are not a valid replacement for a single custom-kernel stage until the planner/runtime has a typed subgraph or multi-kernel manifest that owns all intermediate buffers and launches. A device policy must not select such a decomposition through `gfx.conv_algorithm_kind` if the final `KernelSource` still has one entry point and one request-time binding plan; that silently compiles only part of the decomposition and breaks accuracy.
- MaxPool2D and AvgPool2D custom-kernel builders must also stay on the shared MLIR dispatch contract. They emit `scf.parallel` over channel/tiled-spatial/thread lanes with explicit `gfx.dispatch_tile_*`, `gfx.dispatch_threads_*`, and `gfx.parallel_loop_dims` attrs; do not reintroduce serial `scf.for` Pool2D builders that force Vulkan/SPIR-V into `grid=(1,1,1)` single-dispatch execution.
- Conv3D custom kernels use the common `GfxKernelStageManifest` ABI on every backend that executes them. Keep the role order as `TensorInput`, `ConstTensor`, `TensorOutput`, `RuntimeParams`; `MlirStage` must materialize the const weights and packed runtime params as extra buffers in that order, annotate rank-5 convolution modules as the `conv3d` custom-kernel family before `KernelSource` creation, and derive source signatures from that exact manifest on Metal and Vulkan. Do not add Conv3D-only binding shortcuts, do not let the Conv2D direct/im2col manifest leak into Conv3D, and do not revive `gfx.fixed_arg_count` for this path.

Infer submission parallelism follows the same hierarchy as kernel dispatch
parallelism: first derive a common workload profile from pipeline depth, SIMD
width, backend, and threadgroup limits; then apply a narrow device-family
profile from `GpuDeviceFamily`. Device profiles may scale MAC budgets or slot
capacity, but they must not create a separate executor path, test skip, runtime
define, or CPU fallback. For example, Broadcom V3D uses the same constrained
Vulkan stage/output window as generic constrained Vulkan, then only lowers the
MAC budget coefficient for Conv-heavy windows.

If the change touches infer-request throughput or resource reuse, also read:
- `src/plugin/infer_submission.*`
- `src/plugin/infer_pipeline.*`
- `src/plugin/stateful_execution.*`
- `src/plugin/stateful_stage.*`
- `src/runtime/immutable_gpu_buffer_cache.*`
- `src/runtime/gpu_backend_base.hpp`
- `src/runtime/gpu_buffer_manager.hpp`
- `src/plugin/infer_io_utils.*`
- `src/backends/vulkan/runtime/vulkan_buffer_manager.*` when Vulkan const-upload batching or shared upload-recording behavior changes
- `src/runtime/gfx_mpsrt_model.*` and `src/backends/metal/runtime/mpsrt/*` when Metal request-time binding, prepared dispatch caching, or external-buffer ABI rules changed

For stage-output reuse changes, also inspect:
- `StageOutputBufferWorkspace` in `src/plugin/infer_pipeline.hpp`
- `GpuStage::describe_output_lifetimes()` in `src/runtime/gpu_stage.hpp`
- `src/runtime/fused_sequence_stage.*` for fused-stage lifetime propagation

If the change touches stateful graphs, read `src/plugin/infer_request_state.hpp` as well. `ReadValue` and `Assign` are no longer just generic ops flowing through the normal stateless path: infer-request state now owns persistent variable buffers keyed by OpenVINO variable id.
If the change touches fused stages or rewritten outputs, also inspect `PipelineStageDesc::output_aliases` plus the source-node/source-port plumbing in `src/plugin/infer_pipeline.*`. Public outputs and direct stateful-assign consumers no longer assume a simple `stage.node/output_port` mapping.

## Adding Or Extending An Op
Typical path:
1. Add or extend MLIR support in `src/mlir/`
2. Ensure support probing succeeds through `mlir_supports_node()`
3. Implement or update backend runtime/codegen handling
4. Make sure the relevant backend stage can be created
5. Add tests in `tests/unit/` and the relevant backend test directory

For the current codebase, also check whether the op should participate in:
- absorbed input transforms through `GfxInputTransform`
- stage policy selection in `src/runtime/gfx_stage_policy.*`
- MLIR parallel lowering, cleanup passes, or public-signature rewrites in the pass pipeline
- backend-specialized fast paths such as Vulkan direct or chunked routes; do not add new Conv2D or GroupConv routes outside the shared MLIR/SPIR-V custom-kernel path
- reusable bytes-arg materialization or immutable const-buffer caching
- stateful execution interception through `src/plugin/stateful_execution.*` when the graph uses `ReadValue` / `Assign`

For Reduce-like work, prefer the existing typed reduction extraction in `src/mlir/mlir_builder_reduce.cpp`. The current builder reads axes and `keep_dims` from concrete Reduce op classes such as `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`, `ReduceProd`, `ReduceL1`, and `ReduceL2` instead of relying on a looser generic reduction-base path.

For `MatMul`, keep the compile-time const-buffer story aligned with runtime codegen. The current Metal path may repack a constant RHS from `f32` to `f16` for dynamic-shape `MatMul` stages and then derive kernel input element types from the effective runtime tensors instead of only the original node input types.
If the change touches compressed or quantized `MatMul` weights, also inspect `src/transforms/pipeline.cpp` and `src/mlir/msl_codegen_compressed_matmul.cpp`. The transform pipeline is now backend-aware and can protect decompression subgraphs on Metal so backend-specific compressed-weight handling survives generic optimization passes. It can also horizontally regroup compatible compressed `MatMul` nodes that share one data input into a fused `MatMul` plus `VariadicSplit`, and stage compilation may repack concatenated quantized weight/scales into backend const buffers before selecting the compressed-MatMul MSL source and binding plan.

For RMSNorm-style work, remember that `src/transforms/pipeline.cpp` now runs OpenVINO `RMSFusion` before plugin-local cleanup. The current intended path is fused RMSNorm graph patterns lowering into the dedicated `RMS` builder and backend codegen, not preserving only the unfused arithmetic tail.
On Metal, also account for residual-add fusion into `RMS`: `CompiledModel` can detect a directly attached `Add`, route both add inputs plus gamma into one stage, and mark the MLIR/codegen path with `gfx.fused_residual_add`.

For `ScatterUpdate`, use the dedicated builder in `src/mlir/mlir_builder_scatter_update.cpp`. The current path expects a constant scalar `axis`, static ranks, and normalized negative indices in the generated kernel path.

For `RoPE`, use the dedicated builder in `src/mlir/mlir_builder_rope.cpp` and the Metal codegen path in `src/mlir/rope_codegen.cpp`. The current native Metal route expects rank-3 or rank-4 data, rank-2/3/4 cos/sin inputs, no `input_trans0213` / `output_trans0213`, no ChatGLM or Qwen-special layouts, and no sliced input layout. Supported LLaMA rotate-half arithmetic may be rewritten into native `RoPE` in `src/transforms/pipeline.cpp` before stage compilation.

For fusion work, note that current support is no longer limited to output post-ops. `Multiply` can now absorb selected activations into one chosen input through the fusion plan and backend `fuse_input_activation()` hooks. Keep the transform-side fusion pattern, compiled-model fusion bookkeeping, and backend runtime/codegen support aligned.

For `ScaledDotProductAttention`, the current native path is backend-specific: Metal can keep a rank-4 FP16/FP32 SDPA node and either compile a custom MSL attention kernel through `src/mlir/msl_codegen_attention.*` or route an accepted unmasked/non-causal rank-4 Q/K/V contract through an MPSGraph-backed `MPSSdpa` MPSRT stage. Vulkan still rejects native SDPA and expects other lowering paths. `src/transforms/gfx_llm_ops.hpp` also defines a Metal-only `GfxSDPAWithCausalMask` op used for selected LLM graphs when `src/transforms/pipeline.cpp` can recover explicit `attention_mask[B,K]` and `cache_position[Q]` inputs from the original mask subgraph. That fusion may also peel some broadcast-expanded GQA K/V views back to compact K/V tensors before stage compilation. Vendor attention fusion groups named `VendorAttention` must remain a typed MPSRT/MPSGraph route with explicit Q/K/V external bindings, not a CPU-side attention shortcut.
For `TopK`, prefer the generalized `TopKBase` path instead of wiring only one opset version. The current lowering/codegen reads `get_k()`, preserves the public output index element type from output port 1, and only accepts the currently implemented sort modes. Do not reintroduce a flat `collapse_shape`/`expand_shape` TopK path: the shared builder must generate rank-aware indices from `outer/axis/inner` coordinates so Metal and Vulkan see the same logical tensor contract. On Vulkan, `i64` index outputs are packed as two `i32` shader lanes in the same public output buffer; keep that as a lowering ABI detail and do not replace it with test skips, CPU fallback, device-specific branches, or runtime environment switches. The internal lane dimension is trailing (`public_shape + [2]`), not an arbitrary doubled TopK axis, so `axis`, `inner`, and public OpenVINO layout remain independent. Long-axis `TopK` must use a bounded top-list plus final ordering step, or a future hierarchical/chunked GPU-only implementation; do not restore the old per-candidate insertion sort that made each row do axis-length serial shifting in one shader invocation. Android/RPi failures must be fixed in the shared MLIR/SPIR-V manifest path unless a backend-specific route is promoted to the same manifest contract and replaces the old path completely.

For `Gather`, keep runtime params and i64 storage on the shared custom-kernel contract. Metal `Gather` source expects `data`, `indices`, `output`, then `GatherParams`; compile-time metadata and request-time refresh must both bind the packed `GatherParams` tensor as a `RuntimeParams` extra buffer instead of accidentally reusing the OpenVINO axis input. Vulkan `Gather` must not require shader `Int64` for i64 indices or i64 data. Treat i64 buffers as two `i32` lanes at the shader boundary, preserving the public OpenVINO tensor type while doing index normalization/clamping in i32.

For `GatherElements`, keep the same manifest/runtime-param style instead of emitting a private C/Metal struct or one-off index path. The runtime-param payload is a flat `u32` tensor indexed through `GatherElementsCodegenDesc` offsets on every backend source generator. Vulkan/SPIR-V index normalization must use the narrowest safe integer width for the public indices, typically `i32`, and must not create shader-side `i64` ops for ordinary i32 index tensors.

For Slice-like work, note that the current lowering prefers `tensor.extract_slice` instead of synthesizing a `linalg.generic` copy. Shared metadata extraction in `src/mlir/slice_generic_codegen.cpp` still accepts both forms so older paths and debug flows remain readable.
At runtime, contiguous `Split` and `VariadicSplit` outputs may also alias slices of the input buffer instead of materializing new output allocations. Keep split-plan inference in `src/mlir/mlir_stage.cpp` aligned with any change that affects byte layout or view eligibility.
For `Split` and `VariadicSplit`, the runtime split metadata is owned by `plan_split_runtime_values()` in `src/mlir/gfx_stage_runtime_values.*`. Backend-specific execution code, including Vulkan compact split kernels, must consume that plan instead of re-parsing axis/length constants locally; this keeps inferred `-1` lengths, output-count validation, and alias/direct-dispatch behavior on one contract.
For `Concat`, prefer the shared `MlirStage` GPU copy-region path for contiguous buffer assembly. Vulkan must not carry a separate production-only concat SPIR-V route unless that route is promoted to the same manifest/debug/ABI contract as other custom kernels and replaces the shared path completely. Backend-local concat chunking that re-parses axis/shape state is legacy.

For `ShapeOf`, keep the runtime-materialized dims path aligned with the builder and output ABI. The current stage path wraps the resolved runtime shape into an immutable `i32` or `i64` buffer and treats it as a dedicated kernel input/output contract rather than a generic copied tensor.
Small integral host inputs used as shape/value metadata are carried through `GpuTensor::i64_values` by the shared input-binding path. Runtime planners such as `ShapeOf`, `Gather`, `Split`, and `Range` may use that side-channel to allocate outputs and build runtime-param payloads, while the actual input tensor remains bound/uploaded for GPU execution.
For `Range`, keep output sizing and compact GPU execution on one runtime-value contract. The shared planner computes scalar values/output shape from constants or small bound host metadata; Vulkan then executes `range_linear` as a compact custom kernel with `i32` lanes for supported `i32`/`i64` control tensors. Public `i64` output still uses the two-lane public buffer layout, including an empty-output sentinel allocation only to satisfy the runtime buffer-handle contract. Do not reintroduce generic shader `Int64`, CPU inference fallback, or backend-local shape parsing for this path.

For Metal kernel-dispatch work, inspect `src/backends/metal/runtime/metal_command_encoder.*` before adding new ad-hoc encoder setup code. The current runtime keeps one compute encoder per active command buffer, caches the last bound pipeline plus buffer table, and ends that encoder explicitly before blit/copy paths or command-buffer commit.
For Metal Conv2D / MaxPool work, keep dilation handling and dispatch blocking coherent across `gfx_codegen_desc.hpp`, MLIR/MSL source planning, and request-time dispatch metadata. The current code shares the same dilation and output-block metadata between compile-time codegen and runtime dispatch sizing.
Apple MPS vendor contracts are the canonical typed ABI once selected: their input/output tensor descriptors may replace the intermediate MLIR helper signature before stage-manifest and storage-bridge materialization. Apple MPS Pool2D is a single-output vendor primitive. Do not route opset8/opset14 indexed `MaxPool` through the MPS Pool2D contract until indices are represented as an explicit second vendor output; those nodes must stay on the custom-kernel path instead of producing storage-bridge errors inside MPSRT materialization.
For Metal placement or codegen work, also keep the MPSRT boundary coherent. The current code expects stage policy, MLIR attrs, `gfx_kernel_manifest.hpp`, `gfx_custom_kernel_families.*`, `src/runtime/gfx_mpsrt_model.*`, and `src/backends/metal/runtime/mpsrt/*` to agree on:
- the selected placement domain (`apple_mps` vs `apple_msl`)
- the storage kind (`image`, `matrix`, `buffer`)
- the execution kind (`vendor_primitive` vs `custom_kernel`)
- the stable stage record key
- the external-buffer ABI roles for inputs, outputs, and runtime-parameter buffers
- the dispatch grid, threadgroup size, and precompiled-binary requirement for custom MSL kernels

If the change touches manifest-backed Metal lowering, also inspect:
- `src/runtime/gfx_mpsrt_program.hpp`
- `src/runtime/gfx_mpsrt_model.*`
- `src/kernel_ir/gfx_kernel_manifest.hpp`
- `src/kernel_ir/gfx_custom_kernel_families.*`
- `src/runtime/gfx_mpsrt_kernel_manifest_adapter.hpp`
- `src/runtime/gfx_mpsrt_storage_bridge.hpp`
- `src/mlir/gfx_apple_stage_pipeline.*`
- `src/mlir/gfx_mpsrt_dialect.*`
- `src/mlir/gfx_mpsrt_ops.*`
- `src/mlir/gfx_mpsrt_source_plan.hpp`
- `src/mlir/gfx_mpsrt_const_tensor_sources.hpp`

The current MPSRT path can now represent:
- vendor-only stages such as `MPSGemm`
- custom-kernel-only MSL dispatch stages
- hybrid multi-stage plans such as `MPSGemm + MSL epilogue`
- vendor-only Conv2D, GroupConv, Pool2D, Resize2D, Softmax, TopK, and SDPA stages when Apple MPS/MPSGraph placement is selected
- MPSGraph-backed GEMM, TopK, and SDPA encodes when matrix/NDArray shapes require that vendor route
- explicit storage-conversion stages for image, matrix, ndarray, and alias contracts when external bindings do not match the internal storage class
- explicit runtime resource tables for external IO, runtime-parameter buffers, model-owned const buffers, and transient tensor/image resources

So do not assume one stage equals one dispatch source anymore. For MatMul specifically, the Metal path may choose:
- plain vendor `MPSGemm`
- vendor `MPSGemm` plus one manifest-driven MSL epilogue when bias or a supported activation is fused
- fallback custom MSL kernel compilation when the MPSRT mixed plan is not applicable

When touching the Metal MPSRT boundary, keep four layers aligned:
- the manifest and stage-family contract in `gfx_kernel_manifest.hpp`
- the typed program contract in `gfx_mpsrt_program.hpp`
- generated `gfx_mpsrt_ops` typed-program records in `gfx_mpsrt_ops.*`
- storage-bridge descriptors in `gfx_mpsrt_storage_bridge.hpp`
- runtime resource finalization in `src/runtime/gfx_mpsrt_model.*` and request-time execution/validation in `src/backends/metal/runtime/mpsrt/*`

The current request path can no longer assume one storage class for all external bindings. Image-backed Apple MPS stages may require explicit `buffer_to_image` or `image_to_buffer` bridges, and those bridges are part of the serialized builder plan and reconstructed runtime model. The request path also no longer treats unbound tensors as ad-hoc transient allocations: it binds against finalized `MpsrtRuntimeResource` entries, prepared model-owned const buffers, and heap-backed transient resources.
Also, do not build new tooling on top of stale flat `gfx.mpsrt.*` stage attrs. Current code intentionally materializes generated helpers and erases legacy attrs after the program facade is available.
For Apple-targeted lowering, also prefer the dedicated `run_gfx_apple_stage_pipeline()` path over reintroducing per-op ad-hoc metadata assembly. The current code treats placement, storage, fusion, stage-manifest lowering, and typed program materialization as one connected flow.
Do not add new `gfx.apple.pipeline.*` module attributes. That prefix is treated as legacy side-channel state and is erased when typed MPSRT ops are materialized. New Apple lowering state must flow through `GfxAppleStagePipelineOptions`, `GfxKernelStageManifest`, and the typed `GfxMpsrtProgram`; storage bridges are derived from the stage plan and serialized on `gfx_mpsrt_ops`, not on the module.
Do not add duplicated typed-stage identity attrs under `gfx.mpsrt.op.stage.backend`, `gfx.mpsrt.op.stage.stage_kind`, or `gfx.mpsrt.op.stage.stage_record_key`. Do not add storage/layout side channels under `gfx.mpsrt.op.stage.input_storage`, `gfx.mpsrt.op.stage.output_storage`, or `gfx.mpsrt.op.stage.layout`. Generated MPSRT ops are typed by their op name and validated against `gfx.stage_manifest.*`; readback must reconstruct stage kind, storage/layout, and record key from that manifest plus the explicit typed value edges and tensor descriptors.
Do not pass `stage_record_key` through in-memory builder stage specs, module stage plans, Apple MPS lowering plans, or the single-stage builder-plan API. `GfxMpsrtBuilderPlan` may carry the model-level `model_record_key`; derive each per-stage record key from `GfxMpsrtStageDesc` at the builder/runtime boundary so MLIR, program validation, and runtime records cannot drift.
For Apple MPS vendor stages, prefer adding descriptor support in `gfx_apple_vendor_descriptors.*` and threading it through `GfxAppleMpsVendorPrimitiveContract` plus `materialize_apple_mps_vendor_contract_program()` rather than duplicating per-op ABI extraction inside older metadata helpers.
For Metal custom-kernel source or request-time binding changes, route new behavior through `GfxMslRuntimeBindingPlan` and the split helpers in `src/mlir/msl_codegen_apple_msl*`, `src/mlir/msl_codegen_apple_mps.*`, `src/mlir/msl_codegen_attention.*`, `src/mlir/msl_codegen_compressed_matmul.*`, and `src/mlir/msl_codegen_matmul_*`. Apple MSL source entry-point normalization, custom-kernel binding selection, typed `gfx_mpsrt_ops` materialization, and MSL dispatch stage-manifest materialization belong in `msl_codegen_apple_msl_dispatch.*`; direct Concat/Split structural source generators belong in `msl_codegen_apple_msl_concat_split.cpp` and must use direct-IO manifest bindings, not execute-time runtime-param loops; SDPA and compressed MatMul declarations belong in their family headers. Keep `msl_codegen.cpp`/`.hpp` as the thin cross-route coordinator only. Keep the generated module operands, `GfxKernelStageManifest` external-buffer roles, inferred MSL `[[buffer(N)]]` argument count, and MPSRT `kernel_buffer_order` coherent. The request path treats a missing materialized buffer order for MSL dispatch as invalid runtime-model state.
When `GfxKernelStageManifest` contains explicit custom-kernel external buffer roles, treat that role list as the exact ABI. Do not trim runtime-parameter or metadata roles to fit a raw `KernelSource.signature` count; source signatures are compatibility hints only when the manifest uses a non-exact shape such as leading-IO.
When Apple MSL source configuration has to synthesize the binding through `make_backend_custom_kernel_source_binding_plan()`, keep that valid adapter binding even if an older `KernelSource.signature` advertises more buffers. The synthesized manifest roles become the replacement source signature.
At the Metal compile boundary, the same exact ABI must drive the compiled `KernelBindingPlan` and resolved-MSL cache key. Do not use stale `KernelSource.signature` or inferred MSL buffer-count scans to widen an MPSRT model that already has typed external-buffer roles.
If an MPSRT model exposes explicit roles, even a stale high-index `[[buffer(N)]]` declaration in generated MSL source is only a diagnostic/fallback hint; direct source scanning is allowed only for non-MPSRT MSL kernels that have no typed ABI.
Apply that rule before building the runtime model too: early `source_arg_count` normalization must not let source scans or stale signatures widen an exact manifest/typed ABI.
Keep the two counts separate: MPSRT external-buffer ABI counts only external buffer resources, while the Metal runtime binding schema uses the exact custom-kernel manifest role count and therefore includes `ScalarParam` byte arguments when the MSL entry point needs them.
Runtime-model adaptation must fail on explicit-role/count mismatches. Do not trim `RuntimeParams` or `Metadata` roles to make an older arg count fit; fix the manifest/source-plan count instead.
Once a module has generated `gfx_mpsrt_ops`, read external-buffer ABI from the typed `GfxMpsrtProgram` first. Module-level `gfx.stage_manifest.*` attrs may still exist as annotation breadcrumbs, but they are only a fallback before typed program materialization and must not override the serialized program ABI.
Use the same priority for Metal runtime schema sizing: a typed MSL dispatch stage's custom-kernel manifest supplies the exact argument count before any module-level `gfx.stage_manifest.*` attrs are consulted. If generated typed ops are present but invalid, fail or keep the typed result invalid; do not recover through stale module attrs.
At Metal compile time this is a hard failure, not a raw-MSL compatibility path:
generated `gfx_mpsrt_ops` means the runtime model must be built from the typed
program, and invalid typed readback must be fixed at lowering/materialization.
The same schema-sizing priority applies to direct Apple MSL sources before
MPSRT materialization: a valid stage-manifest external-buffer ABI is exact and
must not be widened by stale `KernelSource.signature` values.
Single-stage Apple MSL dispatch uses the same exact ABI for resource lifetime:
const tensor arguments and generated runtime-parameter buffers remain external
request-time resources. Only typed multi-stage/vendor plans with owned payloads
move constants into the model-owned prepared-resource table.
If a prebuilt/direct Apple MSL source has an exact ABI and its signature matches
that ABI, treat it as source-plan owned: do not pass it back through node-based
family source configuration, and install the supplied runtime binding with
`apply_source_plan_kernel_runtime_binding_state()`.
When a module already has an Apple MSL custom-kernel manifest, source configuration must preserve that manifest. The per-family `stage_type`/`entry_point` factory is a compatibility fallback only, not a license to overwrite exact external-buffer roles emitted by the canonical lowering path.
If the existing manifest-backed binding is valid, do not widen it to the generic family default. A smaller exact role list is still authoritative when it came from `gfx.stage_manifest.kernel.external_buffer_abi.*`.
Apple MSL helper APIs must request manifest/factory resolution through `src/mlir/gfx_backend_custom_kernel_adapter.*`. Do not add local `make_binding_plan_from_*`, backend-domain selection, specialization-prefix selection, or per-family manifest reader stacks in MSL source-plan files; those files may request a binding plan through helpers such as `make_backend_custom_kernel_source_binding_plan()`, but the backend custom-kernel adapter owns the common Apple MSL and SPIR-V custom-kernel ABI mapping.
MPSRT runtime-stage planning follows the same rule through `src/runtime/gfx_mpsrt_kernel_manifest_adapter.hpp`: it may request a resolved custom-kernel stage manifest/dispatch spec, but it must not call `make_gfx_custom_kernel_stage_plan()` directly or duplicate stage-domain/storage prefix assembly in `gfx_mpsrt_plan.hpp`.
MatMul MPSRT epilogues must also request role-based manifests from `gfx_backend_custom_kernel_adapter.*`; do not hand-write Apple MSL custom-kernel manifests or family IDs in `gfx_mpsrt_matmul_metadata.cpp`.
Generated MSL source families such as SDPA and compressed MatMul should use `configure_msl_generated_custom_kernel_source()` so entry-point normalization, source signature materialization, exact role preservation, and optional module annotation stay in one adapter-backed path.
Prebuilt/direct MSL source plans, including direct Split, must apply their resolved binding through `configure_backend_custom_kernel_source_from_binding_plan()` instead of locally repeating module annotation and signature setup.
Direct-IO and role-based source-plan helpers must set the manifest entry point to the generated kernel entry point. Do not leave those exact source plans on a generic family entry name while the MSL source and cache key use another name.
Compile-time stage routes must use the required binding helpers in `gfx_backend_custom_kernel_adapter.*` for manifest annotation. Do not spell out `make_*_binding_plan` plus `annotate_backend_custom_kernel_module_with_binding_plan()` in `mlir_stage.cpp`; that recreates a second ABI materialization boundary. Use the scalar-payload helper only when compile-time scalar values are part of the contract; use the ABI-binding helper for routes that only need the manifest and runtime buffer order.
MSL source-family files must call `require_backend_custom_kernel_source_binding()` directly. Do not reintroduce Apple-only source binding wrappers; the backend custom-kernel adapter is the single source-binding boundary for Apple MSL and SPIR-V.
For Vulkan compact-ABI changes, first express the buffer order through the shared `GfxKernelStageManifest` custom-kernel adapter, then let `src/mlir/spirv_kernel_binding_adapter.hpp` serialize the final SPIR-V attrs. If SPIR-V lowering changes the launch ABI by scalarizing or reordering operands, reconcile that final launch order back into the stage manifest when the common role model can represent it. When final scalar byte-buffer ordering still has to live in `gfx.kernel_operand_*` / `gfx.kernel_scalar_values`, runtime metadata may read those attrs only under a valid `backend_domain=spirv`, `execution_kind=CustomKernel` manifest; no-manifest legacy operand attrs must remain rejected. Cache hits must restore the lowered metadata and reconcile the manifest before runtime metadata is extracted; otherwise a cached SPIR-V binary can be executed with the pre-lowering source-plan ABI. Do not add new per-op `gfx.fixed_arg_count` shortcuts for manual Vulkan builders; that attr is not a runtime metadata source or arg-count source, and the SPIR-V lowering path must use manifest state to preserve compact memref ABI.

When debugging Android/RPi SPIR-V failures, keep the diagnostic path explicit. `run_mlir_pipeline()` reports the verifier text from the pre-SPIR-V normalization pipeline, and retry logic may rebuild from the original module only for known alloca/normalization failures. Do not turn a device failure into a test skip until the same op has been checked through the Mac build and the real Android/RPi runners. Mac results and Android/RPi results are not expected to be byte-for-byte evidence for the same implementation path: Mac may execute through Metal MPS/MSL, while Android/RPi execute through Vulkan/SPIR-V. The acceptance contract is still shared accuracy against the same reference/template outputs on each device.

For layout-cleanup work around DFL-style postprocessing tails, the current rewrite target is a value-preserving `Softmax -> MatMul -> Reshape/Transpose` form. Do not describe the older synthetic 1x1 convolution rewrite as the active implementation.

If the op needs fusion support, inspect:
- `src/transforms/`
- `src/runtime/fused_sequence_stage.*`
- fusion planning inside `src/plugin/compiled_model.cpp`

## Properties
Supported property lists are defined in:
- `src/plugin/gfx_property_lists.cpp`
- `include/openvino/gfx_plugin/properties.hpp`

If you add a property:
1. define the property key if needed
2. add it to the supported property list
3. parse and validate it in plugin or compiled-model code
4. cover it with tests

Current device-selection contract:
- `ov::available_devices` exposes numeric ids such as `"0"` instead of backend-reported human-readable names
- `ov::device::full_name` remains the right property for user-facing device naming
- `ov::device::id` should be validated against those numeric ids
- `ov::hint::inference_precision` accepts `f16`/`fp16`/`half` or `f32`/`fp32`/`float`; default GFX compilation uses `f16`
- `GFX_DIAGNOSTIC_F32_MPS_IMAGE` is diagnostic Metal placement plumbing for localizing selected `f32` image-family MPSRT routes through the normal stage planner; it must not be treated as a runtime switch or test-skip mechanism

## Debugging
Useful environment variables:
- `OV_GFX_TRACE`
- `OV_GFX_DEBUG`
- `OV_GFX_TEST_DEBUG`
- `OV_GFX_DEBUG_MSL`
- `OV_GFX_SAFE_DEBUG`
- `OV_GFX_DUMP_SPIRV_BINDINGS`
- `OV_GFX_DUMP_SPIRV_MLIR`
- `OV_GFX_DUMP_SPIRV_MLIR_FILTER`
- `OV_GFX_DUMP_MLIR_PRE_SPIRV`

These are implemented directly in the runtime and codegen sources; grep for `OV_GFX_` when adding new diagnostics.

For functional comparison against a reference backend, build and run `tests/tools/ov_gfx_compare_runner.cpp`. It is an accuracy-only tool for numeric diffs, per-op narrowing, full-graph per-op output scans, real-image input checks, golden-reference comparisons, and `GFX`-only output summaries. Useful switches now include `--reference-device`, `--reference-plugin`, `--per-op`, `--per-op-all`, `--single-op-output`, `--input-image`, `--dump-reference-dir`, `--golden-dir`, `--gfx-inference-precision`, `--diagnostic-f32-mps-image`, `--dump-gfx-profile`, `--gfx-profiling-level`, `--tinyllama-prompt-inputs`, `--per-op-input-mode`, `--per-op-generated-inputs`, `--per-op-recursive-limit`, `--per-op-recursive-trace`, and `--gfx-only`. The current tool also understands boolean tensors and prints an extra mismatch probe for `Select` failures. Full-graph `--per-op-all` scans respect the configured thresholds and emit `PER_OP_FIRST_MISMATCH` plus `PER_OP_MISMATCH` with a non-zero exit when an internal output drifts; clean scans end with `PER_OP_MATCH max_abs=... max_rel=... tolerance_violations=0`. Do not use it for performance numbers; use `benchmark_app` for that.

For YOLO26x data-dependence checks, prefer real image inputs over synthetic random tensors. Pass three or more portable RGB PPM files with repeated `--input-image` arguments; the same mechanism is staged by `bench/gfx_eval.py --input-image` for host, Android, and SSH/RPi runs. The compare runner resizes RGB to the model's rank-4 batch-1 NCHW/NHWC input and normalizes floating-point tensors to `[0, 1]`, so these inputs are accuracy gates only; performance remains a separate `benchmark_app` measurement. If a remote target cannot finish local TEMPLATE inference in the accuracy timeout, use the runner's golden-reference mode: create the reference tensors with `--dump-reference-dir` on a machine that completes `TEMPLATE`, then run the target with `--golden-dir`. This keeps the target run GPU-only for `GFX` inference while preserving the same reference outputs and image preprocessing contract; do not replace this with CPU plugin accuracy, target-local CPU fallback, or backend-specific skips.

Per-op and debug-output compare-runner modes compile GFX with `GFX_ENABLE_FUSION=false`. That is a diagnostic compile property, not a runtime define: it preserves observable OpenVINO op boundaries and prevents artificial multi-output debug graphs from exercising the production fusion pass. Normal full-graph comparisons and production compilation keep the configured fusion policy.

`--diagnostic-f32-mps-image` sets the compile property `GFX_DIAGNOSTIC_F32_MPS_IMAGE` for GFX only. It is a diagnostic path for localizing `f32` MPSImage Conv/GroupConv/Pool quality/performance through the normal stage planner and MPSRT source planner. It must not be used as a runtime switch, CPU fallback, or backend-specific skip.

`--per-op-input-mode reference|generated|gfx-recursive` controls how isolated per-op inputs are materialized. `reference` is the strict default and compiles the upstream reference subgraph; `generated` fills non-constant external inputs deterministically with the original shapes and element types; `gfx-recursive` is the only GFX upstream-materialization mode and materializes producers one stage at a time through a shared cache before comparing the isolated op against the reference backend. Host-side materialization in this tool is allowed only as compare-runner preparation of isolated test inputs from constants/static shape metadata or reference outputs; it is not a production `GFX` inference mechanism and must not be copied into plugin runtime. Production inference may use CPU work for compilation, shape planning, descriptor setup, input upload, and final output readback, but intermediate tensor execution must stay on the target GPU without CPU fallback or hidden CPU copies. `Split`/`VariadicSplit` producers are materialized inside the compare runner as host-side logical view/copy operations and prefer the model's static output contract when it exists; static `ShapeOf` and 1D shape-list `Gather` producers can also be resolved in the compare runner for recursive diagnostics. This keeps Metal, Android Vulkan, and RPi Vulkan recursive checks on the same tensor-shape contract even when upstream shape arithmetic produces zero-length diagnostic branches or a remote Vulkan driver cannot lower shader `Int64`. `--per-op-recursive-limit N` caps producer materialization when an intentionally bounded remote diagnostic is needed, and `--per-op-recursive-trace N` prints every Nth producer plus host-materialized split/shape-list views, so capped runs fail with an explicit `PER_OP_SKIPPED` reason instead of hanging inside a large upstream graph. The legacy monolithic `gfx-upstream` mode was removed because it compiled one large upstream submodel and could trip Vulkan driver/device pressure on Android/RPi before the isolated op executed; do not reintroduce it as a per-device skip, production CPU fallback, or alternate runtime route. `--per-op-generated-inputs` is a compatibility alias for `--per-op-input-mode generated`. Use generated inputs for remote-device isolation when a single op is already identified but upstream materialization is too slow for the target, for example Raspberry Pi per-op checks on a large YOLO graph. These modes keep the same isolated op and output contract, but generated inputs are isolation checks, not proof that the full upstream graph is correct. Targeted per-op `compile_skip`/`infer_skip` now fails the run through `PER_OP_SKIPPED` instead of reporting `PER_OP_MATCH`; thresholded full-graph per-op drift is preserved as `mode=per_op_mismatch`; `bench/gfx_eval.py` keeps both cases on host, Android, and SSH transports instead of treating them as transport failures.
The compare runner now also prints output ids as `friendly_name:port` and reports the worst mismatch index together with the reference and GFX values, which is useful when only one slice or token position diverges.
For targeted convolution compile-plus-infer sweeps, use the built `ov_gfx_conv_shape_bench` tool from `tests/tools/ov_gfx_conv_shape_bench.cpp`. It measures representative YOLO26x-style Conv2D shapes on `GFX`; `CPU` is available only as a separate performance orienter and must not be treated as a GFX fallback. Use `--case SUBSTRING` for bounded Mac/Android/RPi smoke runs and `--list-cases` to print the stable case names before choosing the same case across devices.

For profiling workflows, calibration artifacts, and cross-device trace correlation, use:
- `tests/tools/ov_gfx_microbench.cpp`
- `docs/MICROBENCH_SCHEMA.md`
- `docs/PROFILING_RUNBOOK.md`
- `tools/gfx_profile_runbook.py`
- `tools/gfx_microbench_smoke.py`
- `tools/gfx_calibration_diff.py`
- `tools/gfx_external_trace_summary.py`

For reuse and submission changes, prefer the focused unit tests under:
- `tests/unit/infer_submission_test.cpp`
- `tests/unit/infer_pipeline_reuse_test.cpp`
- `tests/unit/gpu_const_cache_test.cpp`
- `tests/unit/kernel_arg_reuse_test.cpp`
- `tests/unit/gpu_backend_base_test.cpp`
- `tests/unit/gfx_parallelism_test.cpp`
- `tests/unit/mlir_matmul_parallel_test.cpp`
- `tests/unit/basic_ops_internal_test.cpp`
- `tests/unit/gfx_profiling_report_test.cpp`
- `tests/unit/gfx_stage_policy_test.cpp`
- `tests/unit/runtime_subgraph_test.cpp`

`tests/unit/gfx_parallelism_test.cpp` now also covers Broadcom-oriented matmul and convolution tuning choices, while `tests/unit/gfx_stage_policy_test.cpp` continues to cover submit-window policy decisions against synthetic device-info snapshots.

`tests/unit/infer_pipeline_reuse_test.cpp` now also covers:
- prepared output-resolution plans
- reusable host output tensors for static outputs
- reuse of pre-resolved input vectors across repeated executes

## Documentation Rules For This Module
- Keep published module docs in English
- Keep module docs inside `modules/gfx_plugin/`
- Update `README.md` when user-visible behavior changes
- Update `docs/ARCHITECTURE.md` when the code-truth architecture changes
- Update `docs/TESTING.md` when test layout or commands change

## What Not To Do
- do not rely on silent CPU fallback
- do not document removed architectures as current behavior
- do not put primary module documentation only in repository-level files outside this directory
- do not leave backend-specific route changes undocumented when they affect supported shapes, layout assumptions, or profiling behavior
- do not add ad-hoc backend caches before checking whether `ImmutableGpuBufferCache` or the shared prepared-binding cache already solves the problem
- do not bypass reusable output planning when changing infer I/O paths; keep stage outputs, passthrough outputs, and constant outputs on the same documented resolution path
